from __future__ import print_function

from proximal.tests.base_test import BaseTest

from proximal.lin_ops.vstack import vstack
from proximal.lin_ops.variable import Variable
from proximal.lin_ops.subsample import subsample, uneven_subsample
from proximal.lin_ops.comp_graph import CompGraph
from proximal.lin_ops.conv_nofft import conv_nofft
from proximal.lin_ops.grad import grad
from proximal.lin_ops.transpose import transpose
from proximal.lin_ops.pxwise_matrixmult import pxwise_matrixmult

from pycuda import gpuarray

import functools
import logging
import time

from numpy import random
import numpy as np

class TestCudaCompGraphs(BaseTest):

    def _generic_check_adjoint(self, f, inshape, outshape, s,
                               ntests = 50, eps=1e-5, verbose=False,
                               in_out_sample = None):
        """
        Generic tests used for all comp graph tests on a parametrizable function f
        """
        x = Variable(inshape)
        func = f(x)
        if not type(func) is tuple:
            func = (func,)
        G = CompGraph(vstack(func))

        nin = functools.reduce(lambda x,y: x*y, inshape, 1)
        nout = functools.reduce(lambda x,y: x*y, outshape, 1)

        if not in_out_sample is None:
            # check against the given in/out samples
            x1 = in_out_sample[0] # forward in
            y1s = in_out_sample[1] # forward out
            y2 = in_out_sample[2] # adjoint in
            x2s = in_out_sample[3] # adjoint out

            y1a = G.forward_cuda(gpuarray.to_gpu(x1.astype(np.float32)),gpuarray.to_gpu(y1s.astype(np.float32))).get()
            #print(y1s)
            #print(y1a)
            self.assertTrue(np.amax(np.abs(y1a-y1s)) < eps)

            x2a = G.adjoint_cuda(gpuarray.to_gpu(y2.astype(np.float32)),gpuarray.to_gpu(x2s.astype(np.float32))).get()
            self.assertTrue(np.amax(np.abs(x2a-x2s)) < eps)

        # test with random data that the forward/adjoint operators are consistent
        maxerr = 0.0
        random.seed(0) # make tests reproducable
        for tidx in range(ntests):
            x1 = random.rand(nin).astype(np.float32)
            y2 = random.rand(nout).astype(np.float32)
            y1 = np.zeros(nout, dtype=np.float32)
            x2 = np.zeros(nin, dtype=np.float32)

            if verbose: print("forward: ", end="")
            y1 = G.forward_cuda(gpuarray.to_gpu(x1),gpuarray.to_gpu(y1),printt=verbose).get()
            if verbose: print("adjoint: ", end="")
            x2 = G.adjoint_cuda(gpuarray.to_gpu(y2),gpuarray.to_gpu(x2),printt=verbose).get()

            self.assertTrue(not np.all(y1 == 0) and not np.all(x2 == 0))

            y1o = G.forward(x1,y1.copy())
            x2o = G.adjoint(y2,x2.copy())
            erro = abs(np.dot(x1.flatten().astype(np.float64), x2o.flatten().astype(np.float64)) -
                       np.dot(y1o.flatten().astype(np.float64), y2.flatten().astype(np.float64)))

            err = abs(np.dot(x1.flatten().astype(np.float64),x2.flatten().astype(np.float64)) -
                      np.dot(y1.flatten().astype(np.float64),y2.flatten().astype(np.float64)))
            if err > maxerr:
                maxerr = err
            if verbose and err > eps:
                print("forward CUDA code:")
                print(G.cuda_forward_subgraphs.cuda_code)
                print("backward CUDA code:")
                print(G.cuda_adjoint_subgraphs.cuda_code)
                print("x1\n",np.reshape(x1, inshape))
                print("y1\n",np.reshape(y1, outshape))
                print("y1o\n",np.reshape(y1o, outshape))
                print("y2\n",np.reshape(y2, outshape))
                print("x2\n",np.reshape(x2, inshape))
                print("x2o\n",np.reshape(x2o, inshape))
                print("(%d) Adjoint test (%s): gpu: %f, nogpu: %f" %(tidx,s,err,erro))
                print("max(abs(y1-y1o)): %f" % (np.amax(np.abs(y1-y1o))))
                print("max(abs(x2-x2o)): %f" % (np.amax(np.abs(x2-x2o))))
            self.assertTrue(err <= eps)
        if verbose: print("%s passed %d tests. Max adjoint test error: %f" % (s, ntests, maxerr))

    def test_scale(self):
        fin = np.arange(100)
        fout = fin*5
        self._generic_check_adjoint(lambda x: x*5, (10,10), (10,10), "scale", in_out_sample = (fin,fout,fin,fout))

    def test_sum(self):
        fin = np.arange(100)
        fout = fin*6
        self._generic_check_adjoint(lambda x: x+x*5, (10,10), (10,10), "sum", in_out_sample = (fin,fout,fin,fout))

    def test_constant(self):
        self._generic_check_adjoint(lambda x: x-random.rand(10,10), (10,10), (10,10), "constant")

    def test_subsample(self):
        fin = np.arange(100)
        fout = np.reshape(fin, (10,10))[::2,::4].flatten()
        ain = np.arange(15)
        aout = np.zeros((10,10))
        aout[::2,::4] = np.reshape(ain, (5,3))
        self._generic_check_adjoint(lambda x: subsample(x, [2,4]), (10,10), (5,3), "subsample", in_out_sample = (fin,fout,ain,aout))

    def test_uneven_subsample(self):
        fin = np.arange(100)
        fout = np.reshape(fin, (10,10))[::2,::4].flatten()
        ain = np.arange(15)
        aout = np.zeros((10,10))
        aout[::2,::4] = np.reshape(ain, (5,3))
        idx = np.indices( (5,3) )
        idx[0] *= 2
        idx[1] *= 4
        self._generic_check_adjoint(lambda x: uneven_subsample(x, idx), (10,10), (5,3), "uneven_subsample", in_out_sample = (fin,fout,ain,aout))

    def test_uneven_subsample2(self):
        # check out of range indices
        fin = np.arange(9)+1
        idxy = np.array([[0,-1,-2], [1,1,1], [2,3,4]])
        idxx = np.array([[0,1,2], [-1,1,3], [-2,1,4]])

        fout = np.array([[1,0, 0], [0, 5, 0], [0, 0, 0]])

        ain = fin
        aout = fout
        self._generic_check_adjoint(lambda x: uneven_subsample(x, [idxy,idxx]), (3,3), (3,3), "uneven_subsample2", in_out_sample = (fin,fout,ain,aout))

    def test_grad(self):
        fin = np.arange(100)
        fint = np.reshape(fin, (10,10))
        fout = np.zeros((10,10,2))
        fout[0:-1,:,0] = fint[1:,:] - fint[:-1,:]
        fout[:,0:-1,1] = fint[:,1:] - fint[:,:-1]
        fout = fout.flatten()

        ain = np.arange(200)
        aint = np.reshape(ain, (10,10,2))
        aout = np.zeros((10,10))
        aout[1:-1,:] += aint[1:-1,:,0] - aint[:-2,:,0]
        aout[0,:] += aint[0,:,0]
        aout[-1,:] += -aint[-2,:,0]
        aout[:,1:-1] += aint[:,1:-1,1] - aint[:,:-2,1]
        aout[:,0] += aint[:,0,1]
        aout[:,-1] += -aint[:,-2,1]
        aout = -aout.flatten()

        self._generic_check_adjoint(lambda x: grad(x), (10,10), (10,10,2), "grad", in_out_sample = (fin,fout, ain,aout))

    def test_conv_nofft(self):
        K = np.reshape((np.arange(9)), (3,3))

        fin = np.arange(25)
        fout = np.array([[24,45,81,117,144],[99,120,156,192,219],[279,300,336,372,399],[459,480,516,552,579],[624,645,681,717,744]])

        ain = fin
        aout = np.array([[0+8+40+88, 23+142,44+175,65+208,46+136+24+66],[105+210,312,348,384,240+111],[180+345,492,528,564,345+156],[255+480,672,708,744,450+201],[130+40+62+232,304+65,319+68,334+71,184+24+0+72]])
        self._generic_check_adjoint(lambda x: conv_nofft(K,x), (5,5), (5,5), "conv_nofft1", in_out_sample = (fin,fout,ain,aout), eps=1e-4)

        fin2 = np.zeros((5,5,2))
        fin2[:,:,0] = np.reshape(fin, (5,5))
        fin2[:,:,1] = fin2[:,:,0]
        fout2 = np.zeros((5,5,2))
        fout2[:,:,0] = np.reshape(fout, (5,5))
        fout2[:,:,1] = fout2[:,:,0]

        ain2 = np.zeros((5,5,2))
        ain2[:,:,0] = np.reshape(ain, (5,5))
        ain2[:,:,1] = ain2[:,:,0]
        aout2 = np.zeros((5,5,2))
        aout2[:,:,0] = np.reshape(aout, (5,5))
        aout2[:,:,1] = aout2[:,:,0]
        self._generic_check_adjoint(lambda x: conv_nofft(np.reshape(K, K.shape + (1,)),x), (5,5,2), (5,5,2), "conv_nofft2", in_out_sample = (fin2,fout2,ain2,aout2), eps=1e-4)

        K1 = np.array([[1,2,3]])
        K2 = np.array([[5],[3],[1]])
        fin = np.arange(25)
        fout = np.array([[159,186,240,294,339],[399,426,480,534,579],[669,696,750,804,849],[939,966,1020,1074,1119],[1059,1086,1140,1194,1239]])
        ain = fin
        aout = np.array([[55,70,100,130,95],[227,222,276,330,235],[587,492,546,600,415],[947,762,816,870,595],[1919,1514,1592,1670,1135]])
        self._generic_check_adjoint(lambda x: conv_nofft(K1,conv_nofft(K2, x)), (5,5), (5,5), "conv_nofft3", in_out_sample = (fin,fout,ain,aout), eps=1e-4)

        K = np.abs(random.rand(5,3))
        self._generic_check_adjoint(lambda x: conv_nofft(K,x), (10,10), (10,10), "conv_nofft4")

    def test_vstack(self):
        self._generic_check_adjoint(lambda x: (x, x*5), (10,10), (10*10+10*10,), "vstack")

    def test_transpose(self):
        self._generic_check_adjoint(lambda x: transpose(x, [2,0,1]), (4,3,2), (2,4,3), "transpose")

    def test_pxwise_matmul(self):
        A = random.rand(4,4,3,2)
        self._generic_check_adjoint(lambda x: pxwise_matrixmult(A, x), [4,4,2], (4,4,3), "pxwise_matmul")

        A = np.reshape(np.arange(2*2*3*2), (2,2,3,2))
        fin = np.arange(2*2*2)
        fout = np.array([[[1,3,5],[33,43,53],[113,131,149],[241,267,293]]])

        ain = np.arange(2*2*3)
        aout = np.array([[[10,13],[100,112],[298,319],[604,634]]])
        self._generic_check_adjoint(lambda x: pxwise_matrixmult(A, x), [2,2,2], [2,2,3], "pxwise_matmul2", in_out_sample = (fin,fout,ain,aout))


    def test_complex_graph(self):
        # easier debugging, start with small dimensions
        x = Variable((5,5))
        cx = conv_nofft(np.array([[1,1,1]])/3, conv_nofft(np.array([[1],[1],[1]])/3, x))
        scx = subsample(cx, (2,2))
        ed = scx - np.reshape(np.arange(3*3), (3,3))
        w = Variable(x.shape + (2,))
        gw = grad(w,2)
        Ew = gw + transpose(gw, (0,1,3,2))
        gx = grad(x,2)
        tgx = pxwise_matrixmult(np.reshape(np.arange(5*5*2*2), (5,5,2,2)), gx)
        e1 = tgx - w
        inshape = (5*5 + 5*5*2,)
        outshape = (3*3 + 5*5*2*2 + 5*5*2,)
        self._generic_check_adjoint(lambda x: (ed,e1,Ew), inshape, outshape, "complex", eps=5e-4)

        # continue with more values
        K1 = np.abs(random.rand(1,5,1))
        K2 = np.abs(random.rand(5,1,1))

        x = Variable((320,240,2))
        cx = conv_nofft(K1,conv_nofft(K2, x))
        scx = subsample(cx, (5,5,1))
        ed = scx - random.rand(64,48,2)

        w = Variable(x.shape + (2,))
        gw = grad(w, 2)
        Ew = gw + transpose(gw, (0,1,2,4,3))
        gx = grad(x,2)
        tgx = pxwise_matrixmult(random.rand(320,240,2,2,2), gx)
        e1 = tgx - w

        inshape = (320*240*2 + 320*240*2*2,)
        outshape = (64*48*2 + 320*240*2*2*2 + 320*240*2*2,)
        self._generic_check_adjoint(lambda x: (ed,e1,Ew), inshape, outshape, "complex2", eps=5e-4)

    def test_performance(self):
        c = random.rand(2000,2000)
        x = Variable([2000,2000])
        K = np.abs(random.rand(9,9))
        G = CompGraph(vstack([ subsample((conv_nofft(K, x) -c)*5, [2,4]), x*10 ]))
        xtest1 = random.rand(2000*2000).astype(np.float32)
        ytest1 = np.zeros(G.output_size, dtype=np.float32)
        t1_cpu = time.time()
        for i in range(10):
            ytest1 = G.forward(xtest1, ytest1)
        t2_cpu = time.time()

        xtest = gpuarray.to_gpu(xtest1.astype(np.float32))
        ytest = gpuarray.to_gpu(ytest1.astype(np.float32))
        t1_gpu = time.time()
        for i in range(10):
            ytest = G.forward_cuda(xtest, ytest)
        t2_gpu = time.time()

        t_cpu = t2_cpu - t1_cpu
        t_gpu = t2_gpu - t1_gpu
        logging.info("Forward timing: cpu=%.2f ms gpu=%.2f ms factor=%.3f" % (t_cpu, t_gpu, t_gpu/t_cpu))
        self.assertTrue(t_gpu < t_cpu)

        t1_cpu = time.time()
        for i in range(10):
            xtest1 = G.adjoint(ytest1, xtest1)
        t2_cpu = time.time()

        t1_gpu = time.time()
        for i in range(10):
            xtest = G.adjoint_cuda(ytest, xtest)
        t2_gpu = time.time()

        t_cpu = t2_cpu - t1_cpu
        t_gpu = t2_gpu - t1_gpu
        logging.info("Adjoint timing: cpu=%.2f ms gpu=%.2f ms factor=%.3f" % (t_cpu, t_gpu, t_gpu/t_cpu))
        self.assertTrue(t_gpu < t_cpu)

        #print( G.start.adjoint_cuda(G, 0, "i", None)[0] )

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)
    import unittest
    unittest.main()