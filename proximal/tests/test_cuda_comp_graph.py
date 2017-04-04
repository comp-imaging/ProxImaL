from proximal.lin_ops.vstack import vstack
from proximal.lin_ops.variable import Variable
from proximal.lin_ops.subsample import subsample
from proximal.lin_ops.comp_graph import CompGraph
from proximal.lin_ops.conv_nofft import conv_nofft
from proximal.lin_ops.grad import grad
from proximal.lin_ops.transpose import transpose
from proximal.lin_ops.pxwise_matrixmult import pxwise_matrixmult

import functools

from numpy import random
import numpy as np

from pycuda import gpuarray
import pycuda.compiler

pycuda.compiler.DEFAULT_NVCC_FLAGS.extend(['-ccbin', 'clang-3.8'])

def generic_check_adjoint(f, inshape, outshape, s, 
                          ntests = 50, eps=1e-5, printt=False,
                          in_out_sample = None):
    x = Variable(inshape)
    func = f(x)
    if not type(func) is tuple:
        func = (func,)
    G = CompGraph(vstack(func))
    
    nin = functools.reduce(lambda x,y: x*y, inshape, 1)
    nout = functools.reduce(lambda x,y: x*y, outshape, 1)
    
    if not in_out_sample is None:
        x1 = in_out_sample[0] # forward in
        y1s = in_out_sample[1] # forward out
        y2 = in_out_sample[2] # adjoint in
        x2s = in_out_sample[3] # adjoint out
        
        y1a = G.forward_cuda(gpuarray.to_gpu(x1.astype(np.float32)),gpuarray.to_gpu(y1s.astype(np.float32))).get()
        if not np.all(np.abs(y1a - y1s) < eps):
            print("forward CUDA code:")
            print(G.cuda_forward_subgraphs.cuda_code)
            print("backward CUDA code:")
            print(G.cuda_adjoint_subgraphs.cuda_code)
            assert(False)
            
        x2a = G.adjoint_cuda(gpuarray.to_gpu(y2.astype(np.float32)),gpuarray.to_gpu(x2s.astype(np.float32))).get()
        if not np.all(np.abs(x2a - x2s) < eps):
            print("forward CUDA code:")
            print(G.cuda_forward_subgraphs.cuda_code)
            print("backward CUDA code:")
            print(G.cuda_adjoint_subgraphs.cuda_code)
            print("x2a")
            print(x2a)
            print("x2s")
            print(x2s)
            assert(False)
    
        
    
    maxerr = 0.0
    for tidx in range(ntests):
        x1 = random.rand(nin).astype(np.float32)
        y2 = random.rand(nout).astype(np.float32)
        y1 = np.zeros(nout, dtype=np.float32)
        x2 = np.zeros(nin, dtype=np.float32)
        
        if printt: print("forward: ", end="")
        y1 = G.forward_cuda(gpuarray.to_gpu(x1),gpuarray.to_gpu(y1),printt=printt).get()
        if printt: print("adjoint: ", end="")
        x2 = G.adjoint_cuda(gpuarray.to_gpu(y2),gpuarray.to_gpu(x2),printt=printt).get()
        
        assert not np.all(y1 == 0) and not np.all(x2 == 0)
        
        y1o = G.forward(x1,y1.copy())
        x2o = G.adjoint(y2,x2.copy())
        erro = abs(np.dot(x1.flatten().astype(np.float64), x2o.flatten().astype(np.float64)) - 
                   np.dot(y1o.flatten().astype(np.float64), y2.flatten().astype(np.float64)))
        
        err = abs(np.dot(x1.flatten().astype(np.float64),x2.flatten().astype(np.float64)) - 
                  np.dot(y1.flatten().astype(np.float64),y2.flatten().astype(np.float64)))
        if err > maxerr:
            maxerr = err
        if err > eps:
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
            assert(err <= eps)
    print("%s passed %d tests. Max adjoint test error: %f" % (s, ntests, maxerr))

def check_scale():
    fin = np.arange(100)
    fout = fin*5
    generic_check_adjoint(lambda x: x*5, (10,10), (10,10), "scale", in_out_sample = (fin,fout,fin,fout))

def check_sum():
    fin = np.arange(100)
    fout = fin*6
    generic_check_adjoint(lambda x: x+x*5, (10,10), (10,10), "sum", in_out_sample = (fin,fout,fin,fout))
    
def check_constant():
    generic_check_adjoint(lambda x: x-random.rand(10,10), (10,10), (10,10), "constant")

def check_subsample():
    fin = np.arange(100)
    fout = np.reshape(fin, (10,10))[::2,::4].flatten()
    ain = np.arange(15)
    aout = np.zeros((10,10))
    aout[::2,::4] = np.reshape(ain, (5,3))
    generic_check_adjoint(lambda x: subsample(x, [2,4]), (10,10), (5,3), "subsample", in_out_sample = (fin,fout,ain,aout))
    
def check_grad():
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
    
    generic_check_adjoint(lambda x: grad(x), (10,10), (10,10,2), "grad", in_out_sample = (fin,fout, ain,aout))
    
def check_conv_nofft():
    K = np.reshape((np.arange(9)), (3,3))
    
    fin = np.arange(25)
    fout = np.array([[24,45,81,117,144],[99,120,156,192,219],[279,300,336,372,399],[459,480,516,552,579],[624,645,681,717,744]])
    
    ain = fin
    aout = np.array([[0+8+40+88, 23+142,44+175,65+208,46+136+24+66],[105+210,312,348,384,240+111],[180+345,492,528,564,345+156],[255+480,672,708,744,450+201],[130+40+62+232,304+65,319+68,334+71,184+24+0+72]])
    generic_check_adjoint(lambda x: conv_nofft(K,x), (5,5), (5,5), "conv_nofft1", in_out_sample = (fin,fout,ain,aout), eps=1e-4)  
    
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
    generic_check_adjoint(lambda x: conv_nofft(K,x), (5,5,2), (5,5,2), "conv_nofft2", in_out_sample = (fin2,fout2,ain2,aout2), eps=1e-4)  

    K1 = np.array([[1,2,3]])
    K2 = np.array([[5],[3],[1]])
    fin = np.arange(25)
    fout = np.array([[159,186,240,294,339],[399,426,480,534,579],[669,696,750,804,849],[939,966,1020,1074,1119],[1059,1086,1140,1194,1239]])
    ain = fin
    aout = np.array([[55,70,100,130,95],[227,222,276,330,235],[587,492,546,600,415],[947,762,816,870,595],[1919,1514,1592,1670,1135]])
    generic_check_adjoint(lambda x: conv_nofft(K1,conv_nofft(K2, x)), (5,5), (5,5), "conv_nofft3", in_out_sample = (fin,fout,ain,aout), eps=1e-4)      

    K = np.abs(random.rand(5,3))
    generic_check_adjoint(lambda x: conv_nofft(K,x), (10,10), (10,10), "conv_nofft4")    
    
def check_vstack():
    generic_check_adjoint(lambda x: (x, x*5), (10,10), (10*10+10*10,), "vstack")
    
def check_transpose():
    generic_check_adjoint(lambda x: transpose(x, [2,0,1]), (4,3,2), (2,4,3), "transpose" )
    
def check_pxwise_matmul():
    A = random.rand(4,4,3,2)
    generic_check_adjoint(lambda x: pxwise_matrixmult(A, x), [4,4,2], (4,4,3), "pxwise_matmul")
    
    A = np.reshape(np.arange(2*2*3*2), (2,2,3,2))
    fin = np.arange(2*2*2)
    fout = np.array([[[1,3,5],[33,43,53],[113,131,149],[241,267,293]]])
    
    ain = np.arange(2*2*3)
    aout = np.array([[[10,13],[100,112],[298,319],[604,634]]])
    generic_check_adjoint(lambda x: pxwise_matrixmult(A, x), [2,2,2], [2,2,3], "pxwise_matmul2", in_out_sample = (fin,fout,ain,aout))
    
    
def check_complex_graph():
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
    generic_check_adjoint(lambda x: (ed,e1,Ew), inshape, outshape, "complex", eps=5e-4, printt=1)    
    
    # continue with more values
    K1 = np.abs(random.rand(1,5))
    K2 = np.abs(random.rand(5,1))
    
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
    generic_check_adjoint(lambda x: (ed,e1,Ew), inshape, outshape, "complex2", eps=5e-4, printt=1)
    
if __name__ == "__main__":
    check_scale()
    check_sum()
    check_constant()
    check_subsample()
    check_conv_nofft()
    check_vstack()
    check_grad()
    check_transpose()
    check_pxwise_matmul()
    
    check_complex_graph()    
    
    c = random.rand(2000,2000)
    x = Variable([2000,2000])
    K = np.abs(random.rand(9,9))
    G = CompGraph(vstack([ subsample((conv_nofft(K, x) -c)*5, [2,4]), x*10 ]))
    #G = CompGraph(vstack([ subsample(conv_nofft(np.ones((1,1)), x), [2,4]) ]))
    print("forward code\n")
    #print( G.end.forward_cuda(G, 0, ["i"], None)[0] )
    xtest1 = random.rand(2000*2000).astype(np.float32)
    xtest2 = np.zeros(2000*2000, dtype=np.float32)
    #xtest = np.ones(G.input_size)
    ytest1 = np.zeros(G.output_size, dtype=np.float32)
    ytest2 = random.rand(G.output_size).astype(np.float32)
    
    ytest1 = G.forward_cuda(xtest1,ytest1).get()
    xtest2 = G.adjoint_cuda(ytest2,xtest2).get()
    print("adjoint test",np.dot(ytest1.astype(np.float64),ytest2.astype(np.float64)) - np.dot(xtest1.astype(np.float64),xtest2.astype(np.float64)))
    
    xtest = gpuarray.to_gpu(xtest1.astype(np.float32))
    ytest = gpuarray.to_gpu(ytest1.astype(np.float32))
    for i in range(10):
        ytest = G.forward_cuda(xtest, ytest, printt=True)
    print(ytest[1:10])
    
    
    print("\n\nadjoint code\n")
    for i in range(10):
        xtest = G.adjoint_cuda(ytest, xtest, printt=True)
    print(xtest[1:10])
    #print( G.start.adjoint_cuda(G, 0, "i", None)[0] )
    
    