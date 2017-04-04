from ..lin_ops import lin_op
from ..utils import matlab_support
from ..utils.codegen import indent, ind2sub, sub2ind
import numpy as np
from numpy import random
import scipy.signal

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

class conv_nofft(lin_op.LinOp):
    def trunc0_2D(self, x, s):
        if s[0] == 0 and s[1] == 0:
            return x
        if s[0] == 0:
            return x[:,s[1]:-s[1]]
        if s[1] == 0:
            return x[s[0]:-s[0],:]
        return x[s[0]:-s[0],s[1]:-s[1]]
    
    def truncr(self, x, s):
        if s == 0:
            return x
        return np.concatenate((np.array([np.sum(x[:(s+1),:], 0)]), x[(s+1):-(s+1),:], np.array([np.sum(x[-(s+1):,:], 0)])), axis=0)

    def __init__(self, kernel, arg):
    
        self.truncr_2D = lambda x, s: np.transpose(self.truncr(np.transpose(self.truncr(x, s[0])), s[1]))
        self.padr_2D = lambda x, s: np.pad(x, [[s[0],s[0]], [s[1],s[1]]], mode='edge')
        self.pad0_2D = lambda x, s: np.pad(x, [[s[0],s[0]], [s[1],s[1]]], mode='constant')

        self.kernel = np.array(kernel, np.float32)
        self.cuda_source = None
        #print(self.kernel)
        #self.kernel = self.kernel / np.sum(np.abs(kernel))

        # Set implementation in super-class
        super(conv_nofft, self).__init__([arg], arg.shape)
        
    def cuda_kernel_available(self):
        # it is better to split up the comp graph, and do the convolution 
        # "manually"
        return False
                
    def gen_cuda(self):
        abs_idxt = ind2sub("yidx", self.shape)
        idxdecl = indent("".join(["int idx%d = %s;\n" % (i,s) for i,s in enumerate(abs_idxt)]), 8)
        abs_idx = ["idx%d" % i for i in range(len(self.shape))]
        height = self.shape[0]
        width = self.shape[1]
        dimy = self.size
        code = """
__global__ void conv_nofft_forward(const float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
    for( int yidx = index; yidx < %(dimy)d; yidx += stride )
    {
        %(idxdecl)s
        
        int idxx, idxy;
        float res = 0.0f;
""" % locals()

        ski = zip(*np.unravel_index(np.argsort(self.kernel, axis=None), self.kernel.shape))
        
        idxx = "idxx"
        idxy = "idxy"
        
        kcode = "\n"
        for ky,kx in ski:
            kernelvalue = self.kernel[ky,kx]
            y = ky - self.kernel.shape[0]//2
            x = kx - self.kernel.shape[1]//2
            if y > 0:
                kcode += "%s = max(0, (%s) - %d);\n" % (idxy, abs_idx[0], y)
            elif y < 0:
                kcode += "%s = min(%d, (%s) - %d);\n" % (idxy, height-1, abs_idx[0], y)
            else:
                kcode += "%s = %s;\n" % (idxy, abs_idx[0])

            if x > 0:
                kcode += "%s = max(0, (%s) - %d);\n" % (idxx, abs_idx[1], x)
            elif x < 0:
                kcode += "%s = min(%d, (%s) - %d);\n" % (idxx, width-1, abs_idx[1], x)
            else:
                kcode += "%s = %s;\n" % (idxx, abs_idx[1])
            
            new_idx = [idxy, idxx] + abs_idx[2:]
            svar = "x[%s]" % sub2ind(new_idx, self.shape)
            kcode += "res += %(svar)s * %(kernelvalue).10e;\n" % locals()
        
        code += indent(kcode, 8)
        code += """
        y[yidx] = res;
    }
}
"""

        code += """        
__global__ void conv_nofft_adjoint(const float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
    for( int yidx = index; yidx < %(dimy)d; yidx += stride )
    {
        %(idxdecl)s

""" % locals()
        var = "res"
        idxy = "idxy"
        idxx = "idxx"
        viy1 = "viy1"
        vix1 = "vix1"
        viy2 = "viy2"
        vix2 = "vix2"
        viy = "viy"
        vix = "vix"

        kcode = """
/*conv_nofft*/
float %(var)s = 0.0f;
int %(idxy)s;
int %(idxx)s;
int %(viy1)s, %(viy2)s, %(viy)s;
int %(vix1)s, %(vix2)s, %(vix)s;
""" % locals()

        abs_idxy = abs_idx[0]
        abs_idxx = abs_idx[1]
        ksyd2 = self.kernel.shape[0]//2
        ksxd2 = self.kernel.shape[1]//2
        h = self.shape[0]
        w = self.shape[1]

        kcode += """
if( %(abs_idxy)s == 0 )
{
    %(viy1)s = -%(ksyd2)d;
    %(viy2)s = 0;
} else if( %(abs_idxy)s == %(h)d-1)
{
    %(viy1)s = %(h)d-1;
    %(viy2)s = %(h)d-1+%(ksyd2)d;
} else
{
    %(viy1)s = %(abs_idxy)s;
    %(viy2)s = %(abs_idxy)s;
}

if( %(abs_idxx)s == 0 )
{
    %(vix1)s = -%(ksxd2)d;
    %(vix2)s = 0;
} else if( %(abs_idxx)s == %(w)d-1)
{
    %(vix1)s = %(w)d-1;
    %(vix2)s = %(w)d-1+%(ksxd2)d;
} else
{
    %(vix1)s = %(abs_idxx)s;
    %(vix2)s = %(abs_idxx)s;
}

for( %(viy)s = %(viy1)s; %(viy)s <= %(viy2)s; %(viy)s++ )
{
    for( %(vix)s = %(vix1)s; %(vix)s <= %(vix2)s; %(vix)s++ )
    {
""" % locals()
        
        ski = zip(*np.unravel_index(np.argsort(self.kernel, axis=None), self.kernel.shape))
        for ky,kx in ski:
            kernelvalue = self.kernel[ky,kx]
            y = -(ky - self.kernel.shape[0]//2)
            x = -(kx - self.kernel.shape[1]//2)

            new_idx = [idxy, idxx] + abs_idx[2:]
            svar = "x[%s]" % sub2ind(new_idx, self.shape)
            
            kcode += indent("""
%(idxy)s = (%(viy)s) - %(y)d;
%(idxx)s = (%(vix)s) - %(x)d;
if( %(idxy)s >= 0 && %(idxy)s < %(h)d && %(idxx)s >= 0 && %(idxx)s < %(w)d )
{
    %(var)s += %(svar)s * %(kernelvalue).10e;
}
""" % locals(), 8)
                        
        kcode += """
    }
}
y[yidx] = %(var)s;
""" % locals()
        code += indent(kcode, 8)
        code += """
    }
}
"""
        #print(code)
        try:
            self.cuda_source = code
            self.cuda_mod = SourceModule(code)
        except cuda.CompileError as e:
            print("\n".join("(%4d) %s" % (i,s) for i,s in enumerate(code.split("\n"))))
            print("CUDA compilation error:")
            print(e.stderr)
            raise e

        cuda_forward_func = self.cuda_mod.get_function("conv_nofft_forward")
        block = (min(int(dimy), cuda_forward_func.MAX_THREADS_PER_BLOCK), 1, 1)
        grid = (int(dimy)//block[0],1,1)
        setattr(self, "cuda_conv_nofft_forward", lambda *args: cuda_forward_func(*args, grid=grid, block=block, time_kernel=True))                    
    
        cuda_adjoint_func = self.cuda_mod.get_function("conv_nofft_adjoint")
        block = (min(int(dimy), cuda_adjoint_func.MAX_THREADS_PER_BLOCK), 1, 1)
        grid = (int(dimy)//block[0],1,1)
        setattr(self, "cuda_conv_nofft_adjoint", lambda *args: cuda_adjoint_func(*args, grid=grid, block=block, time_kernel=True))                    

    def forward_cuda(self, inputs, outputs):
        if self.cuda_source is None:
            self.gen_cuda()
        self.cuda_conv_nofft_forward(inputs[0], outputs[0])
            
    def adjoint_cuda(self, inputs, outputs):
        if self.cuda_source is None:
            self.gen_cuda()
        self.cuda_conv_nofft_adjoint(inputs[0], outputs[0])
        
    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        arg = inputs[0]
        if len(arg.shape) == 2:
            ts = [s//2 for s in self.kernel.shape]
            padded = self.padr_2D(arg, ts)
            convolved = scipy.signal.convolve2d(padded, self.kernel, mode='same')
            truncated = self.trunc0_2D(convolved, ts)
            np.copyto(outputs[0], truncated)
        else:
            res = np.zeros(arg.shape)
            res_t = np.zeros(arg.shape[:2])
            for i in range(arg.shape[2]):
                self.forward([arg[:,:,i]], [res_t])
                res[:,:,i] = res_t
            np.copyto(outputs[0], res)
        
        #print("myconv:forward", arg, outputs[0])

    def adjoint(self, outputs, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        arg = outputs[0]
        if len(arg.shape) == 2:
            ts = [s//2 for s in self.kernel.shape]
            padded = self.pad0_2D(arg, ts)
            convolved = scipy.signal.convolve2d(padded, self.kernel[::-1, ::-1], mode='same')
            truncated = self.truncr_2D(convolved, ts)
            np.copyto(inputs[0], truncated)
        else:
            res = np.zeros(arg.shape)
            res_t = np.zeros(arg.shape[:2])
            for i in range(arg.shape[2]):
                self.adjoint([arg[:,:,i]], [res_t])
                res[:,:,i] = res_t
            np.copyto(inputs[0], res)
        #print("myconv:adjoint", arg, inputs[0])

    def init_matlab(self, prefix):
        matlab_support.put_array(prefix + "_kernel_raw", self.kernel, globalvar = True)
        res  = "global %(prefix)s_kernel_raw;" % locals()
        res += "obj.d.%(prefix)s_kernel = gpuArray(%(prefix)s_kernel_raw);\n" % locals()
        res += "obj.d.%(prefix)s_truncy = @(c, s) cat(1, sum(c(1:(1+s),:),1), c((1+s+1):(end-s-1),:), sum(c((end-s):end,:),1));\n" % locals()
        res += "obj.d.%(prefix)s_truncx = @(c, s) cat(2, sum(c(:,1:(1+s)),2), c(:,(1+s+1):(end-s-1)), sum(c(:,(end-s):end),2));\n" % locals()
        return res
            
    def forward_matlab(self, prefix, inputs, outputs):
        arg = inputs[0]
        out = outputs[0]

        ts = str([s//2 for s in self.kernel.shape])
        
        res = """
if numel(size(%(arg)s)) == 2
    padded = padarray(%(arg)s, %(ts)s, 'replicate');
    %(out)s = conv2(padded, obj.d.%(prefix)s_kernel, 'valid');
else
    padded = padarray(%(arg)s, [%(ts)s, 0], 'replicate');
    %(out)s = zeros(size(%(arg)s), 'single', 'gpuArray');
    for i=1:size(%(arg)s, 3)
        %(out)s(:,:,i) = conv2(padded(:,:,i), obj.d.%(prefix)s_kernel, 'valid');
    end
end
clear padded;
""" % locals()
        return res
        
    def adjoint_matlab(self, prefix, outputs, inputs):
        arg = outputs[0]
        out = inputs[0]
        
        ts = list([s//2 for s in self.kernel.shape])
        ts1 = ts[0]
        ts2 = ts[1]
        ts1p1 = ts1+1
        ts2p1 = ts2+1
        
        ts = str(ts)
        
        res = """
if numel(size(%(arg)s)) == 2
    padded = padarray(%(arg)s, %(ts)s, 0);
    c = conv2(padded, obj.d.%(prefix)s_kernel(end:-1:1, end:-1:1), 'same');
    t = obj.d.%(prefix)s_truncy(c, %(ts1)s);
    %(out)s = obj.d.%(prefix)s_truncx(t, %(ts2)s);
else
    padded = padarray(%(arg)s, [%(ts)s, 0], 0);
    %(out)s = zeros(size(%(arg)s), 'single', 'gpuArray');
    for i=1:size(%(arg)s, 3)
        c = conv2(padded(:,:,i), obj.d.%(prefix)s_kernel(end:-1:1, end:-1:1), 'same');
        t = obj.d.%(prefix)s_truncy(c, %(ts1)s);
        %(out)s(:,:,i) = obj.d.%(prefix)s_truncx(t, %(ts2)s);
    end
end
clear padded;
""" % locals()
        return res

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        res = np.sum(np.abs(self.kernel))*input_mags[0]
        #print("norm_bound=", res)
        return res


