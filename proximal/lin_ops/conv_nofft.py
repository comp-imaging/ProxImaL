from ..lin_ops import lin_op
from ..utils.codegen import indent, ind2sub, sub2ind
from ..utils import codegen
import numpy as np
import scipy.ndimage.filters

class conv_nofft(lin_op.LinOp):
    def trunc0_ND(self, x, s):
        slices = [slice(s[i],x.shape[i]-s[i]) for i in range(len(s))]
        return x[slices]
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
        return np.concatenate((np.array([np.sum(x[:(s+1),...], 0)]), x[(s+1):-(s+1),...], np.array([np.sum(x[-(s+1):,...], 0)])), axis=0)
    
    def truncr_ND(self, x, s):
        for a in range(len(s)):
            t = list(range(len(s)))
            t[0] = a
            t[a] = 0
            x = np.transpose(self.truncr(np.transpose(x, t), s[a]), t)
        return x

    def __init__(self, kernel, arg):
    
        #self.truncr_2D = lambda x, s: np.transpose(self.truncr(np.transpose(self.truncr(x, s[0])), s[1]))
        self.padr_ND = lambda x, s: np.pad(x, [ [s[i],s[i]] for i in range(len(s)) ], mode='edge')
        self.pad0_ND = lambda x, s: np.pad(x, [ [s[i],s[i]] for i in range(len(s)) ], mode='constant')

        self.kernel = np.array(kernel, np.float32)
        self.cuda_source = None
        if len(kernel.shape) != len(arg.shape):
            raise RuntimeError("number of kernel dimensions must be equal to arg dimensions, pad with 1's if necessary")
        
        for d in range(len(kernel.shape)):
            if kernel.shape[d] > arg.shape[d]:
                raise RuntimeError("kernel is larger than argument, this is unsupported.")
                
        # Set implementation in super-class
        super(conv_nofft, self).__init__([arg], arg.shape)
        
    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        arg = inputs[0]
        ts = [s//2 for s in self.kernel.shape]
        padded = self.padr_ND(arg, ts)
        convolved = scipy.ndimage.filters.convolve(padded, self.kernel, mode='constant', cval=0.0)
        truncated = self.trunc0_ND(convolved, ts)
        np.copyto(outputs[0], truncated)
        
        #print("myconv:forward", arg, outputs[0])

    def adjoint(self, outputs, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        arg = outputs[0]
        ts = [s//2 for s in self.kernel.shape]
        padded = self.pad0_ND(arg, ts)
        convolved = scipy.ndimage.filters.convolve(padded, self.kernel[::-1, ::-1], mode='constant', cval=0.0)
        truncated = self.truncr_ND(convolved, ts)
        np.copyto(inputs[0], truncated)
        #print("myconv:adjoint", arg, inputs[0])

    def cuda_kernel_available(self):
        # it is better to split up the comp graph, and do the convolution 
        # on dedicated input/output buffers than in-place in the comp graph
        return False
    
    def _gen_cuda_inner(self, func_name, kernel):
        """generate a cuda kernel for the inner part of the convolution with <kernel>
        which doesn't need the extra code for the borders"""
        # for being most generic we don't use the x/y/z dimensions of cuda, but
        # we calculate that for ourselfs
        
        inner_shape = [d-((k//2)*2) for d,k in zip(self.shape, kernel.shape)]
        dimy = int(np.prod(inner_shape))
        offsets = [k//2 for k in kernel.shape]
        strides = [int(np.prod(self.shape[i+1:])) for i in range(len(self.shape))]
        iidx = ind2sub("yidx", inner_shape)
        gidx = ["(%s) + %d" % (i, o) for i,o in zip(iidx, offsets)]
        idxdecl = indent("".join(["int idx%d = %s;\n" % (i,s) for i,s in enumerate(gidx)]), 8)
        linidx = indent("int linidx = %s;\n" % ("+".join(["idx%d * %d" % (i,s) for i,s in enumerate(strides)])), 8)
        
        code = """
__global__ void %(func_name)s(const float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
    for( int yidx = index; yidx < %(dimy)d; yidx += stride )
    {
        %(idxdecl)s
        %(linidx)s
        float res = 0.0f;
""" % locals()
        
        ki = [0] * len(kernel.shape)
        while ki[0] < kernel.shape[0]:
            kernelvalue = codegen.float_constant(kernel.item(tuple(ki)))
            relcoord = [i-o for i,o in zip(ki,offsets)]
            linoff = sum([c*s for c,s in zip(relcoord,strides)])
            
            code += indent("res += %(kernelvalue)s * x[linidx - (%(linoff)s)];\n" % locals(), 8)
            
            ki[-1] += 1
            d = len(ki)-1
            while d > 0 and ki[d] >= kernel.shape[d]:
                ki[d] = 0
                d -= 1
                ki[d] += 1
        code += """
        y[linidx] = res;
    }
}""" % locals()
        return code, int(np.prod(inner_shape))
    
    def _gen_cuda_outer(self, func_name, generator):
        """generate a cuda kernel for the outer part of the convolution using <generator>
        as a generator for the inner part (different generators are used for forward and
        adjoint)"""
        
        # we have to generate the indices 
        #   (0:p(1)-1,...), (s(1)-p(1):end,...)
        #   (:,0:p(2)-1,...), (:,s(2)-p(2):end,...)
        #   (:,:,0:p(3)-1,...), (:,:,s(3)-p(3):end,...)
        #   ... given that p = kernel.shape//2 with <kernel>
        
        borders = [] # a list of processing blocks, given as (start,stop) indices in the original array (self.shape)
        kernel = self.kernel
        for d in range(len(kernel.shape)):
            if kernel.shape[d] > 1:
                borders.append([(kernel.shape[dd]//2, self.shape[dd]-kernel.shape[dd]//2) for dd in range(d)] + [(0,kernel.shape[d]//2)]                            + [(0,self.shape[dd]) for dd in range(d+1,len(kernel.shape))])
                borders.append([(kernel.shape[dd]//2, self.shape[dd]-kernel.shape[dd]//2) for dd in range(d)] + [(self.shape[d]-kernel.shape[d]//2, self.shape[d])] + [(0,self.shape[dd]) for dd in range(d+1,len(kernel.shape))])
        dimy = sum(int(np.prod([cb[1]-cb[0] for cb in b])) for b in borders)
        absidx_decl = "int " + ", ".join(["idx%d"%i for i in range(len(self.shape))])
        code = """
__global__ void %(func_name)s(const float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
    int relidx;
    float res;
    %(absidx_decl)s;
    for( int yidx = index; yidx < %(dimy)d; yidx += stride )
    {
        res = 0.0f;
""" % locals()
        bcode = []
        offset = 0
        for currb in borders:
            bshape = [cb[1] - cb[0] for cb in currb]
            nelem = int(np.prod(bshape))
            
            idx = codegen.ind2sub("relidx", bshape)
            idxdecl = indent("\n".join(["idx%d = %d + (%s);" % (i,currb[i][0], s) for i,s in enumerate(idx)]), 12)
            
            icode = indent(generator(["idx%d" % i for i in range(len(self.shape))]), 12)
            
            bcode.append("""
        if( yidx >= %(offset)d && yidx < %(offset)d + %(nelem)d )
        {
            relidx = yidx - %(offset)d;
            %(idxdecl)s
            %(icode)s
        }
""" % locals())
            offset += nelem
        code += "        else\n".join(bcode)
        code += """
    }
}"""
        return code, offset

    def _replicate_outer_generator(self, idx,kernel):
        code = """
res = 0.0f;
"""
        ki = [0] * len(kernel.shape)
        offsets = [k//2 for k in kernel.shape]
        while ki[0] < kernel.shape[0]:
            kernelvalue = codegen.float_constant(kernel.item(tuple(ki)))
            relcoord = [i-o for i,o in zip(ki, offsets)]
            sidx = []
            for i,r in enumerate(relcoord):
                if r > 0:
                    sidx.append("max(0, (%s) - %d)" % (idx[i], r))
                elif r < 0:
                    sidx.append("min(%d, (%s) - %d)" % (self.shape[i]-1, idx[i], r))
                else:
                    sidx.append("(%s) - %d" % (idx[i], r))
            linidx = sub2ind(sidx, self.shape)
            code += "res += %(kernelvalue)s * x[%(linidx)s];\n" % locals()
            ki[-1] += 1
            d = len(ki)-1
            while d > 0 and ki[d] >= kernel.shape[d]:
                ki[d] = 0
                d -= 1
                ki[d] += 1
        linidx = sub2ind(idx, self.shape)
        code += "y[%s] = res;\n" % linidx
        return code
                  
    def _zerosum_outer_generator(self, idx, kernel):
        idxs = []
        idxe = []
        idxl = []
        for d in range(len(idx)):
            i = idx[d]
            p = kernel.shape[d]//2
            w = self.shape[d]-1
            idxs.append("int idxs_%(d)d = ((%(i)s) == 0) ? -%(p)d : (((%(i)s) == %(w)d) ? %(w)d         : %(i)s);" % locals())
            idxe.append("int idxe_%(d)d = ((%(i)s) == 0) ? 0      : (((%(i)s) == %(w)d) ? %(w)d + %(p)d : %(i)s);" % locals())
            idxl.append("int idxl_%(d)d;" % locals())
        aidxs = indent("\n".join(idxs), 4)
        aidxe = indent("\n".join(idxe), 4)
        aidxl = indent("\n".join(idxl), 4)
        
        code = """
{
    %(aidxl)s;
    %(aidxs)s;
    %(aidxe)s;

    res = 0.0f;
""" % locals()
        
        for d in range(len(idx)):
            code += "for(idxl_%(d)d = idxs_%(d)d; idxl_%(d)d <= idxe_%(d)d; idxl_%(d)d++)\n" % locals()
        code += """
    {
"""
        ki = [0] * len(kernel.shape)
        offsets = [k//2 for k in kernel.shape]
        cidx = ["idxl_%d" % d for d in range(len(self.shape))]
        while ki[0] < kernel.shape[0]:
            kernelvalue = codegen.float_constant(kernel.item(tuple(ki)))
            relcoord = [i-o for i,o in zip(ki, offsets)]
            sidx = []
            for i,r in enumerate(relcoord):
                sidx.append("(%s) - %d" % (cidx[i], r))
            svar = "x[%s]" % sub2ind(sidx, self.shape)
            valididx = ["(%s >= 0) && (%s < %d)" %(s, s, w) for s,w in zip(sidx, self.shape)]
            valididx = " && ".join(valididx)
            
            code += indent("""
if( %(valididx)s )
{
    res += %(svar)s * %(kernelvalue)s;
}
""" % locals(), 8)
            ki[-1] += 1
            d = len(ki)-1
            while d > 0 and ki[d] >= kernel.shape[d]:
                ki[d] = 0
                d -= 1
                ki[d] += 1
        linidx = sub2ind(idx, self.shape)
        code += """        
    }
    y[%(linidx)s] = res;
}
""" % locals()
        return code
                
    def gen_cuda(self):
        slices = []
        for d in range(len(self.kernel.shape)):
            slices.append(slice(None,None,-1))
        kernelM = self.kernel[tuple(slices)]
        code1, n1 = self._gen_cuda_inner("cuda_conv_nofft_forward_inner", self.kernel)
        code2, n2 = self._gen_cuda_outer("cuda_conv_nofft_forward_outer", lambda x: self._replicate_outer_generator(x, self.kernel) )
        code3, n3 = self._gen_cuda_inner("cuda_conv_nofft_adjoint_inner", kernelM)
        code4, n4 = self._gen_cuda_outer("cuda_conv_nofft_adjoint_outer", lambda x: self._zerosum_outer_generator(x, kernelM))
        
        #print(code1+code2+code3+code4)
        try:
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule            
            self.cuda_source = code1 + code2 + code3 + code4
            self.cuda_mod = SourceModule(self.cuda_source)
        except cuda.CompileError as e:
            print("\n".join("(%4d) %s" % (i,s) for i,s in enumerate(self.cuda_source.split("\n"))))
            print("CUDA compilation error:")
            print(e.stderr)
            raise e

        cuda_forward_func_inner = self.cuda_mod.get_function("cuda_conv_nofft_forward_inner")
        cuda_forward_func_outer = self.cuda_mod.get_function("cuda_conv_nofft_forward_outer")
        block1 = (min(n1, cuda_forward_func_inner.MAX_THREADS_PER_BLOCK), 1, 1)
        grid1 = (n1//block1[0],1,1)
        block2 = (min(n2, cuda_forward_func_outer.MAX_THREADS_PER_BLOCK), 1, 1)
        grid2 = (n2//block2[0],1,1)
        cuda_forward_func = lambda *args: cuda_forward_func_inner(*args, grid=grid1, block=block1, time_kernel=True) + cuda_forward_func_outer(*args, grid=grid2, block=block2, time_kernel=True)
        setattr(self, "cuda_conv_nofft_forward", cuda_forward_func)
    
        cuda_adjoint_func_inner = self.cuda_mod.get_function("cuda_conv_nofft_adjoint_inner")
        block3 = (min(n3, cuda_adjoint_func_inner.MAX_THREADS_PER_BLOCK), 1, 1)
        grid3 = (n3//block3[0],1,1)
        cuda_adjoint_func_outer = self.cuda_mod.get_function("cuda_conv_nofft_adjoint_outer")
        block4 = (min(n4, cuda_adjoint_func_outer.MAX_THREADS_PER_BLOCK), 1, 1)
        grid4 = (n4//block4[0],1,1)
        cuda_adjoint_func = lambda *args: cuda_adjoint_func_inner(*args, grid=grid3, block=block3, time_kernel=True) + cuda_adjoint_func_outer(*args, grid=grid4, block=block4, time_kernel=True)
        setattr(self, "cuda_conv_nofft_adjoint", cuda_adjoint_func)                    

    def forward_cuda(self, inputs, outputs):
        if self.cuda_source is None:
            self.gen_cuda()
        self.cuda_conv_nofft_forward(inputs[0], outputs[0])
            
    def adjoint_cuda(self, inputs, outputs):
        if self.cuda_source is None:
            self.gen_cuda()
        self.cuda_conv_nofft_adjoint(inputs[0], outputs[0])
        
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


