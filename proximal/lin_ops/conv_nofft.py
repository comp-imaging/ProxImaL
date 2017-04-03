from ..lin_ops import lin_op
from ..utils import matlab_support
from ..utils.codegen import indent
import numpy as np
from numpy import random
import scipy.signal

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
        #print(self.kernel)
        #self.kernel = self.kernel / np.sum(np.abs(kernel))

        # Set implementation in super-class
        super(conv_nofft, self).__init__([arg], arg.shape)
        
    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("conv_nofft:forward:cuda")
        in_node = cg.input_nodes(self)[0]
        var = "var_%d" % num_tmp_vars
        idxy = "idx_%d" % (num_tmp_vars+1)
        idxx = "idx_%d" % (num_tmp_vars+2)
        num_tmp_vars += 3
        code = """/*conv_nofft*/
float %(var)s = 0.0f;
int %(idxy)s;
int %(idxx)s;
""" % locals()
        
        ski = zip(*np.unravel_index(np.argsort(self.kernel, axis=None), self.kernel.shape))
        
        for ky,kx in ski:
            kernelvalue = self.kernel[ky,kx]
            y = ky - self.kernel.shape[0]//2
            x = kx - self.kernel.shape[1]//2
            if y > 0:
                code += "%s = max(0, (%s) - %d);\n" % (idxy, abs_idx[0], y)
            elif y < 0:
                code += "%s = min(%d, (%s) - %d);\n" % (idxy, self.shape[0]-1, abs_idx[0], y)
            else:
                code += "%s = %s;\n" % (idxy, abs_idx[0])

            if x > 0:
                code += "%s = max(0, (%s) - %d);\n" % (idxx, abs_idx[1], x)
            elif x < 0:
                code += "%s = min(%d, (%s) - %d);\n" % (idxx, self.shape[1]-1, abs_idx[1], x)
            else:
                code += "%s = %s;\n" % (idxx, abs_idx[1])
            
            new_idx = [idxy, idxx] + abs_idx[2:]
            scode, svar, num_tmp_vars = in_node.forward_cuda_kernel(cg, num_tmp_vars, new_idx, self)
            code += scode
            code += "%(var)s += %(svar)s * %(kernelvalue).10e;\n" % locals()
        return code, var, num_tmp_vars
    
    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("conv_nofft:adjoint:cuda")
        in_node = cg.output_nodes(self)[0]
        var = "var_%d" % num_tmp_vars
        idxy = "idx_%d" % (num_tmp_vars+1)
        idxx = "idx_%d" % (num_tmp_vars+2)
        viy1 = "viy1_%d" % (num_tmp_vars+3)
        vix1 = "vix1_%d" % (num_tmp_vars+4)
        viy2 = "viy2_%d" % (num_tmp_vars+5)
        vix2 = "vix2_%d" % (num_tmp_vars+6)
        viy = "viy_%d" % (num_tmp_vars+7)
        vix = "vix_%d" % (num_tmp_vars+8)
        num_tmp_vars += 9
        code = """/*conv_nofft*/
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

        code += """
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
            scode, svar, num_tmp_vars = in_node.adjoint_cuda_kernel(cg, num_tmp_vars, new_idx, self)
            scode = indent(scode, 4)
            
            # zero padding in inside/outside region
            ly = self.shape[0] + y
            lx = self.shape[1] + x
            
            code += indent("""
%(idxy)s = (%(viy)s) - %(y)d;
%(idxx)s = (%(vix)s) - %(x)d;
if( %(idxy)s >= 0 && %(idxy)s < %(h)d && %(idxx)s >= 0 && %(idxx)s < %(w)d )
{
    %(scode)s
    %(var)s += %(svar)s * %(kernelvalue).10e;
}
""" % locals(), 8)
                        
        code += """
    }
}
"""
        return code, var, num_tmp_vars
        
        
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


