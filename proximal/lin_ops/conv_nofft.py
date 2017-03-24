from ..lin_ops import lin_op
from ..utils import matlab_support
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

        if 1: self.check_adjoints(arg)
        
        # Set implementation in super-class
        super(conv_nofft, self).__init__([arg], arg.shape)
        
    def check_adjoints(self, arg):
        # check forward backward operator
        x = random.rand(*arg.shape)
        y = random.rand(*arg.shape)
        
        yt = np.zeros(arg.shape)
        xt = np.zeros(arg.shape)
        self.forward( [x], [yt] )
        self.adjoint( [y], [xt] )
        
        r = abs(np.dot( np.ravel(x), np.ravel(xt) ) - np.dot( np.ravel(y), np.ravel(yt) ))
        if r > 1e-6:
            raise RuntimeError("Adjoint test of myconv failed: " + str(r))
        else:
            print("Adjoint test of myconv passed: " + str(r))
            
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
        res += "obj.d.%(prefix)s_truncy = @(c, s) [ sum(c(1:(1+s),:),1); c((1+s+1):(end-s-1),:); sum(c((end-s):end,:),1)];\n" % locals()
        res += "obj.d.%(prefix)s_truncx = @(c, s) [ sum(c(:,1:(1+s)),2)  c(:,(1+s+1):(end-s-1))  sum(c(:,(end-s):end),2)];\n" % locals()
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
""" % locals()
        return res
        
    def adjoint_matlab(self, prefix, outputs, inputs):
        arg = outputs[0]
        out = inputs[0]
        
        ts = list([s//2 for s in self.kernel.shape])
        ts1 = ts[0]
        ts2 = ts[1]
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


