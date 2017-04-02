from .lin_op import LinOp
import numpy as np
from ..utils.codegen import indent

class subsample(LinOp):
    """Samples every steps[i] pixel along axis i,
       starting with pixel 0.
    """

    def __init__(self, arg, steps):
        self.steps = self.format_shape(steps)
        self.orig_shape = arg.shape
        shape = tuple([(dim - 1) // step + 1 for dim, step in zip(arg.shape, self.steps)])
        super(subsample, self).__init__([arg], shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        # Subsample.
        selection = self.get_selection()
        np.copyto(outputs[0], inputs[0][selection])

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        # Fill in with zeros.
        selection = self.get_selection()
        outputs[0][:] *= 0
        outputs[0][selection] = inputs[0]
        
    def forward_cuda(self, cg, num_tmp_vars, abs_idx, parent):
        #print("subsample:forward:cuda")
        new_abs_idx = list(["(%s)*%d" % (ai, si) if type(ai) is str else ai*si for (ai,si) in zip(abs_idx, self.steps)])
        code, var, num_tmp_vars = cg.input_nodes(self)[0].forward_cuda(cg, num_tmp_vars, new_abs_idx, self)
        return "/*subsample*/\n"+code, var, num_tmp_vars
    
    def adjoint_cuda(self, cg, num_tmp_vars, abs_idx, parent):
        #print("subsample:adjoint:cuda")
        resvar = "var_%(num_tmp_vars)d" % locals()
        new_abs_idx = list(["(%s)/%d" % (ai, si) if type(ai) is str else ai//si for (ai,si) in zip(abs_idx, self.steps)])
        pcode, var, num_tmp_vars = cg.output_nodes(self)[0].adjoint_cuda(cg, num_tmp_vars, new_abs_idx, self)
        pcode = indent(pcode, 4)
        sel = " && ". join( ["((%s %% %d) == 0)" % (ai, si) for (ai,si) in zip(abs_idx, self.steps)] )
        code = """/*subsample*/
float %(resvar)s = 0.0f;
if( %(sel)s )
{
    %(pcode)s;
    %(resvar)s = %(var)s;
}
""" % locals()
        return code, resvar, num_tmp_vars

    def init_matlab(self, prefix):
        return "% no code\n"
        
    def forward_matlab(self, prefix, inputs, outputs):
        indices = ",".join(["1:"+str(s)+":end" for s in self.steps])
        return outputs[0] + " = " + inputs[0] + "(" + indices + ");\n"
        
    def adjoint_matlab(self, prefix, inputs, outputs):
        out = outputs[0]
        arg = inputs[0]
        shape = list(self.orig_shape)
        indices = ",".join(["1:"+str(s)+":end" for s in self.steps])
        res = """
%(out)s = zeros(%(shape)s, 'single', 'gpuArray');
%(out)s(%(indices)s) = %(arg)s;
""" % locals()        
        return res

    def get_selection(self):
        """Return a tuple of slices to index into numpy arrays.
        """
        selection = []
        for step in self.steps:
            selection.append(slice(None, None, step))
        return tuple(selection)

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return not freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        assert not freq
        var_diags = self.input_nodes[0].get_diag(freq)
        selection = self.get_selection()
        self_diag = np.zeros(self.input_nodes[0].shape)
        self_diag[selection] = 1
        for var in var_diags.keys():
            var_diags[var] = var_diags[var] * self_diag.ravel()
        return var_diags

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
        return input_mags[0]
