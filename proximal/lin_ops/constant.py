from .lin_op import LinOp
import numpy as np


class Constant(LinOp):
    """A constant.
    """

    def __init__(self, value):
        if np.isscalar(value):
            value = np.array(value)
        self._value = value
        super(Constant, self).__init__([], value.shape)

    def variables(self):
        return []

    def constants(self):
        return [self]

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        np.copyto(outputs[0], self.value)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        pass
    
    def cuda_additional_buffers(self):
        if np.all(self._value == 0.0):
            return []
        else:
            return [("constant_%d" % self.linop_id, self._value)]
    
    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        var = "var_%(num_tmp_vars)s" % locals()
        if np.all(self._value == 0.0):
            # usually the constants are zero, so we have no need to do the memory lookup
            code = "float %(var)s = 0.0f;\n" % locals()
        else:            
            cname = self.cuda_additional_buffers()[0][0]
            shape = self._value.shape
            index = "+".join(["(%s)*%d" % (ai, np.prod(shape[d+1:])) for d,ai in enumerate(abs_idx)])
            code = """/*constant*/
float %(var)s = %(cname)s[%(index)s];
""" % locals()
        
        return code, var, num_tmp_vars+1
    
    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        return None, None, num_tmp_vars
    
    @property
    def value(self):
        return self._value

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return True

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
        return {}

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
        return 0.0
