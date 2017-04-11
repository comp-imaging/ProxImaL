from .lin_op import LinOp
from ..utils.cuda_codegen import ReverseInOut
import numpy as np


class sum(LinOp):
    """Sums its inputs.
    """

    def __init__(self, input_nodes, implem=None):
        shape = input_nodes[0].shape
        super(sum, self).__init__(input_nodes, shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        if len(inputs) > 1:
            np.copyto(outputs[0], np.sum(inputs, 0))
        else:
            np.copyto(outputs[0], inputs[0])

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        for output in outputs:
            np.copyto(output, inputs[0])
            
    def forward_cuda_kernel(self, cg, num_tmp_variables, abs_idx, parent):
        #print("sum:forward:cuda")
        input_nodes = cg.input_nodes(self)
        res = "var_%d" % num_tmp_variables
        num_tmp_variables += 1
        code = "float %(res)s = 0; /*sum/copy*/ \n" % locals()
        for n in input_nodes:
            icode, ivar, num_tmp_variables = n.forward_cuda_kernel(cg, num_tmp_variables, abs_idx, self)
            code += icode
            code += "%(res)s += %(ivar)s;\n" % locals()
        return code, res, num_tmp_variables 
    
    def adjoint_cuda_kernel(self, cg, num_tmp_variables, abs_idx, parent):
        #print("sum:adjoint:cuda")
        code, var, num_tmp_variables = cg.output_nodes(self)[0].adjoint_cuda_kernel(cg, num_tmp_variables, abs_idx, self)
        return code, var, num_tmp_variables

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return all([arg.is_diag(freq) for arg in self.input_nodes])

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
        var_diags = {var: np.zeros(var.size) for var in self.variables()}
        for arg in self.input_nodes:
            arg_diags = arg.get_diag(freq)
            for var, diag in arg_diags.items():
                var_diags[var] = var_diags[var] + diag
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
        return np.sum(input_mags)


class copy(sum):

    def __init__(self, arg, implem=None):
        if type(arg) is tuple:
            self.shape = arg
            self.input_nodes = []
        elif isinstance(arg, LinOp):
            super(copy,self).__init__([arg], arg.shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        super(copy, self).adjoint(inputs, outputs)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        super(copy, self).forward(inputs, outputs)
        
    def forward_cuda_kernel(self, cg, num_tmp_variables, abs_idx, parent):
        #print("copy:forward:cuda")
        return super(copy, self).adjoint_cuda_kernel(ReverseInOut(cg), num_tmp_variables, abs_idx, parent)
        
    def adjoint_cuda_kernel(self, cg, num_tmp_variables, abs_idx, parent):
        #print("copy:adjoint:cuda")
        return super(copy, self).forward_cuda_kernel(ReverseInOut(cg), num_tmp_variables, abs_idx, parent)

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
