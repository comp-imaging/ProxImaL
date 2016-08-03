from .lin_op import LinOp
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

    def __init__(self, shape, implem=None):
        self.shape = shape
        self.input_nodes = []

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
