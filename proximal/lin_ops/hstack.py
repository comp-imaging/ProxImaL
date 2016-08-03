from .lin_op import LinOp
import numpy as np


class hstack(LinOp):
    """Horizontally concatenates vector inputs.
    """

    def __init__(self, input_nodes, implem=None):
        height = input_nodes[0].size
        width = len(input_nodes)
        super(hstack, self).__init__(input_nodes, (height, width))

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        for idx, input_data in enumerate(inputs):
            outputs[0][:, idx] = input_data.flatten()

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        for idx, output_data in enumerate(outputs):
            data = inputs[0][:, idx]
            output_data[:] = np.reshape(data, output_data.shape)

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return all([arg.is_gram_diag(freq) for arg in self.input_nodes])

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
                var_diags[var] = var_diags[var] + diag * np.conj(diag)
        # Get (A^TA)^{1/2}
        for var in self.variables():
            var_diags[var] = np.sqrt(var_diags[var])
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
        return np.linalg.norm(input_mags, 2)
