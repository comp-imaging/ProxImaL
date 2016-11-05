
from .lin_op import LinOp
import numpy as np


class transpose(LinOp):
    """Permute axes.
    """

    def __init__(self, arg, axes):
        self.axes = axes
        self.inverse = range(len(self.axes))
        for idx, i in enumerate(self.axes):
            self.inverse[idx] = i
        super(transpose, self).__init__([arg], tuple(arg.shape[i] for i in axes))

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        shaped_input = np.transpose(inputs[0], self.axes)
        np.copyto(outputs[0], shaped_input)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        shaped_input = np.transpose(inputs[0], self.inverse)
        np.copyto(outputs[0], shaped_input)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return self.input_nodes[0].is_diag(freq)

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
        return self.input_nodes[0].get_diag(freq)

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
