from .lin_op import LinOp
import numpy as np


class subsample(LinOp):
    """Samples every steps[i] pixel along axis i,
       starting with pixel 0.
    """

    def __init__(self, arg, steps):
        self.steps = self.format_shape(steps)
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
