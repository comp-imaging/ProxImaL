from .lin_op import LinOp
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


class mul_elemwise(LinOp):
    """Elementwise multiplication weight*X with a fixed constant.
    """

    def __init__(self, weight, arg, implem=None):
        assert arg.shape == weight.shape
        self.weight = weight
        shape = arg.shape

        # Halide temp
        if len(shape) in [2, 3]:
            self.weight = np.asfortranarray(self.weight.astype(np.float32))
            self.tmpout = np.zeros(arg.shape, dtype=np.float32, order='F')

        super(mul_elemwise, self).__init__([arg], shape, implem)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        if self.implementation == Impl['halide'] and (len(self.shape) in [2, 3]):

            # Halide implementation
            tmpin = np.asfortranarray(inputs[0].astype(np.float32))
            Halide('A_mask.cpp').A_mask(tmpin, self.weight, self.tmpout)  # Call
            np.copyto(outputs[0], self.tmpout)

        else:
            # Numpy
            np.multiply(inputs[0], self.weight, outputs[0])

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        self.forward(inputs, outputs)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
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
        self_diag = np.reshape(self.weight, self.size)
        for var in var_diags.keys():
            var_diags[var] = var_diags[var] * self_diag
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
        return np.max(np.abs(self.weight)) * input_mags[0]
