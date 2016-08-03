from .lin_op import LinOp


def LinOpFactory(input_shape, output_shape, forward, adjoint, norm_bound=None):
    """Returns a function to generate a custom LinOp.

    Parameters
    ----------
    input_shape : tuple
        The dimensions of the input.
    output_shape : tuple
        The dimensions of the output.
    forward : function
        Applies the operator to an input array and writes to an output.
    adjoint : function
        Applies the adjoint operator to an input array and writes to an output.
    norm_bound : float, optional
        An upper bound on the spectral norm of the operator.
    """
    def get_black_box(arg):
        return BlackBox(arg, input_shape, output_shape,
                        forward, adjoint, norm_bound)
    return get_black_box


class BlackBox(LinOp):
    """A black-box lin op specified by the user.
    """

    def __init__(self, arg, input_shape, output_shape,
                 forward, adjoint, norm_bound=None):
        assert arg.shape == input_shape
        self._forward = forward
        self._adjoint = adjoint
        self._norm_bound = norm_bound
        super(BlackBox, self).__init__([arg], output_shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        self._forward(inputs[0], outputs[0])

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        self._adjoint(inputs[0], outputs[0])

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
        if self._norm_bound is None:
            return super(BlackBox, self).norm_bound(input_mags)
        else:
            return self._norm_bound * input_mags[0]
