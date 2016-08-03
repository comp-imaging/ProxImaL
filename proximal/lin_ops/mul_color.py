from .lin_op import LinOp
import numpy as np
import sys


class mul_color(LinOp):
    """Color transform as blockwise 3x3 transform

       mode can be "opp" (opponenent color space)
       "yuv" (yuv color space)
       3 x 3 array or matrix as general transform
    """

    def __init__(self, arg, mode):
        # General transform or predefined mode
        if isinstance(mode, np.ndarray) or isinstance(mode, np.matrix):

            # General transform
            self.TC = np.asarray(mode)

            if self.TC.shape != (3, 3):
                print >> sys.stderr, "Error, color matrix is not 3x3."
                sys.exit(1)

        else:

            # Check for predefined transforms
            if mode == 'opp':
                self.TC = np.array([[1. / 3., 1. / 3., 1. / 3.], [0.5, 0.0, -0.5],
                                   [0.25, -0.5, 0.25]], dtype=np.float32, order='F')
            elif mode == 'yuv':
                self.TC = np.array([[0.299, 0.587, 0.114],
                                    [-0.16873660714285, -0.33126339285715, 0.5],
                                    [0.5, -0.4186875, -0.0813125]],
                                   dtype=np.float32, order='F')
            else:
                print >> sys.stderr, "Error, unsupported color mode."
                sys.exit(1)

        # Check for the shape
        if len(arg.shape) != 3 or arg.shape[2] != 3:
            print >> sys.stderr, "Error, color transform needs a 3 channel image."
            sys.exit(1)

        super(mul_color, self).__init__([arg], arg.shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """

        inimg = inputs[0]
        inimg_reshaped = inimg.reshape((inimg.shape[0] * inimg.shape[1], inimg.shape[2]))
        result = np.dot(self.TC, inimg_reshaped.T).T.reshape(inimg.shape)
        np.copyto(outputs[0], result)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """

        inimg = inputs[0]
        inimg_reshaped = inimg.reshape((inimg.shape[0] * inimg.shape[1], inimg.shape[2]))
        result = np.dot(self.TC.T, inimg_reshaped.T).T.reshape(inimg.shape)
        np.copyto(outputs[0], result)

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
        return input_mags[0] * np.linalg.norm(self.TC, 2)
