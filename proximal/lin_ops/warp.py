from .lin_op import LinOp
import numpy as np
import cv2

from proximal.halide.halide import Halide
from proximal.utils.utils import Impl


class warp(LinOp):
    """Warp using a homography.
    """

    def __init__(self, arg, H, implem=None):
        self.H = H.copy()

        # Compute inverse
        self.Hinv = np.zeros(H.shape)
        if len(H.shape) > 2:
            for j in range(self.H.shape[2]):
                self.Hinv[:, :, j] = np.linalg.pinv(H[:, :, j])
        else:
            self.Hinv = np.linalg.pinv(H)

        # Check for the shape
        if len(H.shape) < 2 or len(H.shape) > 3:
            raise Exception(
                'Error, warp supports only up to 4d inputs (expects first 3 to be image).')

        # Has to have third dimension
        if len(arg.shape) != 3:
            raise Exception('Images must have third dimension')

        shape = arg.shape
        if len(H.shape) == 3:
            shape += (H.shape[2],)

        # Temp array for halide
        self.tmpfwd = np.zeros((shape[0], shape[1],
                                shape[2] if (len(shape) > 2) else 1,
                                H.shape[2] if (len(H.shape) > 2) else 1),
                               dtype=np.float32, order='FORTRAN')
        self.tmpadj = np.zeros((shape[0], shape[1], shape[2] if (
            len(shape) > 2) else 1), dtype=np.float32, order='FORTRAN')

        # Halide homographies
        if len(H.shape) == 2:
            self.Hf = np.asfortranarray(H[..., np.newaxis].astype(
                np.float32))  # Third axis for halide
            self.Hinvf = np.asfortranarray(
                self.Hinv[..., np.newaxis].astype(np.float32))  # Third axis for halide
        else:
            self.Hf = np.asfortranarray(H.astype(np.float32))
            self.Hinvf = np.asfortranarray(self.Hinv.astype(np.float32))

        super(warp, self).__init__([arg], shape, implem)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """

        if self.implementation == Impl['halide']:

            # Halide implementation
            tmpin = np.asfortranarray(inputs[0].astype(np.float32))
            Halide('A_warp.cpp').A_warp(tmpin, self.Hinvf, self.tmpfwd)  # Call
            np.copyto(outputs[0], np.reshape(self.tmpfwd, self.shape))

        else:

            # CV2 version
            inimg = inputs[0]
            if len(self.H.shape) == 2:
                warpedInput = cv2.warpPerspective(np.asfortranarray(inimg), self.H.T,
                                                  inimg.shape[1::-1], flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
                # Necessary due to array layout in opencv
                np.copyto(outputs[0], warpedInput)

            else:
                for j in range(self.H.shape[2]):
                    warpedInput = cv2.warpPerspective(np.asfortranarray(inimg),
                                                      self.H[:, :, j].T, inimg.shape[1::-1],
                                                      flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT,
                                                      borderValue=0.)
                    # Necessary due to array layout in opencv

                    np.copyto(outputs[0][:, :, :, j], warpedInput)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """

        if self.implementation == Impl['halide']:

            # Halide implementation
            if len(self.H.shape) == 2:
                tmpin = np.asfortranarray(inputs[0][..., np.newaxis].astype(np.float32))
            else:
                tmpin = np.asfortranarray(inputs[0].astype(np.float32))

            Halide('At_warp.cpp').At_warp(tmpin, self.Hf, self.tmpadj)  # Call
            np.copyto(outputs[0], self.tmpadj)

        else:

            # CV2 version
            inimg = inputs[0]
            if len(self.H.shape) == 2:
                # + cv2.WARP_INVERSE_MAP
                warpedInput = cv2.warpPerspective(np.asfortranarray(inimg), self.Hinv.T,
                                                  inimg.shape[1::-1], flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
                np.copyto(outputs[0], warpedInput)

            else:
                outputs[0][:] = 0.0
                for j in range(self.H.shape[2]):
                    warpedInput = cv2.warpPerspective(np.asfortranarray(inimg[:, :, :, j]),
                                                      self.Hinv[:, :, j].T, inimg.shape[1::-1],
                                                      flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT,
                                                      borderValue=0.)
                    # Necessary due to array layout in opencv
                    outputs[0] += warpedInput

    # TODO what is the spectral norm of a warp?
