from .lin_op import LinOp
import numpy as np
from proximal.utils.utils import Impl, psf2otf, fftd, ifftd
from proximal.halide.halide import Halide


class conv(LinOp):
    """Circular convolution of the input with a kernel.
    """

    def __init__(self, kernel, arg, dims=None, implem=None):
        self.kernel = kernel
        if dims is not None and dims < len(arg.shape):
            self.dims = dims
        else:
            self.dims = None
        self.forward_kernel = psf2otf(kernel, arg.shape, dims)
        self.adjoint_kernel = self.forward_kernel.conj()
        self.initialized = False
        # Set implementation in super-class
        super(conv, self).__init__([arg], arg.shape, implem)

    # TODO hack. Either add to every LinOp or make copy function like for ProxFns.
    def init_kernel(self):
        if not self.initialized:
            arg = self.input_nodes[0]
            # Replicate kernel for multichannel deconv
            if len(arg.shape) == 3 and len(self.kernel.shape) == 2:
                self.kernel = np.stack((self.kernel,) * arg.shape[2], axis=-1)

            # Halide kernel
            if self.implementation == Impl['halide'] and \
               (len(arg.shape) == 2 or (len(arg.shape) == 3 and arg.dims == 2)):
                self.kernel = np.asfortranarray(self.kernel.astype(np.float32))
                # Halide FFT (pack into diag)
                # TODO: FIX IMREAL LATER
                hsize = arg.shape if len(arg.shape) == 3 else arg.shape + (1,)
                output_fft_tmp = np.zeros(((hsize[0] + 1) / 2 + 1, hsize[1], hsize[2], 2),
                                          dtype=np.float32, order='F')
                Halide('fft2_r2c.cpp').fft2_r2c(self.kernel, self.kernel.shape[1] / 2,
                                                self.kernel.shape[0] / 2, output_fft_tmp)
                self.forward_kernel[:] = 0.

                if len(arg.shape) == 2:
                    self.forward_kernel[0:(hsize[0] + 1) / 2 + 1, ...] = 1j * \
                                           output_fft_tmp[..., 0, 1]
                    self.forward_kernel[0:(hsize[0] + 1) / 2 + 1, ...] += output_fft_tmp[..., 0, 0]
                else:
                    self.forward_kernel[0:(hsize[0] + 1) / 2 + 1, ...] = 1j * output_fft_tmp[..., 1]
                    self.forward_kernel[0:(hsize[0] + 1) / 2 + 1, ...] += output_fft_tmp[..., 0]

            self.tmpout = np.zeros(arg.shape, dtype=np.float32, order='F')
            self.initialized = True

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        self.init_kernel()
        if self.implementation == Impl['halide'] and \
                (len(self.shape) == 2 or (len(self.shape) == 3 and self.dims == 2)):

            # Halide implementation
            tmpin = np.asfortranarray(inputs[0].astype(np.float32))
            Halide('A_conv.cpp').A_conv(tmpin, self.kernel, self.tmpout)  # Call
            np.copyto(outputs[0], self.tmpout)

        else:

            # Default numpy using FFT
            X = fftd(inputs[0], self.dims)
            X *= self.forward_kernel
            np.copyto(outputs[0], ifftd(X, self.dims).real)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        self.init_kernel()
        if self.implementation == Impl['halide'] and \
                (len(self.shape) == 2 or (len(self.shape) == 3 and self.dims == 2)):

            # Halide implementation
            tmpin = np.asfortranarray(inputs[0].astype(np.float32))
            Halide('At_conv.cpp').At_conv(tmpin, self.kernel, self.tmpout)  # Call
            np.copyto(outputs[0], self.tmpout)

        else:

            # Default numpy using FFT
            U = fftd(inputs[0], self.dims)
            U *= self.adjoint_kernel
            np.copyto(outputs[0], ifftd(U, self.dims).real)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return freq and self.input_nodes[0].is_diag(freq)

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
        assert freq
        var_diags = self.input_nodes[0].get_diag(freq)
        self_diag = np.reshape(self.forward_kernel, self.size)
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
        return np.max(np.abs(self.forward_kernel)) * input_mags[0]
