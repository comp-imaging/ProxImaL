from .lin_op import LinOp
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


class grad(LinOp):
    """
    gradient operation. can be defined for different dimensions.
    default is n-d gradient.
    """

    def __init__(self, arg, dims=None, implem=None):

        if dims is not None:
            self.dims = dims
        else:
            self.dims = len(arg.shape)  # Full n-d gradient

        shape = arg.shape + (self.dims,)

        # Temp array for halide
        self.tmpfwd = None
        self.tmpadj = None
        if len(arg.shape) in [2, 3] and self.dims == 2:
            self.tmpfwd = np.zeros((arg.shape[0], arg.shape[1],
                                    arg.shape[2] if (len(arg.shape) > 2) else 1, 2),
                                   dtype=np.float32, order='FORTRAN')

            self.tmpadj = np.zeros((arg.shape[0], arg.shape[1],
                                    arg.shape[2] if (len(arg.shape) > 2) else 1),
                                   dtype=np.float32, order='FORTRAN')

        super(grad, self).__init__([arg], shape, implem)

    def forward(self, inputs, outputs):
        """The forward operator for n-d gradients.

        Reads from inputs and writes to outputs.
        """

        if self.implementation == Impl['halide'] and \
                (len(self.shape) == 3 or len(self.shape) == 4) and self.dims == 2:
            # Halide implementation
            if len(self.shape) == 3:
                tmpin = np.asfortranarray((inputs[0][..., np.newaxis]).astype(np.float32))
            else:
                tmpin = np.asfortranarray((inputs[0]).astype(np.float32))

            Halide('A_grad.cpp').A_grad(tmpin, self.tmpfwd)  # Call
            np.copyto(outputs[0], np.reshape(self.tmpfwd, self.shape))

        else:

            # Input
            f = inputs[0]

            # Build up index for shifted array
            ss = f.shape
            stack_arr = ()
            for j in range(self.dims):

                # Add grad for this dimension (same as index)
                il = ()
                for i in range(len(ss)):
                    if i == j:
                        il += np.index_exp[np.r_[1:ss[j], ss[j] - 1]]
                    else:
                        il += np.index_exp[:]

                fgrad_j = f[il] - f
                stack_arr += (fgrad_j,)

            # Stack all grads as new dimension
            np.copyto(outputs[0], np.stack(stack_arr, axis=-1))

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """

        if self.implementation == Impl['halide'] and \
                (len(self.shape) == 3 or len(self.shape) == 4) and self.dims == 2:

            # Halide implementation
            if len(self.shape) == 3:
                tmpin = np.asfortranarray(np.reshape(inputs[0],
                                                     (self.shape[0], self.shape[1],
                                                      1, 2)).astype(np.float32))
            else:
                tmpin = np.asfortranarray(inputs[0].astype(np.float32))

            Halide('At_grad.cpp').At_grad(tmpin, self.tmpadj)  # Call
            np.copyto(outputs[0], np.reshape(self.tmpadj, self.shape[:-1]))

        else:

            # Compute comparison (Negative divergence)
            f = inputs[0]

            outputs[0].fill(0.0)
            for j in range(self.dims):

                # Get component
                fj = f[..., j]
                ss = fj.shape

                # Add grad for this dimension (same as index)
                istart = ()
                ic = ()
                iend_out = ()
                iend_in = ()
                for i in range(len(ss)):
                    if i == j:
                        istart += np.index_exp[0]
                        ic += np.index_exp[np.r_[0, 0:ss[j] - 1]]
                        iend_out += np.index_exp[-1]
                        iend_in += np.index_exp[-2]
                    else:
                        istart += np.index_exp[:]
                        ic += np.index_exp[:]
                        iend_in += np.index_exp[:]
                        iend_out += np.index_exp[:]

                # Do the grad operation for dimension j
                fd = fj - fj[ic]
                fd[istart] = fj[istart]
                fd[iend_out] = -fj[iend_in]

                outputs[0] += (-fd)

    def get_dims(self):
        """Return the dimensinonality of the gradient
        """
        return self.dims

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
        # 1D gradient operator has spectral norm = 2.
        # ND gradient is permutation of stacked grad in axis 0, axis 1, etc.
        # so norm is 2*sqrt(dims)
        return 2 * np.sqrt(self.dims) * input_mags[0]
