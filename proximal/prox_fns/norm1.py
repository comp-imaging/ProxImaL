from .prox_fn import ProxFn
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


class norm1(ProxFn):
    """The function ||x||_1.
    """

    def init_tmps(self):
        """Initialize temporary variables for _prox method.
        """
        self.v_sign = np.zeros(self.lin_op.shape, dtype=float)
        self.tmpout = np.zeros(self.lin_op.shape, dtype=np.float32, order='F')

    def _prox(self, rho, v, *args, **kwargs):
        """x = sign(v)*(|v| - 1/rho)_+
        """

        if self.implementation == Impl['halide'] and (len(self.lin_op.shape) in [2, 3, 4]):
            # Halide implementation
            tmpin = np.asfortranarray(v.astype(np.float32))
            Halide('prox_L1.cpp').prox_L1(tmpin, 1. / rho, self.tmpout)
            np.copyto(v, self.tmpout)

        else:
            # Numpy implementation
            np.sign(v, self.v_sign)
            np.absolute(v, v)
            v -= 1. / rho
            np.maximum(v, 0, v)
            v *= self.v_sign

        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return np.linalg.norm(v.ravel(), 1)


class weighted_norm1(norm1):
    """The function ||W.*x||_1.
    """

    def __init__(self, lin_op, weight, **kwargs):
        self.weight = weight
        super(weighted_norm1, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = sign(v)*(|v| - |W|/rho)_+
        """
        idxs = self.weight != 0
        np.sign(v[idxs], self.v_sign[idxs])
        np.absolute(v[idxs], v[idxs])
        v[idxs] -= np.absolute(self.weight[idxs]) / rho
        np.maximum(v[idxs], 0, v[idxs])
        v[idxs] *= self.v_sign[idxs]
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return super(weighted_norm1, self)._eval(self.weight * v)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.weight]
