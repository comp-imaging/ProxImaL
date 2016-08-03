from .prox_fn import ProxFn
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


class poisson_norm(ProxFn):
    """The function ||x||_poisson(b) =
       x + Ind_+(x) - b * log( x_+ )
       This is the log likelihood of a poisson fit
    """

    def __init__(self, lin_op, bp, **kwargs):
        """Initialize temporary variables for _prox method.
        """
        self.bp = bp
        self.bph = np.asfortranarray(bp.astype(np.float32))
        self.maskh = np.ones(lin_op.shape, dtype=np.float32, order='F')
        self.tmpout = np.zeros(lin_op.shape, dtype=np.float32, order='F')

        super(poisson_norm, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = 1/2* ( v - 1./rho + sqrt( ||v - 1./rho||^2 + 4 * 1./rho * b ) )
        """
        if self.implementation == Impl['halide'] and (len(self.lin_op.shape) in [2, 3, 4]):

            # Halide implementation
            tmpin = np.asfortranarray(v)
            Halide('prox_Poisson.cpp').prox_Poisson(
                tmpin, self.maskh, self.bph, 1. / rho, self.tmpout)
            np.copyto(v, self.tmpout)
        else:
            v = 0.5 * (v - 1. / rho + np.sqrt((v - 1. / rho) *
                       (v - 1. / rho) + 4. * 1. / rho * self.bp))

        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """

        # Positivity penalty
        if np.min(v) < -1e-3:
            return np.inf

        # Other penalties
        vsum = v.copy()
        vsum -= self.bp * np.log(np.maximum(v, 1e-9))
        return vsum.sum()

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.bp]


class weighted_poisson_norm(poisson_norm):
    """The function ||W.*x||_poisson.
    """

    def __init__(self, lin_op, bp, weight, **kwargs):
        self.weight = weight
        super(weighted_poisson_norm, self).__init__(lin_op, bp, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = 1/2* ( v - |W|/rho + sqrt( ||v - |W|/rho||^2 + 4 * 1./rho * b ) )
        """
        output = 0.5 * (v - np.absolute(self.weight) / rho +
                        np.sqrt((v - np.absolute(self.weight) / rho)
                        * (v - np.absolute(self.weight) / rho) + 4. * 1. / rho * self.bp))

        # Reference
        idxs = self.weight == 0
        output[idxs] = v[idxs]
        return output

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return super(weighted_poisson_norm, self)._eval(self.weight * v)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.bp, self.weight]
