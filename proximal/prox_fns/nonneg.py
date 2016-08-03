from .prox_fn import ProxFn
import numpy as np


class nonneg(ProxFn):
    """The function I(x >= 0).
    """

    def _prox(self, rho, v, *args, **kwargs):
        """x = pos(x).
        """
        np.maximum(v, 0, v)
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        if np.min(v) > -1e-3:
            return 0
        else:
            return np.inf


class weighted_nonneg(nonneg):
    """The function I(W*x >= 0).
    """

    def __init__(self, lin_op, weight, **kwargs):
        self.weight = weight
        super(weighted_nonneg, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = pos(x).
        """
        idxs = self.weight != 0.
        v[idxs] = np.maximum(self.weight[idxs] * v[idxs], 0.) / self.weight[idxs]
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return super(weighted_nonneg, self)._eval(self.weight * v)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.weight]


class masked_nonneg(nonneg):
    """The function I(x >= 0) for M == 1 else I(x == 0)
       Converts any non-binary mask to a binary one.
    """

    def __init__(self, lin_op, mask, **kwargs):
        self.mask = mask
        self.mask[self.mask < 0.5] = 0.0
        self.mask[self.mask >= 0.5] = 1.0
        super(masked_nonneg, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = pos(x).
        """
        np.maximum(v, 0, v)
        v *= self.mask
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        if np.min(self.mask * v) > -1e-3 or np.max(np.absolute((1.0 - self.mask) * v)) > -1e-3:
            return 0
        else:
            return np.inf

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.mask]
