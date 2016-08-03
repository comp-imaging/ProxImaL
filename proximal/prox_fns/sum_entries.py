from .prox_fn import ProxFn
import numpy as np


def sum_entries(lin_op, **kwargs):
    c = np.ones(lin_op.shape)
    if "c" in kwargs:
        c += kwargs["c"]
    return zero_prox(lin_op, c=c, **kwargs)


class zero_prox(ProxFn):
    """The function f(x) = 0.
    """

    def _prox(self, rho, v, *args, **kwargs):
        """x = v
        """
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return 0
