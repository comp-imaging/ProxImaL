from .prox_fn import ProxFn
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class diff_fn(ProxFn):
    """A generic prox operator for differentiable functions using L-BFGS.
    """

    def __init__(self, lin_op, func, fprime, bounds=None, factr=1e7, **kwargs):
        """Initialization.

        Args:
            lin_op: A linear operator.
            func: A function call for evaluating the function.
            fprime: A function call for evaluating the derivative.
            bounds: A list of (lower bound, upper bound) on each entry.
        """
        self.func = func
        self.fprime = fprime
        self.bounds = bounds
        self.factr = factr
        super(diff_fn, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """Use L-BFGS
        """
        # Derivative of augmented function.
        def prox_func(x):
            return self.func(x) + (rho / 2.0) * np.square(x.ravel() - v.ravel()).sum()

        def prox_fprime(x):
            return self.fprime(x).ravel() + rho * (x.ravel() - v.ravel())

        x, f, d = fmin_l_bfgs_b(prox_func, v, prox_fprime,
                                bounds=self.bounds, factr=self.factr)
        return np.reshape(x, self.lin_op.shape)

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return self.func(v)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.func, self.fprime, self.bounds, self.factr]
