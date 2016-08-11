from __future__ import division
import abc
import numpy as np
from proximal.utils import Impl


class ProxFn(object):
    """Represents alpha*f(beta*x - b) + <c,x> + gamma*<x,x> + d
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, lin_op, alpha=1.0, beta=1.0, b=0.0, c=0.0,
                 gamma=0.0, d=0.0, implem=None):
        # Error checking.
        for elem, name in zip([b, c], ["b", "c"]):
            if not (np.isscalar(elem) or elem.shape == lin_op.shape):
                raise Exception("Invalid dimensions of %s." % name)
        for elem, name in zip([alpha, gamma, d], ["alpha", "gamma"]):
            if not np.isscalar(elem) or elem < 0:
                raise Exception("%s must be a nonnegative scalar." % name)
        for elem, name in zip([beta, d], ["beta", "d"]):
            if not np.isscalar(elem):
                raise Exception("%s must be a scalar." % name)

        self.implem_key = implem
        self.implementation = Impl['numpy']
        if implem is not None:
            self.set_implementation(implem)

        self.lin_op = lin_op
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.b = b
        self.c = c
        if np.isscalar(b):
            self.b = b * np.ones(self.lin_op.shape)
        if np.isscalar(c):
            self.c = c * np.ones(self.lin_op.shape)
        self.gamma = float(gamma)
        self.d = float(d)
        self.init_tmps()
        super(ProxFn, self).__init__()

    def set_implementation(self, im):
        if im in Impl.values():
            self.implementation = im
        elif im in Impl.keys():
            self.implementation = Impl[im]
        else:
            raise Exception("Invalid implementation.")

        return self.implementation

    def implementation(self, im):
        return self.implementation

    def variables(self):
        """Return a list of the variables in the problem.
        """
        return self.lin_op.variables()

    def init_tmps(self):
        """Initialize temporary variables for _prox method.
        """
        pass

    @abc.abstractmethod
    def _prox(self, rho, v, *args, **kwargs):
        """The prox function for a specific atom.
        """
        return NotImplemented

    def prox(self, rho, v, *args, **kwargs):
        """Wrapper on the prox function to handle alpha, etc.
           It is here the iteration for debug purposese etc.
        """
        rho_hat = (rho + 2 * self.gamma) / (self.alpha * self.beta**2)
        # vhat = (rho*v - c)*beta/(rho + 2*gamma) - b
        # Modify v in-place. This is important for the Python to be performant.
        v *= rho
        v -= self.c
        v *= self.beta / (rho + 2 * self.gamma)
        v -= self.b
        xhat = self._prox(rho_hat, v, *args, **kwargs)
        # x = (xhat + b)/beta
        # Modify result in-place.
        xhat += self.b
        xhat /= self.beta
        return xhat

    @abc.abstractmethod
    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return NotImplemented

    def eval(self, v):
        """Evaluate the function on v.
        """
        return self.alpha * self._eval(self.beta * v - self.b) + \
            np.sum(self.c * v) + self.gamma * np.square(v).sum() + self.d

    @property
    def value(self):
        return self.eval(self.lin_op.value)

    def __str__(self):
        """Default to string is name of class.
        """
        return self.__class__.__name__

    def __add__(self, other):
        """ProxFn + ProxFn(s).
        """
        if isinstance(other, ProxFn):
            return [self, other]
        elif type(other) == list:
            return [self] + other
        else:
            return NotImplemented

    def __radd__(self, other):
        """Called for list + ProxFn.
        """
        if type(other) == list:
            return other + [self]
        else:
            return NotImplemented

    def __mul__(self, other):
        """ProxFn * Number.
        """
        # Can only multiply by scalar constants.
        if np.isscalar(other) and other > 0:
            return self.copy(alpha=self.alpha * other)
        else:
            raise TypeError("Can only multiply by a positive scalar.")

    def __rmul__(self, other):
        """Called for Number * ProxFn.
        """
        return self * other

    def __div__(self, other):
        """Called for ProxFn/Number.
        """
        return (1. / other) * self

    def __truediv__(self, other):
        """ProxFn / integer.
        """
        return self.__div__(other)

    def copy(self, lin_op=None, **kwargs):
        """Returns a shallow copy of the object.

        Used to reconstruct an object tree.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the object. If args=None, use the
            current args of the object.

        Returns
        -------
        Expression
        """
        if lin_op is None:
            lin_op = self.lin_op
        data = self.get_data()
        curr_args = {'alpha': self.alpha,
                     'beta': self.beta,
                     'gamma': self.gamma,
                     'c': self.c,
                     'b': self.b,
                     'd': self.d,
                     'implem': self.implem_key}
        for key in curr_args.keys():
            if key not in kwargs:
                kwargs[key] = curr_args[key]
        return type(self)(lin_op, *data, **kwargs)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return []
