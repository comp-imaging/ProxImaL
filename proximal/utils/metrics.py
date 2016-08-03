from __future__ import division
import abc
import numpy as np
from .utils import psnr


class metric(object):
    """Represents an metric for measuring reconstruction quality
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, desc, unit, decimals=None):
        self.desc = desc
        self.unit = unit
        self.decimals = decimals
        super(metric, self).__init__()

    @abc.abstractmethod
    def _eval(self, v):
        """Evaluate the metric
        """
        return NotImplemented

    def eval(self, v):
        """Evaluate the metric
        """
        if self.decimals is None:
            return self._eval(v)
        else:
            return np.round(self._eval(v), decimals=self.decimals)

    def message(self, v):
        """Evaluate the metric
        """
        mval = self.eval(v)
        message = "{0}: {1} {2}".format(self.desc, mval, self.unit)
        return message


class psnr_metric(metric):
    """PSNR metric
    """

    def __init__(self, ref, maxval=1.0, pad=None, decimals=2):
        self.pad = pad
        self.ref = ref
        self.maxval = maxval
        super(psnr_metric, self).__init__("PSNR", "dB", decimals)

    def _eval(self, v):
        """Evaluate PSNR metric
        """
        return psnr(np.reshape(v, self.ref.shape),
                    self.ref,
                    pad=self.pad, maxval=self.maxval)
