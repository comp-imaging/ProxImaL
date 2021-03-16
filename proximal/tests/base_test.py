# Base class for unit tests.
import numpy as np
from typing import Union
from numbers import Real
from pytest import approx


class BaseTest:

    def assertItemsAlmostEqual(self,
                               a: Union[list, np.array],
                               b: Union[list, np.array],
                               places=4,
                               eps=1e-5):
        if type(a) is list:
            a = np.array(a)

        if type(b) is list:
            b = np.array(a)

        assert a.shape == b.shape

        norm_infinity_b = np.linalg.norm(b.ravel(), np.Inf)

        if norm_infinity_b == 0:
            # b is a zero matrix / vector
            assert a == approx(b, abs=eps)
            return

        assert a == approx(b, abs=norm_infinity_b * eps)

    def assertAlmostEqual(self, a: Real, b: Real, places=4, eps=1e-5):
        ''' Make sure that max absolute difference is smaller than a threshold '''
        if b == 0:
            assert a == approx(b, abs=eps)
            return

        assert a == approx(b, rel=eps)

    def assertEqual(self, a, b):
        self.assertEquals(a, b)

    def assertEquals(self, a, b):
        assert np.all(a == b)
