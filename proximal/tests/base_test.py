# Base class for unit tests.
import numpy as np
from typing import Union
from numbers import Real


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

        vmax = np.max(
            (np.linalg.norm(a.ravel(),
                            np.Inf), np.linalg.norm(b.ravel(), np.Inf)))

        rel_diff = np.linalg.norm(a.ravel() - b.ravel(), np.Inf)

        if vmax == 0:
            assert rel_diff < eps
            return

        rel_diff /= vmax

        assert rel_diff < eps

    def assertAlmostEqual(self, a: Real, b: Real, places=4, eps=1e-5):
        ''' Make sure that max absolute difference is smaller than a threshold '''
        delta = abs(a - b)
        vmax = max(abs(a), abs(b))
        if vmax == 0:
            assert delta < eps
            return

        rel_diff = delta / vmax
        assert rel_diff < eps

    def assertEqual(self, a, b):
        self.assertEquals(a, b)

    def assertEquals(self, a, b):
        assert np.all(a == b)
