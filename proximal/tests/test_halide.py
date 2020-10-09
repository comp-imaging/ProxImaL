import os
import numpy as np
from scipy.misc import ascent
from scipy.signal import convolve2d

from proximal.tests.base_test import BaseTest
from proximal.utils.utils import get_test_image, get_kernel
from proximal.halide.halide import Halide

class TestLinOps(BaseTest):
    def test_configure(self):
        """Test configuration of halide project
        """
        self.hl = Halide('A_conv', reconfigure=True)

    def test_compile(self):
        """Test compilation
        """
        Halide('A_conv', reconfigure=True, recompile=True, verbose=False)

    def test_run(self):
        """ Test running
        """
        # Load image
        np_img = np.asfortranarray(ascent(), dtype=np.float32)

        # Load kernel
        K = np.ones((5,5), order='F', dtype=np.float32)
        K /= K.size

        # Test Halide routine
        output = np.zeros_like(np_img)
        Halide('A_conv', recompile=True).run(np_img, K, output)

        # Test numpy implementation
        output_ref = convolve2d(np_img, K, mode='same', boundary='wrap')

        self.assertItemsAlmostEqual(output, output_ref)