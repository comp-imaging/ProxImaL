import os
import numpy as np
from scipy.misc import ascent
from scipy.signal import convolve2d

import proximal as px
from proximal.tests.base_test import BaseTest
from proximal.halide.halide import Halide

class TestHalideOps(BaseTest):
    def test_configure(self):
        """Test configuration of halide project
        """
        self.hl = Halide('A_conv', reconfigure=True)

    def test_compile(self):
        """Test compilation
        """
        Halide('A_conv', reconfigure=True, recompile=True, verbose=False)
    

    def _get_testvector(self):
        # Load image
        np_img = np.asfortranarray(ascent(), dtype=np.float32)

        # Load kernel
        K = np.ones((5,5), order='F', dtype=np.float32)
        K /= K.size

        return np_img, K
    

    def test_run(self):
        """ Test running
        """
        np_img, K = self._get_testvector()

        # Test Halide routine
        output = np.empty(np_img.shape, dtype=np.float32, order='F')
        Halide('A_conv', recompile=True).run(np_img, K, output)

        # Test numpy implementation
        output_ref = convolve2d(np_img, K, mode='same', boundary='wrap')

        self.assertItemsAlmostEqual(output, output_ref)
    
    def _test_algo(self, algo, check_convergence=True):
        """ Ensure all internal buffers of the algortihms are Fortran-style ordered.
        """
        np_img, K = self._get_testvector()
        X = px.Variable(np_img.shape)

        # Force the build system to re-compile the Halide-accelerated prox_L1 function.
        # It is also a good time to assert for np.array(dtype=float32, order='F') .
        prox_fns = [px.norm1(X, b=np_img, beta=2, implem='halide')]
        output = np.empty(np_img.shape, dtype=np.float32, order='F')
        Halide('prox_L1', recompile=True).run(np_img, 1.0, output)

        # Ensure all the internal buffers are float32 type, in F-style data layout.
        sltn = algo.solve(prox_fns, [],
                        max_iters=500,
                        eps_rel=1e-5,
                        eps_abs=1e-5)

        if check_convergence:
            # Convergence testing is not the goal of this unit test case. Write
            # a separate test case for this.
            self.assertAlmostEqual(sltn, 0, eps=2e-2)
            self.assertItemsAlmostEqual(X.value, np_img / 2., eps=2e-2)

    def test_ladmm(self):
        self._test_algo(px.ladmm)

    def test_admm(self):
        self._test_algo(px.admm)

    def test_pc(self):
        self._test_algo(px.pc)

    def test_hqs(self):
        self._test_algo(px.hqs)
