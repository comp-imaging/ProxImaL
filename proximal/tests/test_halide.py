from proximal.tests.base_test import BaseTest
from proximal.halide.halide import Halide, halide_installed
from proximal.utils.utils import im2nparray

import os
import numpy as np
from PIL import Image


class TestHalide(BaseTest):

    def test_compile(self):
        """Test compilation and run.
        """
        if halide_installed():
            # Force recompilation of conv
            Halide('A_conv.cpp', recompile=True, verbose=False)

    def test_slicing(self):
        """Test slicing  over numpy arrays in halide.
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'data', 'angela.jpg')
            np_img = im2nparray(Image.open(testimg_filename))
            np_img = np.asfortranarray(np.tile(np_img[..., np.newaxis], (1, 1, 1, 3)))

            # Test problem
            output = np.zeros_like(np_img)
            mask = np.asfortranarray(np.random.randn(*list(np_img.shape[0:3])).astype(np.float32))
            mask = np.maximum(mask, 0.)

            for k in range(np_img.shape[3]):
                Halide('A_mask.cpp').A_mask(np.asfortranarray(np_img[:, :, :, k]),
                                            mask, output[:, :, :, k])  # Call

                output_ref = np.zeros_like(np_img)
                for k in range(np_img.shape[3]):
                    output_ref[:, :, :, k] = mask * np_img[:, :, :, k]

            self.assertItemsAlmostEqual(output, output_ref)
