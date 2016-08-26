from __future__ import division
from proximal.tests.base_test import BaseTest
from proximal.lin_ops import (Variable, subsample, conv, sum, vstack, LinOpFactory,
                              mul_elemwise, CompGraph)
from proximal.halide.halide import Halide, halide_installed
from proximal.utils.utils import im2nparray, psf2otf
import numpy as np

import os
from PIL import Image
from scipy import signal
from scipy import ndimage
import cv2


class TestLinOps(BaseTest):

    def test_variable(self):
        """Test Variable"""
        var = Variable(3)
        self.assertEqual(var.shape, (3,))
        var = Variable((3,))
        self.assertEqual(var.shape, (3,))
        var = Variable((3, 2))
        self.assertEqual(var.shape, (3, 2))
        var = Variable([3, 2])
        self.assertEqual(var.shape, (3, 2))

    def test_subsample(self):
        """Test subsample lin op.
        """
        # Forward.
        var = Variable((10, 5))
        fn = subsample(var, (2, 1))
        x = np.arange(50) * 1.0
        x = np.reshape(x, (10, 5))
        out = np.zeros(fn.shape)
        fn.forward([x], [out])
        self.assertItemsAlmostEqual(out, x[0::2, :])

        # Adjoint.
        x = np.arange(25) * 1.0
        x = np.reshape(x, (5, 5))
        out = np.zeros(var.shape)
        fn.adjoint([x], [out])
        zeroed_x = np.zeros((10, 5)) * 1.0
        zeroed_x[0::2, :] = x
        self.assertItemsAlmostEqual(out, zeroed_x)

        # Constant arg.
        val = np.arange(50) * 1.0
        val = np.reshape(val, (10, 5))
        fn = subsample(val, (2, 1))
        self.assertItemsAlmostEqual(fn.value, val[0::2, :])

        # Diagonal form.
        var = Variable((5, 1))
        fn = subsample(var, (2, 1))
        assert not fn.is_gram_diag(freq=True)
        assert fn.is_gram_diag(freq=False)
        self.assertItemsAlmostEqual(fn.get_diag(freq=False)[var],
                                    [1, 0, 1, 0, 1])

        # 1D
        x = Variable(5)
        expr = subsample(x, 2)
        self.assertEqual(expr.shape, (3,))

    def test_sum(self):
        # Forward.
        x = Variable((2, 3))
        y = Variable((2, 3))
        fn = sum([x, y])
        x_val = np.reshape(np.arange(6) * 1.0, x.shape)
        y_val = np.reshape(np.arange(6) * 1.0 - 5, x.shape)
        out = np.zeros(fn.shape)
        fn.forward([x_val, y_val], [out])
        self.assertItemsAlmostEqual(out, 2 * np.arange(6) - 5)

        # Adjoint.
        x_val = np.reshape(np.arange(6) * 1.0, x.shape)
        out = [np.zeros(fn.shape), np.zeros(fn.shape)]
        fn.adjoint([x_val], out)
        for arr in out:
            self.assertItemsAlmostEqual(arr, x_val)

        # Constant args.
        x = Variable((2, 3))
        y = Variable((2, 3))
        x_val = np.reshape(np.arange(6) * 1.0, x.shape)
        y_val = np.reshape(np.arange(6) * 1.0 - 5, x.shape)
        fn = sum([x_val, y_val])
        self.assertItemsAlmostEqual(fn.value, 2 * np.arange(6) - 5)

        # Diagonal form.
        x = Variable(5)
        term = mul_elemwise(np.arange(5) - 3, x)
        fn = sum([term, x])
        assert not fn.is_diag(freq=True)
        assert fn.is_diag(freq=False)
        self.assertItemsAlmostEqual(fn.get_diag(freq=False)[x],
                                    np.arange(5) - 3 + np.ones(5))

    def test_conv(self):
        """Test convolution lin op.
        """
        # Forward.
        var = Variable((2, 3))
        kernel = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        fn = conv(kernel, var)
        x = np.arange(6) * 1.0
        x = np.reshape(x, (2, 3))
        out = np.zeros(fn.shape)

        fn.forward([x], [out])
        ks = np.array(kernel.shape)
        cc = np.floor((ks - 1) / 2.0).astype(np.int)  # Center coordinate
        y = np.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                for s in range(2):
                    for t in range(3):
                        y[i, j] += kernel[ks[0] - 1 - s, ks[1] - 1 - t] * \
                                   x[(s + i - cc[0]) % 2, (t + j - cc[1]) % 3]

        # For loop same as convolve
        #y = ndimage.convolve(x, kernel, mode='wrap')

        self.assertItemsAlmostEqual(out, y)

        # Adjoint.#
        x = np.arange(6) * 1.0
        x = np.reshape(x, (2, 3))
        out = np.zeros(var.shape)
        fn.adjoint([x], [out])
        y = np.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                for s in range(2):
                    for t in range(3):
                        y[i, j] += kernel[s, t] * x[(s + i - (ks[0] - 1 - cc[0])) % 2,
                                                    (t + j - (ks[1] - 1 - cc[1])) % 3]

        # For loop same as correlate
        #y = ndimage.correlate(x, kernel, mode='wrap')

        self.assertItemsAlmostEqual(out, y)

        # Diagonal form.
        x = Variable(5)
        kernel = np.arange(5)
        fn = conv(kernel, x)
        assert fn.is_diag(freq=True)
        assert not fn.is_diag(freq=False)
        forward_kernel = psf2otf(kernel, (5,), 1)
        self.assertItemsAlmostEqual(fn.get_diag(freq=True)[x],
                                    forward_kernel)

    def test_conv_halide(self):
        """Test convolution lin op in halide.
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            # opens the file using Pillow - it's not an array yet
            img = Image.open(testimg_filename)
            np_img = np.asfortranarray(im2nparray(img))

            # Convert to gray
            np_img = np.mean(np_img, axis=2)

            # Test problem
            output = np.zeros_like(np_img)
            K = np.ones((11, 11), dtype=np.float32, order='FORTRAN')
            K /= np.prod(K.shape)

            #Convolve in halide
            Halide('A_conv.cpp').A_conv(np_img, K, output)

            #Convolve in scipy
            output_ref = signal.convolve2d(np_img, K, mode='same', boundary='wrap')

            # Transpose
            output_corr = np.zeros_like(np_img)
            Halide('At_conv.cpp').At_conv(np_img, K, output_corr)  # Call

            output_corr_ref = signal.convolve2d(np_img, np.flipud(np.fliplr(K)),
                                                mode='same', boundary='wrap')

            self.assertItemsAlmostEqual(output, output_ref)
            self.assertItemsAlmostEqual(output_corr, output_corr_ref)

    def vstack(self):
        """Test vstack operator.
        """
        # diagonals.
        x = Variable(1)
        y = Variable(1)
        fn = vstack([x, y])
        assert fn.is_gram_diag(freq=False)
        assert fn.is_gram_diag(freq=True)
        fn = vstack([fn])
        assert fn.is_gram_diag(freq=False)
        assert fn.is_gram_diag(freq=True)

    def test_combo(self):
        """Test subsampling followed by convolution.
        """
        # Forward.
        var = Variable((2, 3))
        kernel = np.array([[1, 2, 3]])  # 2x3
        fn = vstack([conv(kernel, subsample(var, (2, 1)))])
        fn = CompGraph(fn)
        x = np.arange(6) * 1.0
        x = np.reshape(x, (2, 3))
        out = np.zeros(fn.output_size)
        fn.forward(x.flatten(), out)
        y = np.zeros((1, 3))

        xsub = x[::2, ::1]
        y = ndimage.convolve(xsub, kernel, mode='wrap')

        self.assertItemsAlmostEqual(np.reshape(out, y.shape), y)

        # Adjoint.
        x = np.arange(3) * 1.0
        x = np.reshape(x, (1, 3))
        out = np.zeros(var.size)
        fn.adjoint(x.flatten(), out)

        y = ndimage.correlate(x, kernel, mode='wrap')
        y2 = np.zeros((2, 3))
        y2[::2, :] = y

        self.assertItemsAlmostEqual(np.reshape(out, y2.shape), y2)
        out = np.zeros(var.size)
        fn.adjoint(x.flatten(), out)
        self.assertItemsAlmostEqual(np.reshape(out, y2.shape), y2)

    def test_mul_elemwise(self):
        """Test mul_elemwise lin op.
        """
        # Forward.
        var = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = mul_elemwise(W, var)
        x = W.copy()
        out = np.zeros(x.shape)
        fn.forward([x], [out])
        self.assertItemsAlmostEqual(out, W * W)

        # Adjoint.
        x = W.copy()
        out = np.zeros(x.shape)
        fn.adjoint([x], [out])
        self.assertItemsAlmostEqual(out, W * W)

        # Diagonal form.
        x = Variable(5)
        fn = mul_elemwise(np.arange(5) - 3, x)
        assert not fn.is_diag(freq=True)
        assert fn.is_diag(freq=False)
        self.assertItemsAlmostEqual(fn.get_diag(freq=False)[x],
                                    np.arange(5) - 3)

    def test_diagonalization(self):
        """Test automatic diagonalization.
        """
        var = Variable((2, 5))
        K = np.array([[-1, 1]])
        expr = 2*vstack([conv(K, var), conv(K, var)])
        assert expr.is_gram_diag(freq=True)


    def test_black_box(self):
        """Test custom linear operators.
        """
        scale = 2

        def op(input, output):
            output[:] = 2 * input

        my_op = LinOpFactory((2, 2), (2, 2), op, op, scale)
        x = Variable((2, 2))
        expr = my_op(x)
        val = np.reshape(np.arange(4), (2, 2))
        x.value = val
        self.assertItemsAlmostEqual(expr.value, 2 * val)
        output = np.zeros((2, 2))
        expr.forward([val], [output])
        self.assertItemsAlmostEqual(output, 2 * val)
        expr.adjoint([val], [output])
        self.assertItemsAlmostEqual(output, 2 * val)

    def test_op_overloading(self):
        """Test operator overloading.
        """
        # Multiplying by a scalar.

        # Forward.
        var = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = -2 * mul_elemwise(W, var)
        fn = CompGraph(fn)
        x = W.copy()
        out = np.zeros(x.shape)
        fn.forward(x.flatten(), out)
        self.assertItemsAlmostEqual(out, -2 * W * W)

        # Adjoint.
        x = W.copy()
        out = np.zeros(x.shape).flatten()
        fn.adjoint(x, out)
        self.assertItemsAlmostEqual(out, -2 * W * W)

        # Forward.
        var = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = mul_elemwise(W, var) * 0.5
        fn = CompGraph(fn)
        x = W.copy()
        out = np.zeros(x.shape)
        fn.forward(x.flatten(), out)
        self.assertItemsAlmostEqual(out, W * W / 2.)

        # Adjoint.
        x = W.copy()
        out = np.zeros(x.shape).flatten()
        fn.adjoint(x, out)
        self.assertItemsAlmostEqual(out, W * W / 2.)

        # Dividing by a scalar.
        # Forward.
        var = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = mul_elemwise(W, var) / 2
        fn = CompGraph(fn)
        x = W.copy()
        out = np.zeros(x.shape)
        fn.forward(x, out)
        self.assertItemsAlmostEqual(out, W * W / 2.)

        # Adding lin ops.
        # Forward.
        x = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = mul_elemwise(W, x)
        fn = fn + x + x
        self.assertEquals(len(fn.input_nodes), 3)
        fn = CompGraph(fn)
        x = W.copy()
        out = np.zeros(fn.shape)
        fn.forward(x, out)
        self.assertItemsAlmostEqual(out, W * W + 2 * W)

        # Adjoint.
        x = W.copy()
        out = np.zeros(x.shape).flatten()
        fn.adjoint(x, out)
        self.assertItemsAlmostEqual(out, W * W + 2 * W)

        # Adding in a constant.
        # CompGraph should ignore the constant.
        x = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = mul_elemwise(W, x)
        fn = fn + x + W
        self.assertEquals(len(fn.input_nodes), 3)
        fn = CompGraph(fn)
        x = W.copy()
        out = np.zeros(fn.shape)
        fn.forward(x, out)
        self.assertItemsAlmostEqual(out, W * W + W)

       # Subtracting lin ops.
        # Forward.
        x = Variable((2, 5))
        W = np.arange(10)
        W = np.reshape(W, (2, 5))
        fn = -mul_elemwise(W, x)
        fn = x + x - fn
        self.assertEquals(len(fn.input_nodes), 3)
        fn = CompGraph(fn)
        x = W.copy()
        out = np.zeros(fn.shape)
        fn.forward(x, out)
        self.assertItemsAlmostEqual(out, W * W + 2 * W)

        # Adjoint.
        x = W.copy()
        out = np.zeros(x.shape).flatten()
        fn.adjoint(x, out)
        self.assertItemsAlmostEqual(out, W * W + 2 * W)

    def test_mask_halide(self):
        """Test mask lin op in halide.
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            # opens the file using Pillow - it's not an array yet
            img = Image.open(testimg_filename)
            np_img = np.asfortranarray(im2nparray(img))

            # Test problem
            output = np.zeros_like(np_img)
            mask = np.asfortranarray(np.random.randn(*list(np_img.shape)).astype(np.float32))
            mask = np.maximum(mask, 0.)

            Halide('A_mask.cpp').A_mask(np_img, mask, output)  # Call
            output_ref = mask * np_img

            # Transpose
            output_trans = np.zeros_like(np_img)
            Halide('At_mask.cpp').At_mask(np_img, mask, output_trans)  # Call

            self.assertItemsAlmostEqual(output, output_ref)
            self.assertItemsAlmostEqual(output_trans, output_ref)

    def test_grad_halide(self):
        """Test gradient lin op in halide.
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            img = Image.open(testimg_filename)
            np_img = np.asfortranarray(im2nparray(img))

            # Convert to gray
            np_img = np.mean(np_img, axis=2)

            # Test problem
            output = np.zeros((np_img.shape[0], np_img.shape[1],
                               np_img.shape[2] if (len(np_img.shape) > 2) else 1, 2),
                              dtype=np.float32, order='FORTRAN')

            #Gradient in halide
            Halide('A_grad.cpp').A_grad(np_img, output)  # Call

            # Compute comparison
            f = np_img
            if len(np_img.shape) == 2:
                f = f[..., np.newaxis]

            ss = f.shape
            fx = f[:, np.r_[1:ss[1], ss[1] - 1], :] - f
            fy = f[np.r_[1:ss[0], ss[0] - 1], :, :] - f
            Kf = np.asfortranarray(np.stack((fy, fx), axis=-1))

            # Transpose
            output_trans = np.zeros(f.shape, dtype=np.float32, order='F')
            Halide('At_grad.cpp').At_grad(Kf, output_trans)  # Call

            # Compute comparison (Negative divergence)
            Kfy = Kf[:, :, :, 0]
            fy = Kfy - Kfy[np.r_[0, 0:ss[0] - 1], :, :]
            fy[0, :, :] = Kfy[0, :, :]
            fy[-1, :, :] = -Kfy[-2, :, :]

            Kfx = Kf[:, :, :, 1]
            ss = Kfx.shape
            fx = Kfx - Kfx[:, np.r_[0, 0:ss[1] - 1], :]
            fx[:, 0, :] = Kfx[:, 0, :]
            fx[:, -1, :] = -Kfx[:, -2, :]

            # TODO are these wrong?
            # KtKf = -fx - fy
            # self.assertItemsAlmostEqual(output, Kf)
            # self.assertItemsAlmostEqual(output_trans, KtKf)

    def test_warp_halide(self):
        """Test warp lin op in halide.
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            img = Image.open(testimg_filename)
            np_img = np.asfortranarray(im2nparray(img))

            # Convert to gray
            np_img = np.mean(np_img, axis=2)

            # Generate problem
            theta_rad = 5.0 * np.pi / 180.0
            H = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0.0001],
                          [np.sin(theta_rad), np.cos(theta_rad), 0.0003],
                          [0., 0., 1.]], dtype=np.float32, order='F')

            # Reference
            output_ref = cv2.warpPerspective(np_img, H.T, np_img.shape[1::-1], flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.)

            # Halide
            output = np.zeros_like(np_img)
            Hc = np.asfortranarray(np.linalg.pinv(H)[..., np.newaxis])  # Third axis for halide
            Halide('A_warp.cpp').A_warp(np_img, Hc, output)  # Call

            # Transpose
            output_trans = np.zeros_like(np_img)
            Hinvc = np.asfortranarray(H[..., np.newaxis])  # Third axis for halide
            Halide('At_warp.cpp').At_warp(output, Hinvc, output_trans)  # Call

            # Compute reference
            output_ref_trans = cv2.warpPerspective(output_ref, H.T, np_img.shape[1::-1],
                                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
            # Opencv does inverse warp
            self.assertItemsAlmostEqual(output, output_ref, places=1)
            # Opencv does inverse warp
            self.assertItemsAlmostEqual(output_trans, output_ref_trans, places=1)
