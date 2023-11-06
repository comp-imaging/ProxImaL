# Spatially Varying PSF Deconvolution Algorithm based on:
# Flicker & Rigaut, 2005 https://doi.org/10.1364/JOSAA.22.000504

import sys

sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter

import proximal as px
from proximal.lin_ops.black_box import LinOpFactory
from proximal.utils.utils import get_kernel, get_test_image

############################################################


def planewise_conv(psf_modes, im_shape, n_psf: int):
    ''' Planewise convolution.

        For each PSF with small supports, simulate the image blur with 2D
        shift-invariant convolution.
    '''

    def forward(im_src, im_sim):
        assert im_sim.shape[2] == n_psf
        assert np.all(im_src.shape == im_sim.shape[:2])

        for i in range(n_psf):
            im_sim[..., i] = convolve(im_src, psf_modes[..., i], mode='wrap')

    def adjoint(im_src, im_sim):
        assert im_src.shape[2] == n_psf
        assert np.all(im_src.shape[:2] == im_sim.shape)

        im_sim[:] = 0
        for i in range(n_psf):
            im_sim[:] += convolve(im_src[..., i],
                                  np.rot90(psf_modes[..., i], k=2),
                                  mode='wrap')

    # Estimate the largest eigenvalue of convT[ conv(x)] .
    scale_factor = np.sum(psf_modes, axis=(0, 1)).max() * n_psf
    blur_op = LinOpFactory(im_shape, (*im_shape, n_psf),
                           forward,
                           adjoint,
                           norm_bound=scale_factor)

    return blur_op


def weighted_planewise_conv(psf_modes, weights, im_shape, n_psf: int):
    ''' Weighted planewise convolution.

        For each PSF with small supports, apply corresponding weights to the
        source image, then simulate the image blur with 2D shift-invariant
        convolution.
    '''

    #assert im_shape[0] * im_shape[1] == psf_modes.shape[1] * psf_modes.shape[2]

    def forward(im_src, im_sim):
        assert np.all(im_src.shape == im_sim.shape)

        im_sim[:] = 0
        for i in range(n_psf):
            im_sim[:] += convolve(im_src * weights[..., i],
                                  psf_modes[..., i],
                                  mode='wrap')

    def adjoint(im_src, im_sim):
        assert np.all(im_src.shape == im_sim.shape)

        im_sim[:] = 0
        for i in range(n_psf):
            im_sim[:] += convolve(im_src * weights[..., i],
                                  np.rot90(psf_modes[..., i], 2),
                                  mode='wrap')

    # Estimate the largest eigenvalue of convT[ conv(x)] .
    scale_factor = np.sum(psf_modes, axis=(0, 1)).max() * n_psf * weights.max()
    blur_op = LinOpFactory(im_shape,
                           im_shape,
                           forward,
                           adjoint,
                           norm_bound=scale_factor)

    return blur_op


def NagyAndOLeary1998(x, im, psf_modes, weights, mu=0.01, n_psf=15):
    ''' Convolve first, then apply weights.

        According to Eq. 23 of the review article (link below), there exists a
        non-equivalent image interpolation formulation introduced by Nagy and
        O'Leary.
         
        Reference: Denis, Thiebaut, Soulez et al. 2015,
        https://doi.org/10.1007/s11263-015-0817-x
    '''
    blur_op = planewise_conv(psf_modes, im.shape, n_psf)
    im_blur = blur_op(x)

    im_stack = np.empty(weights.shape, dtype=np.float32, order='F')
    im_stack[:] = im[..., np.newaxis]

    # The weights are manually "absorbed" into the L2-norm, which has a fast
    # elemwise proximal operator.
    #
    # TODO: Crop the im_stack and weight to 1/4 of image grid size; elminiate
    # redundant 2D convolution with zero weights.
    data_term = px.weighted_sum_squares(im_blur - im_stack, weights)

    im_grad = px.grad(x)
    grad_term = mu * px.norm1(im_grad) + (1 - mu) * px.sum_squares(im_grad)
    return data_term + grad_term


def FlickerAndRigaut2005(x,
                         im,
                         psf_modes,
                         weights,
                         mu=0.01,
                         alpha=0.9,
                         n_psf=15,
                         using_blackbox=False):
    ''' Weight, then convolve.

        According to Eq. 22 of the review article (link below), PSF
        interpolation technique preserves PSF symmetry and normalization with
        minimal sacrifice of the computation complexity.
         
        Reference: Denis, Thiebaut, Soulez et al. 2015,
        https://doi.org/10.1007/s11263-015-0817-x
    '''
    if using_blackbox:
        blur_op = weighted_planewise_conv(psf_modes, weights, im.shape, n_psf)
        im_blur = blur_op(x)

        data_term = px.sum_squares(im_blur - im)
    else:
        im_blur = px.sum([
            px.conv(psf_modes[..., i], px.mul_elemwise(weights[..., i], x))
            for i in range(n_psf)
        ])

        data_term = px.sum_squares(im_blur - im)

    # The weights are manually "absorbed" into the L2-norm, which has a fast
    # elemwise proximal operator.
    #
    # TODO: Crop the im_stack and weight to double of image grid size; elminiate
    # redundant 2D convolution with zero weights.

    im_grad = px.grad(x)
    grad_term = (mu * alpha * n_psf * px.group_norm1(im_grad, group_dims=[2]) +
                 mu * (1 - alpha) * n_psf * px.sum_squares(im_grad))
    return data_term + grad_term


def draw(im, idx: int, title: str):
    plt.subplot(3, 4, idx)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(title)


def get_psf(n_psf: int, kwidth: int):
    assert n_psf <= 4

    psf_modes = np.empty((kwidth, kwidth, n_psf), dtype=np.float32, order='F')
    psf_modes[..., 0] = get_kernel(kwidth, 2)

    for i in range(1, n_psf):
        psf_modes[..., i] = np.rot90(psf_modes[..., 0], k=i)

    return psf_modes


def get_mask(n_psf: int, imwidth: int, feather_edge: int = 100):
    assert n_psf <= 4

    mask = np.zeros((imwidth, imwidth, n_psf), dtype=np.float32, order='F')

    W = int(imwidth / 2)
    mask[:W, :W, 0] = 1
    mask[..., 0] = gaussian_filter(mask[..., 0], feather_edge)

    for i in range(1, n_psf):
        mask[..., i] = np.rot90(mask[..., 0], k=i)

    # Normalize the weig
    mask[:] /= np.sum(mask, axis=-1)[..., np.newaxis]

    return mask


def simulate(im, psf_modes, mask, noise_level: float = 1e-2):
    _x = px.Variable(im.shape)
    blur_op = weighted_planewise_conv(psf_modes, mask, im.shape, n_psf)(_x)
    im_sb = np.empty(im.shape, dtype=np.float32, order='F')
    blur_op.forward([im], [im_sb])

    im_sb += noise_level * im_sb.max() * np.random.randn(*im.shape)

    return im_sb


# Sample Data (synthetic or load from file)
imwidth = 512
kwidth = 11
n_psf = 4

raw = get_test_image(imwidth)
draw(raw, 1, 'Raw')

psf_modes = get_psf(n_psf, kwidth)
mask = get_mask(n_psf, imwidth)
measured = simulate(raw, psf_modes, mask)

# Delete the raw image to ensure the information doesn't leak to the solver
del raw

# Define the problem
mu, alpha, maxiter = 2e-3, 0.9, 200
x = px.Variable(measured.shape)
using_weights_then_conv = True
if using_weights_then_conv:
    prob = px.Problem(
        FlickerAndRigaut2005(x,
                             measured,
                             psf_modes,
                             mask,
                             mu=mu,
                             alpha=alpha,
                             n_psf=n_psf))
else:
    prob = px.Problem(
        NagyAndOLeary1998(x, measured, psf_modes, mask, mu=mu, n_psf=n_psf))

prob.solve(solver='pc',
           eps_abs=1e-3,
           max_iters=maxiter,
           conv_check=20,
           verbose=True,
           x0=measured)

for i in range(4):
    draw(psf_modes[..., i], 5 + i, f'camera shake[{i}]')
    draw(mask[..., i], 9 + i, f'weight[{i}]')

draw(measured, 2, 'measured')
draw(x.value, 3, 'reconstructed')
plt.show()
