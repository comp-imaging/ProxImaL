# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import cv2

############################################################

# Load image
WIDTH = 512
x = get_test_image(WIDTH)

# Scale
scale = 1.
x = np.maximum(x, 0.0) * scale

# Kernel
K = get_kernel(15, x.ndim)

# Generate observation
sigma_noise = 0.01
b = ndimage.convolve(x, K, mode='wrap')

# Simulate multiplative shot noise
peak_photon_count = np.float32(sigma_noise**-2)
b = np.random.poisson(b * peak_photon_count) / peak_photon_count

# Display data
plt.figure()
plt.subplot(221)
plt.imshow(x / scale, interpolation="nearest", clim=(0.0, 1.0), cmap='gray')
plt.title('Original Image')

plt.subplot(222)
imgplot = plt.imshow(K / np.amax(K),
                     interpolation="nearest",
                     clim=(0.0, 1.0),
                     cmap='gray')
plt.title('K')

plt.subplot(223)
plt.imshow(b / scale, interpolation="nearest", clim=(0.0, 1.0), cmap='gray')
plt.title('Observation')

# Sparse gradient deconvolution with quadratic definition
lambda_tv = 0.15
lambda_data = 4000.0

x = Variable(x.shape)
prox_fns = [
    poisson_norm(conv(K, x, implem=Impl['halide']),
                 b,
                 alpha=lambda_data,
                 implem=Impl['halide']),
    group_norm1(grad(x, dims=2, implem=Impl['halide']), [2],
                alpha=lambda_tv,
                implem=Impl['halide'])
]  # Isotropic
quad_funcs = []

# Method options
method = 'lin-admm'
diag = True
verbose = 1
options = cg_options(tol=1e-5, num_iters=100, verbose=False)

tic()
if method == 'pc':

    Halide('fft2_r2c',
           target_shape=(WIDTH, WIDTH),
           recompile=True,
           reconfigure=True)
    Halide('ifft2_c2r', recompile=True)

    pc.solve(prox_fns,
             omega_fns=quad_funcs,
             tau=None,
             sigma=None,
             theta=None,
             max_iters=100,
             eps_abs=1e-3,
             lin_solver="cg",
             lin_solver_options=options,
             try_diagonalize=diag,
             verbose=verbose)

elif method == 'lin-admm':

    ladmm.solve(prox_fns,
                omega_fns=quad_funcs,
                lmb=1,
                max_iters=100,
                eps_abs=1e-4,
                eps_rel=1e-4,
                lin_solver="cg",
                lin_solver_options=options,
                try_diagonalize=diag,
                verbose=verbose)

elif method == 'admm':

    admm.solve(prox_fns,
               omega_fns=quad_funcs,
               rho=1,
               max_iters=100,
               eps_abs=1e-4,
               eps_rel=1e-4,
               lin_solver="cg",
               lin_solver_options=options,
               try_diagonalize=diag,
               verbose=verbose)

elif method == 'hqs':

    hqs.solve(prox_fns,
              omega_fns=[],
              lin_solver="cg",
              lin_solver_options=options,
              eps_rel=1e-6,
              max_iters=10,
              max_inner_iters=10,
              x0=b,
              try_diagonalize=diag,
              verbose=verbose)

print('Running took: {0:.1f}s'.format(toc() / 1000.0))

plt.subplot(224)
plt.imshow(x.value / scale,
           interpolation="nearest",
           clim=(0.0, 1.0),
           cmap='gray')
plt.title('Results from Scipy')
plt.show()
