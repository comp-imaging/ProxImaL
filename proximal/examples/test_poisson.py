# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import cvxpy as cvx
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from PIL import Image
import cv2

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
x = np.asfortranarray(im2nparray(img))
x = np.mean(x, axis=2)

# Scale
scale = 1.
x = np.maximum(x, 0.0) * scale

# Kernel
K = Image.open('./data/kernel_snake.png')  # opens the file using Pillow - it's not an array yet
K = np.mean(np.asfortranarray(im2nparray(K)), axis=2)
K = np.maximum(cv2.resize(K, (15, 15), interpolation=cv2.INTER_LINEAR), 0)
K /= np.sum(K)

# Generate observation
sigma_noise = 0.01
b = ndimage.convolve(x, K, mode='wrap') + sigma_noise * np.random.randn(*x.shape)

# Display data
plt.ion()
plt.figure()
imgplot = plt.imshow(x / scale, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Original Image')
plt.show()

plt.figure()
imgplot = plt.imshow(K / np.amax(K), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('K')
plt.show()

plt.figure()
imgplot = plt.imshow(b / scale, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Observation')
plt.show()

# Sparse gradient deconvolution with quadratic definition
lambda_tv = 0.15
lambda_data = 4000.0

x = Variable(x.shape)
prox_fns = [poisson_norm(conv(K, x), b, alpha=lambda_data),
            group_norm1(grad(x, dims=2), [2], alpha=lambda_tv)]  # Isotropic
quad_funcs = []


# Method options
method = 'lin-admm'
diag = True
verbose = 1
options = cg_options(tol=1e-5, num_iters=100, verbose=False)

tic()
if method == 'pc':

    pc(prox_fns, quad_funcs=quad_funcs, tau=None, sigma=None, theta=None, max_iters=100,
       eps=1e-2, lin_solver="cg", lin_solver_options=options,
       try_diagonalize=diag, verbose=verbose)

elif method == 'lin-admm':

    lin_admm(prox_fns, quad_funcs=quad_funcs, lmb=1, max_iters=100,
             eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
             try_diagonalize=diag, verbose=verbose)

elif method == 'admm':

    admm(prox_fns, quad_funcs=quad_funcs, rho=1, max_iters=100,
         eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
         try_diagonalize=diag, verbose=verbose)

elif method == 'hqs':

    hqs(prox_fns, lin_solver="cg", lin_solver_options=options,
        eps_rel=1e-6, max_iters=10, max_inner_iters=10, x0=b,
        try_diagonalize=diag, verbose=verbose)

print('Running took: {0:.1f}s'.format(toc() / 1000.0))


plt.figure()
imgplot = plt.imshow(x.value / scale, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Results from Scipy')
plt.show()

# Wait until done
raw_input("Press Enter to continue...")
