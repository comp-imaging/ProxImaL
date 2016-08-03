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
x = np.maximum(x, 0.0)

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
imgplot = plt.imshow(x, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Original Image')
plt.show()

plt.figure()
imgplot = plt.imshow(K / np.amax(K), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('K')
plt.show()

plt.figure()
imgplot = plt.imshow(b, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Observation')
plt.show()

# Sparse gradient deconvolution with quadratic definition
lambda_tv = 1.0
lambda_data = 500.0

x = Variable(x.shape)
quad_funcs = [sum_squares(conv(K, x), b=b, alpha=lambda_data)]
prox_fns = [group_norm1(grad(x, dims=2), [2], alpha=lambda_tv)]  # Isotropic

tic()
# options = lsqr_options(atol=1e-5, btol=1e-5, num_iters=100, verbose=False)
# pc(prox_fns, quad_funcs = quad_funcs, tau=None, sigma=None, theta=None, max_iters=100,
#       eps=1e-2, lin_solver="lsqr", lin_solver_options=options,
# 		try_diagonalize = False, verbose=True)
sdlsqrtime = toc() / 1000

tic()
options = cg_options(tol=1e-5, num_iters=100, verbose=False)
pc(prox_fns, quad_funcs=quad_funcs, sigma=10.0, max_iters=200,
   eps=5e-1, lin_solver="cg", lin_solver_options=options,
   try_diagonalize=False, verbose=True)
sdcgtime = toc() / 1000

plt.figure()
imgplot = plt.imshow(x.value, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Results without diagonalization')
plt.show()

# Sparse gradient deconvolution with quadratic definition
tic()
pc(prox_fns, quad_funcs=quad_funcs, sigma=10.0, max_iters=200,
   eps=5e-1, lin_solver="cg", lin_solver_options=options,
   try_diagonalize=True, verbose=True)
fdtime = toc() / 1000

plt.figure()
imgplot = plt.imshow(x.value, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Results with frequency diagonalization')
plt.show()

# Show output
print('\nRunning in SPATIAL (LSQR) domain: {0:.1f}sec'.format(sdlsqrtime))
print('Running in SPATIAL (CG, warm start) domain: {0:.1f}sec'.format(sdcgtime))
print('Running in FREQUENCY domain: {0:.1f}sec\n'.format(fdtime))

# Wait until done
raw_input("Press Enter to continue...")
