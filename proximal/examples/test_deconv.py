# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.metrics import *
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
I = np.asfortranarray(im2nparray(img))
I = np.maximum(cv2.resize(I, (2048, 2048), interpolation=cv2.INTER_LINEAR), 0)
I = np.mean(I, axis=2)
I = np.asfortranarray(I)
I = np.maximum(I, 0.0)

# Kernel
K = Image.open('./data/kernel_snake.png')  # opens the file using Pillow - it's not an array yet
K = np.mean(np.asfortranarray(im2nparray(K)), axis=2)
K = np.maximum(cv2.resize(K, (15, 15), interpolation=cv2.INTER_LINEAR), 0)
K /= np.sum(K)

# Generate observation
sigma_noise = 0.01
b = ndimage.convolve(I, K, mode='wrap') + sigma_noise * np.random.randn(*I.shape)

# Create a mask for fun
#mask = np.zeros(b.shape)
#mask[::2,::2] = 1.
#b *= mask

# Display data
plt.ion()
plt.figure()
imgplot = plt.imshow(I, interpolation="nearest", clim=(0.0, 1.0))
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

# Sparse gradient deconvolution without quadratic
# x = Variable( x.shape )
# options = cg_options(tol=1e-5, num_iters=50, verbose=False)
# #nonquad_fns = [sum_squares(conv(K, x), b=b, alpha = 10000), norm1( grad(x, dims = 2) ) ] #Isotropic
# nonquad_fns = [sum_squares(conv(K, x), b=b, alpha = 10000), group_norm1( grad(x, dims = 2), [2] ) ] #Anisotropic
# prox_fns = nonquad_fns + quad_funcs

# hqs(prox_fns, lin_solver="cg", lin_solver_options = options,
# 				eps_rel=1e-6, max_iters = 10, max_inner_iters = 10, x0 = b, verbose=True)

# # Sparse gradient deconvolution with quadratic definition
# x = Variable( x.shape )
# quad_funcs = [sum_squares(conv(K, x), b=b, alpha = 400.0)]
# #nonquad_fns = [ norm1( grad(x, dims = 2) ) ] #Anisotropic
# nonquad_fns = [group_norm1( grad(x, dims = 2), [2] )] #Isotropic
# prox_fns = nonquad_fns + quad_funcs

# options = cg_options(tol=1e-3, num_iters=100, verbose=False)
# hqs(prox_fns, quad_funcs=quad_funcs, lin_solver="cg", lin_solver_options = options,
# 			eps_rel=1e-3, max_iters = 10, max_inner_iters = 10, x0 = b, verbose=True)

# # Sparse gradient deconvolution with quadratic definition using deconvolution
# x = Variable( x.shape )
# quad_funcs = [sum_squares(conv(K, x), b=b, alpha = 400.0)]
# dx = np.array([[0.0, 1.0, -1.0]])
# dy = np.array([[0.0], [1.0], [-1.0]])
# nonquad_fns = [norm1(conv(dx, x)), norm1(conv(dy, x))] #Isotropic
# prox_fns = nonquad_fns + quad_funcs

# options = cg_options(tol=1e-3, num_iters=100, verbose=False)
# hqs(prox_fns, quad_funcs=quad_funcs, lin_solver="cg", lin_solver_options = options,
# 			eps_rel=1e-3, max_iters = 10, max_inner_iters = 10, x0 = b, verbose=True)


# Recompile
hflags = ['-DWTARGET={0} -DHTARGET={1}'.format(I.shape[1], I.shape[0])]
Halide('fft2_r2c.cpp', compile_flags=hflags, recompile=True)
Halide('ifft2_c2r.cpp', compile_flags=hflags, recompile=True)

# Sparse gradient deconvolution with quadratic definition
lambda_tv = 1.0
lambda_data = 500.0

x = Variable(I.shape)
#quad_funcs = [sum_squares(mul_elemwise(mask, conv(K, x)), b=b, alpha = lambda_data)]
quad_funcs = [sum_squares(conv(K, x, implem='numpy', dims=2), b=b, alpha=lambda_data)]
nonquad_fns = [group_norm1(grad(x, dims=2, implem='numpy'), [2],
                           alpha=lambda_tv, implem='numpy')]  # Isotropic

# quad_funcs = [group_norm1( grad(x, dims = 2), [2], alpha = lambda_tv ), sum_squares(conv(K, x), b=b, alpha = lambda_data)] #Isotropic
#nonquad_fns = []

# Output PSNR metric
psnrval = psnr_metric(I, pad=(10, 10), decimals=2)

# Prox functions are the union
prox_fns = nonquad_fns + quad_funcs

options = cg_options(tol=1e-5, num_iters=100, verbose=False)
#options = lsqr_options(atol=1e-5, btol=1e-5, num_iters=100, verbose=False)
tic()

pc(prox_fns, quad_funcs=quad_funcs, tau=0.088, sigma=1.000, theta=1.000, max_iters=100,
   eps=1e-2, lin_solver="cg", lin_solver_options=options, metric=None, verbose=1)

print('Overall solver took: {0:.1f}ms'.format(toc()))


plt.figure()
imgplot = plt.imshow(x.value, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Results from Scipy')
plt.show()

# Wait until done
raw_input("Press Enter to continue...")
