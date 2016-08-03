# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.convergence_log import *
from proximal.utils.metrics import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import cvxpy as cvx
import numpy as np
from scipy import ndimage

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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

# Setup problem
lambda_tv = 1.0
lambda_data = 500.0

I = x.copy()
#psnrval = psnr_metric( I, pad = (10,10), decimals = 2 )
psnrval = None
x = Variable(I.size)
shaped_x = reshape(x, I.shape)
# Modify with equilibration.
np.random.seed(1)
op1 = grad(shaped_x, dims=2)
op2 = conv(K, shaped_x)
wrand1 = np.random.lognormal(0, 1, size=op1.shape)
wrand2 = np.random.lognormal(0, 1, size=op2.shape)
# wrand = np.ones(op2.shape)
op1 = mul_elemwise(wrand1, op1)
b = wrand2 * b
op2 = mul_elemwise(wrand2, op2)

stacked_ops = vstack([op1, op2])
equil_iters = 100
d, e = equil(CompGraph(stacked_ops), equil_iters, 1e-1, 5)

op1_d = np.reshape(d[:op1.size], op1.shape)  # /wrand1
op2_d = np.reshape(d[op1.size:], op2.shape)  # /wrand2

new_x = mul_elemwise(e, x)
shaped_x = reshape(new_x, I.shape)
op1 = grad(shaped_x, dims=2)
op2 = conv(K, shaped_x)
op1 = mul_elemwise(wrand1, op1)
op2 = mul_elemwise(wrand2, op2)

orig_fns = [norm1(op1, alpha=lambda_tv), sum_squares(op2, b=b, alpha=lambda_data)]
op1 = mul_elemwise(op1_d, op1)
op2 = mul_elemwise(op2_d, op2)

stacked_ops = vstack([op1, op2])
L = est_CompGraph_norm(CompGraph(stacked_ops))
print "||K||_2 = ", L
# Quadratic or non quadratic splitting
print 'Splitting quadratics'

# print np.linalg.norm(new_x.weight)
# op1_d /= np.sqrt(L)
# op2_d /= np.sqrt(L)
# e /= np.sqrt(L)
# print np.linalg.norm(new_x.weight)

nonquad_fns = [weighted_norm1(1 / op1_d, op1, alpha=lambda_tv)]  # Anisotropic
# nonquad_fns = [group_norm1( grad(x, dims = 2), [2], alpha = lambda_tv )] #Isotropic
quad_funcs = [weighted_sum_squares(1 / op2_d, op2, b=op2_d * b, alpha=lambda_data)]

# print 'No splitting'
# #nonquad_fns = [sum_squares(conv(K, x), b=b, alpha = 400), norm1( grad(x, dims = 2), alpha = lambda_tv ) ] #Anisotropic
# nonquad_fns = [sum_squares(conv(K, x), b=b, alpha = 400), group_norm1( grad(x, dims = 2), [2] ) ] #Isotropic
# quad_funcs = []

# In 100 - equil iters.
# 0: 39595062.8522
# 10: 8708627.07193
# 25: 1972630.38285
# 50: 551021.415309
# 75: 385803.229338
# 0/perfect: 85337.5131483

# In 200 - equil iters.
# 0: 34864879.4644
# 100: 75537.9407767
# 0/perfect: 87258.0455795

# Prox functions are the union
prox_fns = nonquad_fns + quad_funcs

method = 'pc'
verbose = 1
diag = False

convlog = ConvergenceLog()

tic()
if method == 'pc':

    options = cg_options(tol=1e-5, num_iters=100, verbose=True)
    #options = lsqr_options(atol=1e-5, btol=1e-5, num_iters=100, verbose=False)
    pc(prox_fns, quad_funcs=[], tau=1 / L, sigma=1 / L, theta=1, max_iters=1000 - equil_iters,
       eps_rel=1e-5, eps_abs=1e-5, lin_solver="cg", lin_solver_options=options,
       try_split=False, try_diagonalize=diag,
       metric=psnrval, verbose=verbose, convlog=None)


elif method == 'lin-admm':

    options = cg_options(tol=1e-5, num_iters=100, verbose=True)
    lin_admm(prox_fns, quad_funcs=quad_funcs, lmb=0.1, max_iters=300,
             eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
             try_diagonalize=diag, metric=psnrval, verbose=verbose)

elif method == 'admm':

    options = cg_options(tol=1e-5, num_iters=100, verbose=True)
    admm(prox_fns, quad_funcs=[], rho=10, max_iters=300,
         eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
         try_diagonalize=diag, metric=psnrval, verbose=verbose)

elif method == 'hqs':

    # Need high accuracy when quadratics are not splitted
    options = cg_options(tol=1e-5, num_iters=100, verbose=True)
    hqs(prox_fns, lin_solver="cg", lin_solver_options=options,
        eps_rel=1e-6, max_iters=10, max_inner_iters=10, x0=b,
        try_diagonalize=diag, metric=psnrval, verbose=verbose)

print convlog.objective_val

print reduce(lambda x, y: x + y, [fn.value for fn in orig_fns])

print('Running took: {0:.1f}s'.format(toc() / 1000.0))

plt.figure()
imgplot = plt.imshow(shaped_x.value, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Result')
plt.show()

# Wait until done
raw_input("Press Enter to continue...")
