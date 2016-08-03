# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

#import cvxpy as cvx
import numpy as np
from math import sqrt
import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena
import argparse

#import matplotlib.pyplot as plt
from PIL import Image
import cv2

############################################################
# Parse options
parser = argparse.ArgumentParser(
    description='Deconvolution Splitting test with different algorithms and optimizations.')
parser.add_argument('method', metavar='method', nargs='?', default='pc',
                    help='Algorithm [admm, hqs, pc, lin-admm].')

parser.add_argument('quadratics', metavar='quadratics', type=int, nargs='?', default=0,
                    help='Exploit quadratics in solver.')

parser.add_argument('convolutional', metavar='convolutional', type=int, nargs='?', default=0,
                    help='Compute grad via convolution.')

parser.add_argument('diagonalize', metavar='diagonalize', type=int, nargs='?', default=0,
                    help='Try diagonalizing (in spatial or frequency domain).')

parser.add_argument('verbose', metavar='verbose', type=int, nargs='?', default=1,
                    help='Try diagonalizing (0 - None, 1 - brief, 2 - full (expensive) ).')

args = parser.parse_args()
args.quadratics = args.quadratics != 0
args.convolutional = args.convolutional != 0
diag = args.diagonalize != 0
verbose = args.verbose
print "\n<<<RUNNING method=%s, using quadratics=%d, using convolutional grad=%d, trying to diagonalize=%d, verbose = %d >>>\n\n" % (args.method,
                                                                                                                                    args.quadratics, args.convolutional, diag, verbose)

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
# plt.show()

plt.figure()
imgplot = plt.imshow(K / np.amax(K), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('K')
# plt.show()

plt.figure()
imgplot = plt.imshow(b, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Observation')
# plt.show()

# Derivatives
dx = np.array([[0.0, 1.0, -1.0]])
dy = np.array([[0.0], [1.0], [-1.0]])

# Setup problem
lambda_tv = 1.0
lambda_data = 500.0

I = x.copy()
#psnrval = psnr_metric( I, pad = (10,10), decimals = 2 )
psnrval = None
x = Variable(I.shape)

# Quadratic or non quadratic splitting
if args.quadratics:

    if not args.convolutional:
        print 'Splitting quadratics'
        # nonquad_fns = [norm1( grad(x, dims = 2), alpha = lambda_tv )] #Anisotropic
        nonquad_fns = [group_norm1(grad(x, dims=2), [2], alpha=lambda_tv)]  # Isotropic
        quad_funcs = [sum_squares(conv(K, x), b=b, alpha=lambda_data)]
    else:
        print 'Splitting quadratics and convolutional gradient'
        # Sparse gradient deconvolution with quadratic definition using deconvolution
        quad_funcs = [sum_squares(conv(K, x), b=b, alpha=400.0)]
        nonquad_fns = [norm1(conv(dx, x), alpha=lambda_tv), norm1(
            conv(dy, x), alpha=lambda_tv)]  # Isotropic

else:

    if not args.convolutional:
        print 'No splitting'
        # nonquad_fns = [sum_squares(conv(K, x), b=b, alpha = 400), norm1( grad(x,
        # dims = 2), alpha = lambda_tv ) ] #Anisotropic
        nonquad_fns = [sum_squares(conv(K, x), b=b, alpha=400),
                                   group_norm1(grad(x, dims=2), [2])]  # Isotropic
        quad_funcs = []
    else:
        print 'No splitting and convolutional gradient'
        # Sparse gradient deconvolution with quadratic definition using deconvolution
        nonquad_fns = [sum_squares(conv(K, x), b=b, alpha=400.0), norm1(
            conv(dx, x), alpha=lambda_tv), norm1(conv(dy, x), alpha=lambda_tv)]
        quad_funcs = []

# Prox functions are the union
prox_fns = nonquad_fns + quad_funcs

print prox_fns

tic()
if args.method == 'pc':

    options = cg_options(tol=1e-5, num_iters=100, verbose=False)
    #options = lsqr_options(atol=1e-5, btol=1e-5, num_iters=100, verbose=False)
    pc(prox_fns, quad_funcs=quad_funcs, tau=None, sigma=10.0, theta=None, max_iters=300,
       eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
       try_diagonalize=diag, metric=psnrval, verbose=verbose)


elif args.method == 'lin-admm':

    options = cg_options(tol=1e-5, num_iters=100, verbose=False)
    lin_admm(prox_fns, quad_funcs=quad_funcs, lmb=0.1, max_iters=300,
             eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
             try_diagonalize=diag, metric=psnrval, verbose=verbose)

elif args.method == 'admm':

    options = cg_options(tol=1e-5, num_iters=100, verbose=False)
    admm(prox_fns, quad_funcs=quad_funcs, rho=10, max_iters=300,
         eps_abs=1e-4, eps_rel=1e-4, lin_solver="cg", lin_solver_options=options,
         try_diagonalize=diag, metric=psnrval, verbose=verbose)

elif args.method == 'hqs':

    # Sparse gradient deconvolution without quadratic
    if not args.quadratics:

        # Need high accuracy when quadratics are not splitted
        options = cg_options(tol=1e-5, num_iters=50, verbose=False)
        hqs(prox_fns, lin_solver="cg", lin_solver_options=options,
            eps_rel=1e-6, max_iters=10, max_inner_iters=10, x0=b,
            try_diagonalize=diag, metric=psnrval, verbose=verbose)

    else:

        # Krishnan and Fergus schedule
        options = cg_options(tol=1e-3, num_iters=100, verbose=False)
        hqs(prox_fns, quad_funcs=quad_funcs, lin_solver="cg", lin_solver_options=options,
            eps_rel=1e-3, max_iters=10, max_inner_iters=10, x0=b,
            try_diagonalize=diag, metric=psnrval, verbose=verbose)


print('Running took: {0:.1f}s'.format(toc() / 1000.0))

plt.figure()
imgplot = plt.imshow(x.value, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Results from Scipy')
# plt.show()

# Wait until done
raw_input("Press Enter to continue...")
