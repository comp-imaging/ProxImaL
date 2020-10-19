# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import numpy as np
from scipy import ndimage
from scipy.misc import ascent

import matplotlib.pyplot as plt
import cv2

############################################################

# Load image
WIDTH = 2048
I = get_test_image(WIDTH)

# Kernel
K = get_kernel(31, len(I.shape))

# Generate observation
sigma_noise = 0.01
b = ndimage.convolve(I, K,
                     mode='wrap')
b += sigma_noise * np.random.randn(*I.shape)

# Display data
plt.figure()
plt.subplot(221)
imgplot = plt.imshow(I, cmap='gray', interpolation='nearest', clim=(0.0, 1.0))
plt.title('Original Image')

plt.subplot(222)
imgplot = plt.imshow(K,
                     cmap='gray',
                     interpolation='nearest',
                     clim=(0.0, K.max()))
plt.title('K')

plt.subplot(223)
imgplot = plt.imshow(b, cmap='gray', interpolation='nearest', clim=(0.0, 1.0))
plt.title('Observation')

lambda_tv = 1.0
lambda_data = 500.0

tic()
Halide('fft2_r2c', target_shape=(WIDTH, WIDTH), recompile=True, reconfigure=True)
Halide('ifft2_c2r', recompile=True)
print('Halide compile time = {:.0f}ms'.format(toc()))

x = Variable(I.shape)

options = cg_options(tol=1e-4, num_iters=100, verbose=False)
tic()

solver_options = dict(
    max_iters=300,
    eps_abs=1e-2,
    eps_rel=1e-2,
    lin_solver_options=options,
    metric=None,
    verbose=True,
)

hl = Impl['halide']

prob = Problem(
    sum_squares(conv(K, x, dims=2, implem=hl) - b) * lambda_data +
    lambda_tv *
    group_norm1(grad(x, dims=2, implem=hl), [2], implem=hl),
    implem=hl,
    lin_solver='cg',
)
result = prob.solve(solver='ladmm', **solver_options)

print('Overall solver took: {:.1f}ms; cost function = {}'.format(toc(), result))

plt.subplot(224)
imgplot = plt.imshow(x.value,
                     cmap='gray',
                     interpolation='nearest',
                     clim=(0.0, 1.0))
plt.title('Results')
plt.show()