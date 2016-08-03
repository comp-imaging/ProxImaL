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
from scipy import signal

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
K = cv2.resize(K, (15, 15), interpolation=cv2.INTER_LINEAR)
K /= np.sum(K)

tic()
b = signal.convolve2d(x, K, mode='same', boundary='wrap')
print('Running Scipy.convolve2d took: {0:.1f}ms'.format(toc()))

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
plt.title('Output from Scipy')
plt.show()

# Now test the solver with some sparse gradient deconvolution
x = Variable(x.shape)
prox_fns = [nonneg(x), sum_squares(conv(K, x), b=b, alpha=5000)]
hqs(prox_fns, eps_rel=1e-6, max_iters=100, max_inner_iters=10, x0=b, verbose=True)

plt.figure()
imgplot = plt.imshow(x.value, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Results from Scipy')
plt.show()

print('Minimum {0}', np.amin(x.value))

# Wait until done
raw_input("Press Enter to continue...")
