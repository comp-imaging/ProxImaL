

# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *

import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
np_img = np.asfortranarray(im2nparray(img))

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()

tic()
Halide('A_conv.cpp', recompile=True, verbose=False,
       cleansource=True)  # Force recompile in local dir
print('Compilation took: {0:.1f}ms'.format(toc()))

# Test the runner
output = np.zeros_like(np_img)
kimg = Image.open('./data/kernel_snake.png')  # opens the file using Pillow - it's not an array yet
kimg = np.asfortranarray(im2nparray(kimg))
K = np.asfortranarray(kimg)  # K[2:-2,:] )
K[:, :, 0] /= np.sum(K[:, :, 0])
K[:, :, 1] /= np.sum(K[:, :, 1])
K[:, :, 2] /= np.sum(K[:, :, 2])

K[:, :, 1] = np.flipud(K[:, :, 1])
K[:, :, 2] = np.fliplr(K[:, :, 2])

output_scipy = np.zeros(np_img.shape, dtype=np.float32, order='F')
tic()
for j in range(K.shape[2]):
    output_scipy[:, :, j] = ndimage.convolve(np_img[:, :, j], K[:, :, j], mode='wrap')
print('Running Scipy.convolve2d took: {0:.1f}ms'.format(toc()))

fn = conv(K, Variable(np_img.shape), dims=2, implem='numpy')
output_ref = np.zeros(np_img.shape, dtype=np.float32, order='F')

tic()
fn.forward([np_img], [output_ref])
print('Running conv fft convolution took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output_ref, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from conv')
plt.show()

plt.figure()
imgplot = plt.imshow(output_scipy, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from Scipy')
plt.show()

# Error
print('Maximum error correlation {0}'.format(np.amax(np.abs(output_scipy - output_ref))))

# Wait until done
raw_input("Press Enter to continue...")
