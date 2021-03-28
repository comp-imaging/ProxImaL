

# Proximal
import sys
sys.path.append('../../')


from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')

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

#img = lena().astype(np.uint8)
#img = img.copy().reshape(img.shape[0],img.shape[1],1)

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
np_img = np.mean(np_img, axis=2)
print('Type ', np_img.dtype, 'Shape', np_img.shape)

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
# plt.show()
plt.savefig('conv0.png')

tic()
Halide('A_conv.cpp', recompile=True, verbose=False,
       cleansource=True)  # Force recompile in local dir
print('Compilation took: {0:.1f}ms'.format(toc()))

# Test the runner
output = np.zeros_like(np_img)
kimg = Image.open('./data/kernel_snake.png')  # opens the file using Pillow - it's not an array yet
kimg = np.asfortranarray(im2nparray(kimg))
K = np.mean(kimg, axis=2)
#K = np.asfortranarray( K[2:-2,:] )
K /= np.sum(K)

# Replicate kernel for multichannel deconv
if len(np_img.shape) == 3 and len(K.shape) == 2:
    K = np.asfortranarray(np.stack((K,) * np_img.shape[2], axis=-1))

tic()
Halide('A_conv.cpp').A_conv(np_img, K, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from Halide')
# plt.show()
plt.savefig('conv1.png')

tic()
output_scipy = signal.convolve2d(np_img, K, mode='same', boundary='wrap')
#output_scipy = ndimage.convolve(np_img,K, mode='wrap')
print('Running Scipy.convolve2d took: {0:.1f}ms'.format(toc()))

fn = conv(K, Variable(np_img.shape), implem='numpy')
output_ref = np.zeros(np_img.shape, dtype=np.float32, order='F')
tic()
fn.forward([np_img], [output_ref])
print('Running conv fft convolution took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))

plt.figure()
imgplot = plt.imshow(output_ref, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from Scipy')
# plt.show()
plt.savefig('conv2.png')

############################################################################
# Check correlation
############################################################################

Halide('At_conv.cpp', recompile=True, verbose=False,
       cleansource=True)  # Force recompile in local dir

output_corr = np.zeros_like(np_img)
tic()
Halide('At_conv.cpp').At_conv(np_img, K, output_corr)  # Call
print('Running correlation took: {0:.1f}ms'.format(toc()))

#output_corr_ref = signal.convolve2d(np_img, np.flipud(np.fliplr(K)), mode='same', boundary='wrap')
output_corr_ref = ndimage.correlate(np_img, K, mode='wrap')

# Adjoint.
output_corr_ref = np.zeros(np_img.shape, dtype=np.float32, order='F')
tic()
fn.adjoint([np_img], [output_corr_ref])
print('Running transpose conv fft convolution took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error correlation {0}'.format(np.amax(np.abs(output_corr_ref - output_corr))))
