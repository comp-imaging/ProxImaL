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


############################################################

# Load image
np_img = get_test_image(2048)
print('Type ', np_img.dtype, 'Shape', np_img.shape)

imgplot = plt.imshow(np_img, interpolation='nearest', clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')

# Force recompile in local dir
tic()
Halide('A_conv', recompile=True)
Halide('At_conv', recompile=True)  # Force recompile in local dir
print('Compilation took: {0:.1f}ms'.format(toc()))

# Test the runner
output = np.zeros_like(np_img)
K = get_kernel(15, len(np_img.shape))

tic()
Halide('A_conv').A_conv(np_img, K, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output, interpolation='nearest', clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from Halide')

tic()
output_scipy = signal.convolve2d(np_img, K, mode='same', boundary='wrap')
print('Running Scipy.convolve2d took: {0:.1f}ms'.format(toc()))

fn = conv(K, Variable(np_img.shape), implem='halide')
output_ref = np.zeros(np_img.shape, dtype=np.float32, order='F')
tic()
fn.forward([np_img], [output_ref])
print('Running conv fft convolution took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))

plt.figure()
imgplot = plt.imshow(output_ref * 255,
                     interpolation='nearest',
                     clim=(0.0, 255.0))
imgplot.set_cmap('gray')
plt.title('Output from Scipy')

############################################################################
# Check correlation
############################################################################

output_corr = np.zeros_like(np_img)
tic()
Halide('At_conv').At_conv(np_img, K, output_corr)  # Call
print('Running correlation took: {0:.1f}ms'.format(toc()))

#output_corr_ref = signal.convolve2d(np_img, np.flipud(np.fliplr(K)), mode='same', boundary='wrap')
output_corr_ref = ndimage.correlate(np_img, K, mode='wrap')

# Adjoint.
output_corr_ref = np.zeros(np_img.shape, dtype=np.float32, order='F')
tic()
fn.adjoint([np_img], [output_corr_ref])
print('Running transpose conv fft convolution took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error correlation {0}'.format(
    np.amax(np.abs(output_corr_ref - output_corr))))
plt.show()
