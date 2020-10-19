# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *

import numpy as np
import numexpr as ne

from scipy import signal
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn
import cv2

from common import get_test_image, get_kernel

# Load image
np_img = get_test_image(2048)
print('Type ', np_img.dtype, 'Shape', np_img.shape)

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation='nearest', clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()

# Kernel
K = get_kernel(15, len(np_img.shape))

plt.figure()
imgplot = plt.imshow(K / np.amax(K), interpolation='nearest', clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('K')
plt.colorbar()
plt.show()

######################################################################
# NUMPY FFT
######################################################################

# Pad K if necessary
if len(K.shape) < len(np_img.shape):
    K = K[..., (1,) * len(np_img.shape)]

# Convolve reference
tic()
Khat = psf2otf(K, np_img.shape, dims=2)
Kx_fft_ref = ifftd(np.multiply(Khat, fftd(np_img, dims=2)), dims=2).real
print('Running Numpy fft convolution took: {0:.1f}ms'.format(toc()))

######################################################################
# Scipy spatial convolution
######################################################################

Kx_ref = np.zeros_like(np_img)
tic()
if len(np_img.shape) > 2:
    for c in range(np_img.shape[2]):
        Kx_ref[:, :, c] = signal.convolve2d(np_img[:, :, c],
                                            K[:, :, c],
                                            mode='same',
                                            boundary='wrap')
else:
    Kx_ref = signal.convolve2d(np_img, K, mode='same', boundary='wrap')

print('Running Scipy.convolve2d took: {0:.1f}ms'.format(toc()))

#print('Maximum error {0}'.format( np.amax( np.abs( Kx_ref - Kx_fft_ref ) ) ) )

######################################################################
# Halide spatial convolution
######################################################################

# Halide spatial
output = np.zeros_like(np_img)
tic()
Halide('A_conv').A_conv(np_img, K, output)  # Call
print('Running Halide spatial conv took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(Kx_ref, interpolation='nearest', clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Kx_ref')
plt.colorbar()
plt.show()

#print('Maximum Halide spatial error {0}'.format( np.amax( np.abs( output - Kx_fft_ref ) ) ) )

######################################################################
# #Test the fft in halide
######################################################################

# Check for low length
hsize = np_img.shape
freq_shape = (int(
    (hsize[0] + 1) / 2) + 1, hsize[1], (hsize[2] if np_img.shape == 3 else 1))
output_fft = np.empty(freq_shape, dtype=np.complex64, order='F')
output_kfft = np.empty(freq_shape, dtype=np.complex64, order='F')
output_Kf = np.zeros_like(np_img)
target_shape = hsize[:2]

# Recompile
tic()
Halide('fft2_r2c', target_shape=hsize[:2], reconfigure=True)
print('Halide compile time: {:.1f}ms'.format(toc()))

tic()
Halide('fft2_r2c').fft2_r2c(np_img, 0, 0,
                                                      output_fft)  # Call
Halide('fft2_r2c').fft2_r2c(K, int(K.shape[1] / 2),
                                                      int(K.shape[0] / 2),
                                                      output_kfft)  # Call

Kmultres = np.empty(output_fft.shape, dtype=np.complex64)
ne.evaluate('a*b', {
    'a': output_fft,
    'b': output_kfft,
},
            out=Kmultres,
            casting='unsafe')

Halide('ifft2_c2r').ifft2_c2r(Kmultres,
                                                        output_Kf)  # Call

print('Running Halide FFT took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output_Kf, interpolation='nearest', clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('output_Kf')
plt.colorbar()

print('Maximum Halide FFT error {0}'.format(
    np.amax(np.abs(output_Kf - Kx_fft_ref))))

plt.show(block=True)

######################################################################
# #Test the monlithic fft convolution in halide
######################################################################

# #Halide fft
# output_fft = np.zeros_like(np_img);
# tic()
# Halide('fft_conv.cpp', compile_flags=hflags).fft_conv(np_img, K, output_fft) #Call
# print( 'Running full Halide FFT took: {0:.1f}ms'.format( toc() ) )

# #Error
# print('Maximum error fft and spatial {0}'.format( np.amax( np.abs( Kx_fft_ref - output_fft ) ) ) )
# # print('Maximum error fft and halide fft {0}'.format( np.amax( np.abs( Kx_fft_ref - output_fft ) ) ) )

# plt.figure()
# imgplot = plt.imshow(output_fft , interpolation="nearest", clim=(0.0, 1.0))
# imgplot.set_cmap('gray')
# plt.title('Kf FFT')
# plt.colorbar()
# plt.show()
