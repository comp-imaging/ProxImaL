

# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena
from numpy.fft import fftn, ifftn
import cv2


def complex_mult(a, b):
    """ Complex array multiplication for plane array a and b """

    c = np.zeros_like(a)

    #re(a) * re(b) - im(a) * im(b),
    c[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]

    #re(a) * im(b) + im(a) * re(b)
    c[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]

    return c

# Load image
img = Image.open('./data/angela_large.jpg')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
np_img = np.mean(np_img, axis=2)
print 'Type ', np_img.dtype, 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()

# Kernel
K = Image.open('./data/kernel_snake.png')  # opens the file using Pillow - it's not an array yet
K = np.mean(im2nparray(K), axis=2)
K = np.maximum(cv2.resize(K, (15, 15), interpolation=cv2.INTER_LINEAR), 0)
K /= np.sum(K)
K = np.asfortranarray(K)

plt.figure()
imgplot = plt.imshow(K / np.amax(K), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('K')
plt.colorbar()
plt.show()

######################################################################
# NUMPY FFT
######################################################################

# Pad K if necessary
if len(K.shape) < len(np_img.shape):
    K = np.asfortranarray(np.stack((K,) * np_img.shape[2], axis=-1))

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
        Kx_ref[:, :, c] = signal.convolve2d(
            np_img[:, :, c], K[:, :, c], mode='same', boundary='wrap')
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
Halide('A_conv.cpp').A_conv(np_img, K, output)  # Call
print('Running Halide spatial conv took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(Kx_ref, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Kx_ref')
plt.colorbar()
plt.show()

#print('Maximum Halide spatial error {0}'.format( np.amax( np.abs( output - Kx_fft_ref ) ) ) )

######################################################################
# #Test the fft in halide
######################################################################

# Check for low length
pad = False
if len(np_img.shape) == 2:
    pad = True
    np_img = np.asfortranarray(np_img[..., np.newaxis])
    K = np.asfortranarray(K[..., np.newaxis])

hsize = np_img.shape
output_fft = np.zeros(((hsize[0] + 1) / 2 + 1, hsize[1], hsize[2], 2), dtype=np.float32, order='F')
output_kfft = np.zeros_like(output_fft)
output_Kf = np.zeros_like(np_img)
hflags = ['-DWTARGET={0} -DHTARGET={1}'.format(hsize[1], hsize[0])]

# Recompile
#Halide('fft2_r2c.cpp', compile_flags=hflags, recompile=True)
#Halide('ifft2_c2r.cpp', compile_flags=hflags, recompile=True)

tic()
Halide('fft2_r2c.cpp', compile_flags=hflags).fft2_r2c(np_img, 0, 0, output_fft)  # Call
Halide('fft2_r2c.cpp', compile_flags=hflags).fft2_r2c(
    K, K.shape[1] / 2, K.shape[0] / 2, output_kfft)  # Call


np_imghat = 1j * output_fft[..., 1]
np_imghat += output_fft[..., 0]

Khat = 1j * output_kfft[..., 1]
Khat += output_kfft[..., 0]

Kmultres = Khat * np_imghat
Khat = np.asfortranarray(np.stack((Kmultres.real, Kmultres.imag), axis=-1))
#Kx_fft = complex_mult(output_fft, output_kfft)

Halide('ifft2_c2r.cpp', compile_flags=hflags).ifft2_c2r(Khat, output_Kf)  # Call

print('Running Halide FFT took: {0:.1f}ms'.format(toc()))

if pad:
    np_img = np_img[:, :, 0]
    output_Kf = output_Kf[:, :, 0]

plt.figure()
imgplot = plt.imshow(output_Kf, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('output_Kf')
plt.colorbar()
plt.show()

print('Maximum Halide FFT error {0}'.format(np.amax(np.abs(output_Kf - Kx_fft_ref))))

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

# Wait until done
raw_input("Press Enter to continue...")

exit()
