
# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.halide.halide import *
from proximal.prox_fns import *
from proximal.lin_ops import Variable

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena
import cv2

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
np_img_color = np_img
np_img = np.mean(np_img_color, axis=2)
# print 'Type ', np_img.dtype , 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()


############################################################################
# Test NLM
############################################################################

# #Compile
ext_libs = '-lopencv_core', '-lopencv_imgproc', '-lopencv_cudaarithm', '-lopencv_cudev', '-lopencv_photo', '-lm'
ext_srcs = ['external/external_NLM.cpp']
Halide('prox_NLM.cpp', external_source=ext_srcs, external_libs=ext_libs,
       recompile=True, verbose=False, cleansource=True)  # Compile

# Works currently on color image
v = np_img_color
sigma_fixed = 0.6
lambda_prior = 0.5
sigma_scale = 1.5 * 1
prior = 1.0
params = np.asfortranarray(
    np.array([sigma_fixed, lambda_prior, sigma_scale, prior], dtype=np.float32)[..., np.newaxis])
theta = 0.5

# #Output
output = np.zeros_like(v)

# #Run
tic()
Halide('prox_NLM.cpp').prox_NLM(v, theta, params, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(v, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Input NLM')
plt.show()

# No modifiers.
v = np_img_color
tmp = Variable(v.shape)
fp = patch_NLM(tmp, sigma_fixed=sigma_fixed, sigma_scale=sigma_scale,
               templateWindowSizeNLM=3, searchWindowSizeNLM=11, gamma_trans=1.0,
               prior=prior)  # group over all but first two dims
rho = 1.0 / theta
dst = fp.prox(rho, v.copy())

plt.figure()
imgplot = plt.imshow(dst, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('NLM denoised CV2')
plt.show()

# Error
print('Maximum error NLM (CUDA vs. CPU) {0}'.format(np.amax(np.abs(output - dst))))

############################################################################
# Compute PSNR
############################################################################

ref = np_img_color
print('PSRN: Full {0} dB, Pad {1} dB, Max {2} dB'.format(psnr(output, ref),
      psnr(output, ref, (10, 10)), psnr(output * 255., ref * 255., maxval=255.)))

# Test metric
imgmetric = psnr_metric(ref, pad=(10, 10), decimals=2)

print(imgmetric.message(output))

# Wait until done
raw_input("Press Enter to continue...")
