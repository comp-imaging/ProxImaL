# Proximal
import sys
sys.path.append('../../')

from scipy import ndimage

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import cv2

############################################################

# Load image
np_img = get_test_image(512)

print('Type ', np_img.dtype, 'Shape', np_img.shape)

plt.figure()
plt.subplot(231)
imgplot = plt.imshow(np_img,
                     interpolation="nearest",
                     clim=(0.0, 255.0),
                     cmap='gray')
plt.title('Numpy')

# Generate transform
theta_rad = 5.0 * np.pi / 180.0
H = np.array([[np.cos(theta_rad), -np.sin(theta_rad), -128.],
              [np.sin(theta_rad), np.cos(theta_rad), 0.], [0., 0., 1.]],
             dtype=np.float32,
             order='F')
Hinv = np.asfortranarray(np.linalg.pinv(H))

tic()
# Reference
output_ref = cv2.warpPerspective(np_img,
                                 H,
                                 np_img.shape[1::-1],
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0.)
print('Running cv2.warpPerspective took: {0:.1f}ms'.format(toc()))

plt.subplot(232)
imgplot = plt.imshow(output_ref,
                     interpolation="nearest",
                     clim=(0.0, 255.0),
                     cmap='gray')
plt.title('Output from CV2')

# Test halide interface
output = np.empty(np_img.shape, order='F', dtype=np.float32)
hl = Halide('A_warp', recompile=True)  # Force recompile

tic()
hl.A_warp(np_img, H, output)  # Call
print('Running halide took: {0:.1f}ms'.format(toc()))

plt.subplot(233)
imgplot = plt.imshow(output,
                     interpolation="nearest",
                     clim=(0.0, 255.0),
                     cmap='gray')
plt.title('Output from halide')

# Error
delta = np.linalg.norm(output_ref.ravel() - output.ravel(), np.Inf)
norm = np.amax((output_ref.max(), output.max()))
print('Relative error {0}'.format(delta / norm))

############################################################################
# Check correlation
############################################################################

output_trans = np.zeros_like(np_img)

hl = Halide('At_warp', recompile=True)  # Force recompile

tic()
hl.At_warp(output, Hinv, output_trans)  # Call
print('Running correlation took: {0:.1f}ms'.format(toc()))

plt.subplot(236)
imgplot = plt.imshow(output_trans,
                     interpolation="nearest",
                     clim=(0.0, 255.0),
                     cmap='gray')
plt.title('Output trans from halide')

# Compute reference
tic()
output_ref_trans = cv2.warpPerspective(output_ref,
                                       H,
                                       np_img.shape[1::-1],
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0.)
print('Running cv2.warpPerspective took: {0:.1f}ms'.format(toc()))

plt.subplot(235)
plt.imshow(output_ref_trans,
           interpolation="nearest",
           clim=(0.0, 255.0),
           cmap='gray')
plt.title('Output trans from CV2')

# Error
delta = np.linalg.norm(output_ref_trans.ravel() - output_trans.ravel(), np.Inf)
norm = np.amax((output_ref_trans.max(), output_trans.max()))
print('Relative error trans {0}'.format(delta / norm))
plt.show()
