

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

import matplotlib.pyplot as plt
from PIL import Image
import cv2


############################################################

# Load image
# img = Image.open('./data/angela_trans.jpg')  # opens the file using Pillow - it's not an array yet
img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
#np_img = np.mean( np_img, axis=2)
print('Type ', np_img.dtype, 'Shape', np_img.shape)

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
# plt.show()
plt.savefig('warp0.png')
plt.savefig('mask1.png')

# Generate transform
theta_rad = 5.0 * np.pi / 180.0
H = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0.0001],
              [np.sin(theta_rad), np.cos(theta_rad), 0.0003],
              [0., 0., 1.]], dtype=np.float32, order='F')


tic()
# Reference
# output_ref = cv2.warpPerspective(np_img, H.T, np_img.shape[1::-1], flags=cv2.INTER_LINEAR,
#                    		borderMode=cv2.BORDER_CONSTANT, borderValue=0.) #cv2.WARP_INVERSE_MAP,
var = Variable(np_img.shape)
fn = warp(var, H)  # 2d Gradient
output_ref = np.zeros(np_img.shape, dtype=np.float32, order='F')
fn.forward([np_img], [output_ref])
print('Running cv2.warpPerspective took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output_ref, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from CV2')
# plt.show()
plt.savefig('warp1.png')

# Test halide interface
output = np.zeros_like(np_img)
Hc = np.asfortranarray(np.linalg.pinv(H)[..., np.newaxis])  # Third axis for halide

# Compile
Halide('A_warp.cpp', recompile=True, cleansource=False)  # Force recompile

tic()
Halide('A_warp.cpp').A_warp(np_img, Hc, output)  # Call
print('Running halide took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from halide')
# plt.show()
plt.savefig('warp2.png')

# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))


############################################################################
# Check correlation
############################################################################

output_trans = np.zeros_like(np_img)
Hinvc = np.asfortranarray(H[..., np.newaxis])  # Third axis for halide

Halide('At_warp.cpp', recompile=True, cleansource=False)  # Force recompile

tic()
Halide('At_warp.cpp').At_warp(output, Hinvc, output_trans)  # Call
print('Running correlation took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output_trans, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output trans from halide')
# plt.show()
plt.savefig('warp3.png')

# Compute reference
tic()
# Reference
# output_ref_trans = cv2.warpPerspective(output_ref, H.T, np_img.shape[1::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
#                    		borderMode=cv2.BORDER_CONSTANT, borderValue=0.)

# Adjoint.
output_ref_trans = np.zeros(var.shape, dtype=np.float32, order='F')
fn.adjoint([output_ref], [output_ref_trans])
print('Running cv2.warpPerspective took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output_ref_trans, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output trans reference')
# plt.show()
plt.savefig('warp4.png')

# Error
print('Maximum error trans {0}'.format(np.amax(np.abs(output_ref_trans - output_trans))))
