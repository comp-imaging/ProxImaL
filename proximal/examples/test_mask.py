

# Proximal
import sys
sys.path.append('../../')

from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')

from proximal.utils.utils import *
from proximal.halide.halide import *

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena

############################################################

#img = lena().astype(np.uint8)
#img = img.copy().reshape(img.shape[0],img.shape[1],1)

# Load image
img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet
# img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
np_img = np.mean(np_img, axis=2)
# print 'Type ', np_img.dtype , 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
# plt.show()
plt.savefig('mask0.png')

# Test the runner
output = np.zeros_like(np_img)
mask = np.asfortranarray(np.random.randn(*list(np_img.shape)).astype(np.float32))
mask = np.maximum(mask, 0.)
#mask[mask < 0.5] = 0.
#mask[mask >= 0.5] = 1.
print 'Type ', mask.dtype, 'Shape', mask.shape

# Halide('A_mask.cpp', recompile=True) #Call

tic()
hl = Halide('A_mask.cpp', recompile=True, verbose=False, cleansource=True)  # Force recompile
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.A_mask(np_img, mask, output)  # Call
print('Running halide (first) run took: {0:.1f}ms'.format(toc()))


tic()
hl.A_mask(np_img, mask, output)  # Call
print('Running halide (second) run took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from Halide')
# plt.show()
plt.savefig('mask1.png')

tic()
output_ref = mask * np_img
print('Running mask in scipy took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))

plt.figure()
imgplot = plt.imshow(output_ref, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output from numpy')
# plt.show()
plt.savefig('mask2.png')

############################################################################
# Check transpose
############################################################################

output_trans = np.zeros_like(np_img)

tic()
hl = Halide('At_mask.cpp', recompile=True, verbose=False, cleansource=True)  # Force recompile
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.At_mask(np_img, mask, output_trans)  # Call
print('Running trans (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.At_mask(np_img, mask, output_trans)  # Call
print('Running trans (second) took: {0:.1f}ms'.format(toc()))


# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output_trans))))
