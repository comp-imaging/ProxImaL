# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *

import numpy as np
import matplotlib.pyplot as plt

############################################################

# Load image
np_img = get_test_image(512)
print('Type ', np_img.dtype, 'Shape', np_img.shape)

plt.figure()
plt.subplot(221)
plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0), cmap='gray')
plt.title('Numpy')

# Test the runner
output = np.empty(np_img.shape, dtype=np.float32, order='F')

mask = np.asfortranarray(np.random.randn(*(np_img.shape)), dtype=np.float32)
mask[:] = np.maximum(mask, 0.)

tic()
hl = Halide('A_mask', recompile=True)  # Force recompile
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.A_mask(np_img, mask, output)  # Call
print('Running halide (first) run took: {0:.1f}ms'.format(toc()))

tic()
hl.A_mask(np_img, mask, output)  # Call
print('Running halide (second) run took: {0:.1f}ms'.format(toc()))

plt.subplot(222)
plt.imshow(output, interpolation="nearest", clim=(0.0, 1.0), cmap='gray')
plt.title('Output from Halide')

tic()
output_ref = mask * np_img
print('Running mask in scipy took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error {0}'.format(
    np.linalg.norm(output_ref.ravel() - output.ravel(), np.Inf)))

plt.subplot(223)
plt.imshow(output_ref, interpolation="nearest", clim=(0.0, 1.0), cmap='gray')
plt.title('Output from numpy')

############################################################################
# Check transpose
############################################################################

output_trans = np.zeros_like(np_img)

tic()
hl = Halide('At_mask', recompile=True)  # Force recompile
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.At_mask(np_img, mask, output_trans)  # Call
print('Running trans (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.At_mask(np_img, mask, output_trans)  # Call
print('Running trans (second) took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error {0}'.format(
    np.linalg.norm(output_ref.ravel() - output_trans.ravel(), np.Inf)))

plt.show()