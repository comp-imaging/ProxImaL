

# Proximal
import sys
sys.path.append('../../')

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
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
np_img = im2nparray(img)
#np_img = np.mean( np_img, axis=2)
np_img = np.asfortranarray(np.tile(np_img[..., np.newaxis], (1, 1, 1, 3)))

print 'Type ', np_img.dtype, 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np.reshape(np_img, (np_img.shape[0], np_img.shape[
                     1] * np_img.shape[2] * np_img.shape[3]), order='F'), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()

# Test the runner
output = np.zeros_like(np_img)
mask = np.asfortranarray(np.random.randn(*list(np_img.shape[0:3])).astype(np.float32))
mask = np.maximum(mask, 0.)
#mask[mask < 0.5] = 0.
#mask[mask >= 0.5] = 1.
print 'Type ', mask.dtype, 'Shape mask', mask.shape

# Recompile
# Halide('A_mask.cpp', recompile=True) #Call

tic()
for k in range(np_img.shape[3]):
    Halide('A_mask.cpp').A_mask(np.asfortranarray(
        np_img[:, :, :, k]), mask, output[:, :, :, k])  # Call

print('Running took: {0:.1f}ms'.format(toc()))


tic()
output_ref = np.zeros_like(np_img)
for k in range(np_img.shape[3]):
    output_ref[:, :, :, k] = mask * np_img[:, :, :, k]

print('Running mask in scipy took: {0:.1f}ms'.format(toc()))

# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))

# Wait until done
raw_input("Press Enter to continue...")
