# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *

import numpy as np
from scipy import signal
from scipy.misc import face
import matplotlib.pyplot as plt

############################################################

# Load image
np_img = get_test_image(512, color=True)

plt.figure()
plt.subplot(221)
plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0), cmap='gray')
plt.title('Numpy')

tic()
fn = mul_color(Variable(np_img.shape), mode='yuv')
output = np.empty(np_img.shape, dtype=np.float32, order='F')
fn.forward([np_img], [output])
print('Running color transform took: {0:.1f}ms'.format(toc()))

mi = np.amin(output)
ma = np.amax(output)
print('Y colorspace Min/Max are: [{:0.1f}, {:0.1f}]'.format(mi, ma))

plt.subplot(222)
plt.imshow(np.maximum(output[..., 0], 0.0),
           interpolation="nearest",
           clim=(0.0, 1.0),
           cmap='gray')
plt.colorbar()
plt.title('Y colorspace')

plt.subplot(223)
plt.imshow(output[..., 1],
           interpolation="nearest",
           cmap='gray')
plt.colorbar()
plt.title('U colorspace')

plt.subplot(224)
plt.imshow(output[..., 2],
           interpolation="nearest",
           cmap='gray')
plt.colorbar()
plt.title('V colorspace')
plt.show()
