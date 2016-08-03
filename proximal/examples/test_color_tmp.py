

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
from PIL import Image
from scipy.misc import lena

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
# print 'Type ', np_img.dtype , 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()

tic()
#T = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32, order='F')
fn = mul_color(Variable(np_img.shape), mode='yuv')
output = np.zeros(np_img.shape, dtype=np.float32, order='F')
fn.forward([np_img], [output])
print('Running color transform took: {0:.1f}ms'.format(toc()))

plt.figure()
mi = np.amin(output)
ma = np.amax(output)
print 'Min/Max are: ', mi, ma
imgplot = plt.imshow(np.maximum(output, 0.0), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Output from mul_color')
plt.show()

# Wait until done
raw_input("Press Enter to continue...")
