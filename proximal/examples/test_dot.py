

# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *

from math import sqrt
import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena

halide_src = 'A_dot_prod.cpp'

############################################################

#img = lena().astype(np.uint8)
#img = img.copy().reshape(img.shape[0],img.shape[1],1)

# Load image
img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet
# img = Image.open('./data/largeimage_pow2.png')  # opens the file using
# Pillow - it's not an array yet
img = Image.new("RGB", (1, 10000000), "white")

np_img = np.asfortranarray(im2nparray(img))
#np_img = np.array([[[1.1,0.7],[1.5,1.0]],[[1.1,1.0],[1.0,1.0]]], dtype=np.float32, order='FORTRAN')
np_img0 = np.mean(np_img, axis=2)
np_img1 = np_img0

print('Type ', np_img.dtype, 'Shape', np_img.shape)
output = np.array([0.0], dtype=np.float32)
print('Type ', output.dtype, 'Shape', output.shape)

output_ref_reordered = np.sum(np.sum(np_img0 * np_img1, 1))
print('ref reordered: ', output_ref_reordered)

tic()
# hl = Halide(halide_src, recompile=True, verbose=False, cleansource=True)
# #Force recompile in local dir
hl_2D = Halide(halide_src, generator_name="dotImg", recompile=True,
               verbose=False, cleansource=True)  # Force recompile in local dir
hl_1D = Halide(halide_src, generator_name="dot_1DImg", func="A_dot_1D", recompile=True,
               verbose=False, cleansource=True)  # Force recompile in local dir
hl_dot = hl_2D.A_dot_prod
if np_img.shape[1] < 8:
    hl_dot = hl_1D.A_dot_1D
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl_dot(np_img0, np_img1, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

tic()
hl_dot(np_img0, np_img1, output)  # Call
print('Running second time took: {0:.1f}ms'.format(toc()))

output = output[0]

tic()
output_ref = np.dot(np_img0.ravel(), np_img1.ravel())
print('Running numpy .dot took: {0:.1f}ms'.format(toc()))

# Ours vs ref
print('Ref vs. our output: ', output_ref, output)


# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))
