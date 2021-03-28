

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

halide_src = 'A_norm_L2.cpp'

############################################################

#img = lena().astype(np.uint8)
#img = img.copy().reshape(img.shape[0],img.shape[1],1)

# Load image
# img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet
# img = Image.open('./data/largeimage_pow2.png')  # opens the file using Pillow - it's not an array yet
#img = Image.new("RGB", (1,10000000), "white")
np_img = np.asfortranarray(im2nparray(img).astype(np.float32))

#np_img = np.array([[[1.1,0.7],[1.5,1.0]],[[1.1,1.0],[1.0,1.0]]], dtype=np.float32, order='FORTRAN')
#np_img = np.mean( np_img, axis=2)
print('Type ', np_img.dtype, 'Shape', np_img.shape)
output = np.array([0.0], dtype=np.float32)
print('Type ', output.dtype, 'Shape', output.shape)

output_ref_reordered = sqrt(np.sum(np.sum(np_img * np_img, 1)))
print('ref reordered: ', output_ref_reordered)

tic()
hl_2D = Halide(halide_src, generator_name="normL2Img", recompile=True,
               verbose=False, cleansource=True)  # Force recompile in local dir
# hl_1D = Halide(halide_src, generator_name="normL2_1DImg", func="A_norm_L2_1D", recompile=True, verbose=False, cleansource=True) #Force recompile in local dir
# hl = Halide(halide_src, recompile=True, verbose=False, cleansource=True)
# #Force recompile in local dir
hl_norm2 = hl_2D.A_norm_L2
# if np_img.shape[1] < 8:
#	hl_norm2 = hl_1D.A_norm_L2_1D
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl_norm2(np_img, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

tic()
hl_norm2(np_img, output)  # Call
print('Running second time took: {0:.1f}ms'.format(toc()))

output = output[0]

tic()
output_ref = np.linalg.norm(np_img.ravel(), 2)
print('Running numpy norm took: {0:.1f}ms'.format(toc()))
print(output_ref.dtype)

# Ours vs ref
print('Ref vs. our output: ', output_ref, output)


# Error
print('Maximum error {0}'.format(np.amax(np.abs(output_ref - output))))
