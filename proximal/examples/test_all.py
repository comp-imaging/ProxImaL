

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


############################################################
numIterations = 10

#img = lena().astype(np.uint8)
#img = img.copy().reshape(img.shape[0],img.shape[1],1)

# Load image
img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet
# img = Image.open('./data/largeimage_pow2.png')  # opens the file using Pillow - it's not an array yet
#img = Image.new("RGB", (1,10000000), "white")
np_img = np.asfortranarray(im2nparray(img))

#np_img = np.array([[[1.1,0.7],[1.5,1.0]],[[1.1,1.0],[1.0,1.0]]], dtype=np.float32, order='FORTRAN')
#np_img = np.mean( np_img, axis=2)
print 'Type ', np_img.dtype, 'Shape', np_img.shape
output = np.array([0.0], dtype=np.float32)
print 'Type ', output.dtype, 'Shape', output.shape

############################################################
# NORM2
############################################################
output_ref_reordered = sqrt(np.sum(np.sum(np_img * np_img, 1)))
print('ref reordered: ', output_ref_reordered)

hl_2D = Halide('A_norm_L2.cpp', generator_name="normL2Img", recompile=True,
               verbose=False, cleansource=True)  # Force recompile in local dir
hl_1D = Halide('A_norm_L2.cpp', generator_name="normL2_1DImg", func="A_norm_L2_1D",
               recompile=True, verbose=False, cleansource=True)  # Force recompile in local dir
hl_norm2 = hl_2D.A_norm_L2
if np_img.shape[1] < 8:
    hl_norm2 = hl_1D.A_norm_L2_1D

hl_norm2(np_img, output)  # Dummy call (to load dll?)

timeNorm2_halide = 0.0
timeNorm2_numpy = 0.0
for x in range(0, numIterations):
# print "x %d" % (x)
    tic()
    hl_norm2(np_img, output)  # Call
    timeNorm2_halide += toc()

    # run numpy reference
    tic()
    output_ref = np.linalg.norm(np_img.ravel(), 2)
    timeNorm2_numpy += toc()

timeNorm2_halide /= numIterations
timeNorm2_numpy /= numIterations
print('Running time norm2_halide took: {0:.1f}ms'.format(timeNorm2_halide))
print('Running time norm2_numpy took: {0:.1f}ms'.format(timeNorm2_numpy))


############################################################
# DOT
############################################################
np_img0 = np.mean(np_img, axis=2)
np_img1 = np_img0

output_ref_reordered = np.sum(np.sum(np_img0 * np_img1, 1))
print('ref reordered: ', output_ref_reordered)

hl_2D = Halide('A_dot_prod.cpp', generator_name="dotImg", recompile=True,
               verbose=False, cleansource=True)  # Force recompile in local dir
hl_1D = Halide('A_dot_prod.cpp', generator_name="dot_1DImg", func="A_dot_1D",
               recompile=True, verbose=False, cleansource=True)  # Force recompile in local dir
hl_dot = hl_2D.A_dot_prod
if np_img.shape[1] < 8:
    hl_dot = hl_1D.A_dot_1D

hl_dot(np_img0, np_img1, output)  # Dummy call (to load dll?)

timeDot_halide = 0.0
timeDot_numpy = 0.0
for x in range(0, numIterations):
# print "x %d" % (x)
    tic()
    hl_dot(np_img0, np_img1, output)  # Call
    timeDot_halide += toc()

    # run numpy reference
    output_ref = np.dot(np_img0.ravel(), np_img1.ravel())
    timeDot_numpy += toc()

timeDot_halide /= numIterations
timeDot_numpy /= numIterations
print('Running time dot_halide took: {0:.1f}ms'.format(timeDot_halide))
print('Running time dot_numpy took: {0:.1f}ms'.format(timeDot_numpy))
