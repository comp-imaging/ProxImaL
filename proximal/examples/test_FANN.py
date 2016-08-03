# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.halide.halide import *

import cvxpy as cvx
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from PIL import Image

import matlab.engine
import StringIO
import subprocess

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
I = np.asfortranarray(im2nparray(img))
I = np.maximum(cv2.resize(I, (512, 512), interpolation=cv2.INTER_LINEAR), 0)
#I = np.mean( I, axis=2)
I = np.maximum(I, 0.0)
I = np.asfortranarray(I, dtype=np.float32)

# Generate observation
sigma_noise = 0.1
b = I + sigma_noise * np.random.randn(*I.shape)
b = np.asfortranarray(b, dtype=np.float32)

# Display data
plt.ion()
plt.figure()
imgplot = plt.imshow(I, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Original Image')
plt.show()

plt.figure()
imgplot = plt.imshow(np.clip(b, 0, 1), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Observation')
plt.show()

# Compile
compile = False
if compile:
    make_process = subprocess.Popen(["make"], stderr=subprocess.STDOUT,
                                    cwd="../libs/FastANN/cinterface")
    if make_process.wait() != 0:
        raise Exception("Could not build FastANN.")

    # Compile halide
    ext_libs = '-L../libs/FastANN', '-I../libs/FastANN', '-L/usr/local/cuda-7.5/lib64', '-ldenoise', '-lcudart'
    ext_srcs = ['external/external_FANN.cpp']
    Halide('prox_FANN.cpp', external_source=ext_srcs, external_libs=ext_libs,
           recompile=True, verbose=False, cleansource=True)  # Compile

# Parameters - [Algorithm, Format, (BlockSize, TileSize, NumCandidates, ClusterSize)];
# Algorithm" (0 - SlidingDCT, 1 - NlmAverage, 2 - NlmWeightedAverage, 3 - BM3D, 4 - BM3D Wiener)
# "Format" (0 - RGBNoConvert, 1 - RGB, 2 - YUV420, 3 - Greyscale, 4 - LumaChroma)
# Arrays are all heigh x width
alg = 4
form = 1
blocksize = 8
tilesize = 15
numcandidates = 32
clustersize = 16
params = np.asfortranarray(np.array(
    [alg, form, blocksize, tilesize, numcandidates, clustersize], dtype=np.float32)[..., np.newaxis])

# Output
sigma_denoise = sigma_noise
verbose = 0

# Run
output = np.zeros_like(b)
tic()
Halide('prox_FANN.cpp').prox_FANN(b, sigma_denoise, params, verbose, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(np.clip(output, 0, 1), interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Denoised FANN result')
plt.show()

# Wait until done
raw_input("Press Enter to continue...")
