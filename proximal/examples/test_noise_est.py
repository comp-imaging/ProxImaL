# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.lin_ops import *
from proximal.prox_fns import *

import cvxpy as cvx
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import matlab.engine
import StringIO

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
I = np.asfortranarray(im2nparray(img))
I = np.maximum(cv2.resize(I, (512, 512), interpolation=cv2.INTER_LINEAR), 0)
I = np.mean(I, axis=2)
I = np.asfortranarray(I)
I = np.maximum(I, 0.0)

# Generate observation
sigma_noise = 0.01
b = I + sigma_noise * np.random.randn(*I.shape)

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

# #Add compare with matlab version
# eng = matlab.engine.start_matlab()
# eng.addpath(r'../../apps/ihdr/code',nargout=0)
# vmat = matlab.double(b.tolist())
# method = matlab.double([2])
# result = np.array( eng.function_stdEst2D(vmat, method) )
# eng.quit()
# print 'Matlab Estimate:', result

# Estimate the noise
tic()
ndev = estimate_std(b, 'daub_replicate')
print('Estimation took: {0:.1f}ms'.format(toc()))

# Result
print('Noise estimate is: {0:1.4f}, Original was {1:1.4f}'.format(np.mean(ndev), sigma_noise))

# Wait until done
raw_input("Press Enter to continue...")
