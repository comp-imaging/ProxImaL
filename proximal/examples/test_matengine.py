# Proximal
import sys
sys.path.append('../../')

import cvxpy as cvx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import matlab.engine
import StringIO

############################################################

# Start the engine
eng = matlab.engine.start_matlab()

# Add test path
eng.addpath(r'../../apps/poisson', nargout=0)
eng.addpath(r'../../apps/poisson/images', nargout=0)
eng.addpath(r'../../apps/poisson/kernels', nargout=0)
eng.addpath(r'../../apps/poisson/hyperlaplacian_code', nargout=0)

# Call script function
# eng.deblurring_launch_single_poisson(nargout=0)
#result = np.array( eng.workspace['I_deconv'] )
# print result.shape, ' ', result.dtype

result = np.array(eng.deblurring_launch_single_poisson_test())
print result.shape, ' ', result.dtype

# Show result
plt.ion()
plt.figure()
imgplot = plt.imshow(result, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Original Image')
plt.show()

# Stop enging
eng.quit()

# Wait until done
raw_input("Press Enter to continue...")
