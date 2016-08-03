

# Proximal
import sys
sys.path.append('../../')


from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')


from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.prox_fns import *
from proximal.lin_ops import Variable

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena
import cv2

############################################################

# Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
# img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
np_img_color = np_img
np_img = np.mean(np_img_color, axis=2)
# print 'Type ', np_img.dtype , 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
# plt.show()
plt.savefig('prox0.png')

# Compile
# Halide('prox_L1.cpp', recompile=True) #Compile
# Halide('prox_IsoL1.cpp', recompile=True) #Compile
# Halide('prox_Poisson.cpp', recompile=True) #Compile


############################################################################
# Test the L1 prox
############################################################################
v = np_img
theta = 0.5

# Output
output = np.zeros_like(np_img)

tic()
hl = Halide('prox_L1.cpp', recompile=True, verbose=False, cleansource=True)  # Call
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.prox_L1(v, theta, output)
print('Running Halide (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.prox_L1(v, theta, output)
print('Running Halide (second) took: {0:.1f}ms'.format(toc()))


# Reference
#output_ref = np.maximum( 0.0, v - theta ) - np.maximum( 0.0, -v - theta )

 # No modifiers.
tmp = Variable(v.shape)
tic()
fn = norm1(tmp, implem='halide')  # group over all but first two dims
print('Prox Norm1 running took: {0:.1f}ms'.format(toc()))
output_ref = fn.prox(1.0 / theta, v.copy())

# Error
print('Maximum error L1 {0}'.format(np.amax(np.abs(output_ref - output))))

############################################################################
# Test the Iso L1 prox
############################################################################

# Compute gradient for fun
f = np_img
if len(np_img.shape) == 2:
    f = f[..., np.newaxis]

ss = f.shape
fx = f[:, np.r_[1:ss[1], ss[1] - 1], :] - f
fy = f[np.r_[1:ss[0], ss[0] - 1], :, :] - f
v = np.asfortranarray(np.stack((fx, fy), axis=-1))

# Output
output = np.zeros_like(v)

tic()
hl = Halide('prox_IsoL1.cpp', recompile=True, verbose=False, cleansource=True)  # Call
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.prox_IsoL1(v, theta, output)  # Call
print('Running Halide (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.prox_IsoL1(v, theta, output)  # Call
print('Running Halide (second) took: {0:.1f}ms'.format(toc()))


# Reference
normv = np.sqrt(np.multiply(v[:, :, :, 0], v[:, :, :, 0]) + \
                np.multiply(v[:, :, :, 1], v[:, :, :, 1]))
normv = np.stack((normv, normv), axis=-1)
with np.errstate(divide='ignore'):
    output_ref = np.maximum(0.0, 1.0 - theta / normv) * v

 # No modifiers.
tmp = Variable(v.shape)
fn = group_norm1(tmp, range(2, len(v.shape)), implem='halide')  # group over all but first two dims
rho = 1.0 / theta
output_ref = fn.prox(rho, v.copy())

# Error
print('Maximum error IsoL1 {0}'.format(np.amax(np.abs(output_ref - output))))

############################################################################
# Test Poisson prox
############################################################################
v = np_img
theta = 0.5

mask = np.asfortranarray(np.random.randn(*list(np_img.shape)).astype(np.float32))
mask = np.maximum(mask, 0.)
mask.fill(1.0)
b = np_img * np_img

# Output
output = np.zeros_like(v)

tic()
hl = Halide('prox_Poisson.cpp', recompile=True, verbose=False, cleansource=True)  # Call
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.prox_Poisson(v, mask, b, theta, output)  # Call
print('Running Halide (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.prox_Poisson(v, mask, b, theta, output)  # Call
print('Running Halide (second) took: {0:.1f}ms'.format(toc()))


# Reference
#output_ref = 0.5 * ( v - theta + np.sqrt( (v - theta)*(v - theta) + 4 * theta * b ) )
#output_ref[mask <= 0.5] = v[mask <= 0.5]

 # No modifiers.
tmp = Variable(v.shape)
fp = poisson_norm(tmp, b, implem='halide')  # group over all but first two dims
rho = 1.0 / theta
output_ref = fp.prox(rho, v.copy())

# Error
print('Maximum error Poisson {0}'.format(np.amax(np.abs(output_ref - output))))

############################################################################
# Test NLM
############################################################################

# #Compile
ext_libs = '-lopencv_core', '-lopencv_imgproc', '-lopencv_cudaarithm', '-lopencv_cudev', '-lopencv_photo', '-lm'
ext_srcs = ['external/external_NLM.cpp']
Halide('prox_NLM.cpp', external_source=ext_srcs, external_libs=ext_libs,
       recompile=True, verbose=False, cleansource=True)  # Compile

# Works currently on color image
v = np_img_color
sigma_fixed = 0.6
lambda_prior = 0.5
sigma_scale = 1.5 * 1
prior = 1.0
params = np.asfortranarray(
    np.array([sigma_fixed, lambda_prior, sigma_scale, prior], dtype=np.float32)[..., np.newaxis])
theta = 0.5

# #Output
output = np.zeros_like(v)

# #Run
tic()
Halide('prox_NLM.cpp').prox_NLM(v, theta, params, output)  # Call
print('Running took: {0:.1f}ms'.format(toc()))

plt.figure()
imgplot = plt.imshow(output, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Output Halide (CUDA EXTERN)')
# plt.show()
plt.savefig('prox1.png')

# No modifiers.
v = np_img_color
tmp = Variable(v.shape)
p = patch_NLM(tmp, sigma_fixed=sigma_fixed, sigma_scale=sigma_scale,
              templateWindowSizeNLM=3, searchWindowSizeNLM=11, gamma_trans=2.0,
              prior=prior, implem='halide')  # group over all but first two dims
rho = 1.0 / theta
dst = fp.prox(rho, v.copy())

plt.figure()
imgplot = plt.imshow(dst, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('NLM denoised CV2')
# plt.show()
plt.savefig('prox2.png')

# Error
print('Maximum error NLM (CUDA vs. CPU) {0}'.format(np.amax(np.abs(output - dst))))

# Wait until done
raw_input("Press Enter to continue...")
