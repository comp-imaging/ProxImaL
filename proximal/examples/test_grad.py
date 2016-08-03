

# Proximal
import sys
sys.path.append('../../')

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *

import numpy as np
from scipy import signal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import lena

############################################################

#img = lena().astype(np.uint8)
#img = img.copy().reshape(img.shape[0],img.shape[1],1)

# Load image
# img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
img = Image.open('./data/largeimage.png')  # opens the file using Pillow - it's not an array yet

np_img = np.asfortranarray(im2nparray(img))
np_img = np.mean(np_img, axis=2)
print 'Img Type ', np_img.dtype, 'Shape', np_img.shape

plt.ion()
plt.figure()
imgplot = plt.imshow(np_img, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.title('Numpy')
plt.show()

# tic()
# Halide('A_grad.cpp', recompile=True, verbose=False) #Force recompile
# Halide('At_grad.cpp', recompile=True, verbose=False) #Force recompile in local dir
#print( 'Compilation took: {0:.1f}ms'.format( toc() ) )

# Test the runner
output = np.zeros((np_img.shape[0], np_img.shape[1], np_img.shape[2] if (
    len(np_img.shape) > 2) else 1, 2), dtype=np.float32, order='FORTRAN')
print 'Out Type ', output.dtype, 'Shape', output.shape

tic()
hl = Halide('A_grad.cpp', recompile=True, verbose=False, cleansource=True)  # Force recompile
print('Compilation took: {0:.1f}ms'.format(toc()))

tic()
hl.A_grad(np_img, output)  # Call
print('Running halide (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.A_grad(np_img, output)  # Call
print('Running halide (second) took: {0:.1f}ms'.format(toc()))


# Compute comparison
f = np_img
if len(np_img.shape) == 2:
    f = f[..., np.newaxis]

tic()

# #Build up index for shifted array
# ss = f.shape;
# stack_arr = ()
# for j in [0,1]:

# 	#Add grad for this dimension (same as index)
# 	il = ()
# 	for i in range(len(ss)):
# 		if i == j:
# 			il += np.index_exp[ np.r_[1:ss[j],ss[j] - 1] ]
# 		else:
# 			il += np.index_exp[:]

# 	fgrad_j = f[ il ] - f;
# 	stack_arr += (fgrad_j,)

# #Stack all grads
# Kf = np.asfortranarray( np.stack( stack_arr, axis=-1 ) )

 # Forward.
var = Variable(f.shape)
fn = grad(var, dims=2)  # 2d Gradient
Kf = np.zeros(fn.shape, dtype=np.float32, order='F')
fn.forward([f], [Kf])

print('Running nd-grad took: {0:.1f}ms'.format(toc()))
Kfh = np.asfortranarray(Kf[:, :, :, :])  # For halide

# 2D code
# tic()
# ss = f.shape;
# fx = f[ :, np.r_[1:ss[1],ss[1] - 1],: ] - f;
# fy = f[ np.r_[1:ss[0],ss[0] - 1],:,:] - f;
# Kf = np.asfortranarray( np.stack( (fx, fy), axis=-1 ) )
# print( 'Running grad 1 took: {0:.1f}ms'.format( toc() ) )
# Kfh = Kf

# Error
print('Maximum error {0}'.format(np.amax(np.abs(Kfh - output))))

tic()
hl = Halide('At_grad.cpp', recompile=True, verbose=False, cleansource=True)  # Force recompile
print('Compilation took: {0:.1f}ms'.format(toc()))

output_t = np.zeros(f.shape, dtype=np.float32, order='F')
tic()
hl.At_grad(Kfh, output_t)  # Call
print('Running trans halid (first) took: {0:.1f}ms'.format(toc()))

tic()
hl.At_grad(Kfh, output_t)  # Call
print('Running trans halid (second) took: {0:.1f}ms'.format(toc()))


# Compute comparison (Negative divergence)
f = Kf

tic()

# KTf = np.zeros(f.shape[0:-1], dtype=np.float32)
# for j in [0,1]:

# 	#Get component
# 	fj = f[...,j]
# 	ss = fj.shape

# 	#Add grad for this dimension (same as index)
# 	istart = ()
# 	ic = ()
# 	iend_out = ()
# 	iend_in = ()
# 	for i in range(len(ss)):
# 		if i == j:
# 			istart += np.index_exp[0]
# 			ic += np.index_exp[ np.r_[0,0:ss[j]-1] ]
# 			iend_out += np.index_exp[-1]
# 			iend_in += np.index_exp[-2]
# 		else:
# 			istart += np.index_exp[:]
# 			ic += np.index_exp[:]
# 			iend_in += np.index_exp[:]
# 			iend_out += np.index_exp[:]

# 	#Do the grad operation for dimension j
# 	fd = fj - fj[ic]
# 	fd[istart] = fj[istart]
# 	fd[iend_out] = -fj[iend_in]

# 	KTf += (-fd)

# Adjoint.
KTf = np.zeros(var.shape, dtype=np.float32, order='F')
fn.adjoint([f], [KTf])

print('Running trans nd-grad took: {0:.1f}ms'.format(toc()))

# #2D reference
# Kfx = Kf[:,:,:,0]
# ss = Kfx.shape
# fx = Kfx - Kfx[ :, np.r_[0,0:ss[1]-1],:]
# fx[:,0,:] = Kfx[:,0,:];
# fx[:,-1,:] = -Kfx[:,-2,:];

# Kfy = Kf[:,:,:,1]
# fy = Kfy - Kfy[ np.r_[0,0:ss[0]-1],:,:];
# fy[0,:,:] = Kfy[0,:,:];
# fy[-1,:,:] = -Kfy[-2,:,:];
# KtKf = -fx -fy

print('Maximum trans error {0}'.format(np.amax(np.abs(KTf - output_t))))
