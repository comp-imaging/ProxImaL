""" 
total generalized variation zooming to demonstrate the cuda implementation

You have to install pycuda and the cuda toolchain to be able to use the pycuda
implementation.

If you get errors during cuda kernel compilation, you might have to set the 
environment variable PYCUDA_DEFAULT_NVCC_FLAGS to an appropriate value.
"""

from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *
from proximal.utils import *

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

import time 

def solve(subsampled, alpha, f, K1, K2, implem):
    x = Variable([s*f for s in subsampled.shape])
    x.initval = np.zeros(x.shape)
    for i1 in range(f):
        for i2 in range(f):
            x.initval[i1::f,i2::f] = subsampled
    
    w = Variable(x.shape + (2,))
    w.initval = np.zeros(w.shape)
    w.initval[:-1,:,0] = x.initval[1:,:] - x.initval[:-1,:]
    w.initval[:,:-1,1] = x.initval[:,1:] - x.initval[:,:-1]
    
    e_data = sum_squares(subsample(conv_nofft(K1, conv_nofft(K2, x)), [f,f]) - subsampled)
    gw = grad(w, 2)
    Ew = gw + transpose(gw, (0,1,3,2))
    
    e_reg = alpha[1] * group_norm1(grad(x) - w, [2]) + alpha[0] * group_norm1(Ew, [2,3])
    
    f = e_data + e_reg
    
    prob = Problem(f, implem=Impl[implem])
    t0 = time.time()
    prob.solve(verbose = 1, eps_abs=1e-4, eps_rel=1e-4)
    t1 = time.time()
    
    return t1-t0, x.value    

if __name__ == "__main__":
    ############################################################
    
    # Load image
    I = ndimage.imread('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
    I = np.mean(I.astype(np.float32)/np.amax(I), 2)
    print(np.amin(I), np.amax(I))
    
    # Convolve and subsample
    def binom(n):
        if n == 2:
            return np.array([1, 1], np.float32)*0.5
        b = binom(n-1)
        r = np.zeros(n)
        r[:-1] = b
        r[1:] += b
        return r*0.5
    
    K1 = np.array([binom(15)])
    K2 = K1.T
    
    f = 10
    subsampled = ndimage.filters.convolve(ndimage.filters.convolve(I, K1), K2)[::f,::f]
    
    plt.ion()
    plt.figure()
    imgplot = plt.imshow(I, interpolation="nearest", clim=(0.0, 1.0))
    imgplot.set_cmap('gray')
    plt.title('Original Image')
    plt.show()
    
    plt.figure()
    plt.imshow(subsampled, interpolation="nearest", clim=(0.0,1.0)).set_cmap('gray')
    plt.title('Subsampled Image')
    plt.show()
    
    #opfun = lambda x: np.sum(np.ravel((solve(subsampled, x, f, K1, K2, 'pycuda')[1][:I.shape[0],:I.shape[1]] - I)**2))
    #import scipy.optimize
    #alpha = scipy.optimize.fmin(opfun, [0.00388916,  0.03439691], maxfun=50)
    
    alpha = [0.00388916,  0.03439691]
    tcuda, Icuda = solve(subsampled, alpha, f, K1, K2, 'pycuda')
    plt.figure()
    plt.imshow(Icuda, interpolation="nearest", clim=(0.0,1.0)).set_cmap('gray')
    plt.title('Reconstructed Image (cuda) %.1f s' % tcuda)
    plt.show()

    tnumpy, Inumpy = solve(subsampled, alpha, f, K1, K2, 'numpy')   
    
    plt.figure()
    plt.imshow(Inumpy, interpolation="nearest", clim=(0.0,1.0)).set_cmap('gray')
    plt.title('Reconstructed Image (numpy) %.1f s' % tnumpy)
    plt.show()
    
    