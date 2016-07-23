# Deconvolution example.
import numpy as np
from scipy import ndimage

from proximal import *
from proximal.utils.utils import *
from proximal.utils.metrics import *

import matplotlib.pyplot as plt
from PIL import Image
import cv2

############################################################

#Load image
img = Image.open('./data/angela.jpg')  # opens the file using Pillow - it's not an array yet
I = np.asfortranarray( im2nparray(img) )
I = np.maximum( cv2.resize(I,(512,512), interpolation=cv2.INTER_LINEAR), 0)
I = np.mean( I, axis=2)
I = np.asfortranarray( I )
I = np.maximum(I, 0.0)

#Generate observation
sigma_noise = 0.1
b = I + sigma_noise * np.random.randn(*I.shape)

x = Variable( I.shape )
funcs = [7.5*sum_squares(x - b),
         norm1( grad(x))]
prob = Problem(funcs, implem=Impl['numpy'])

tic()
prob.solve(verbose=True, solver='pc', eps_abs=1e-3, eps_rel=1e-3)
print( 'Overall solver took: {0:.1f}ms'.format( toc() ) )

plt.figure()
imgplot = plt.imshow(I, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Original image')

plt.figure()
imgplot = plt.imshow(b, interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Noisy image')

plt.figure()
imgplot = plt.imshow(x.value , interpolation="nearest", clim=(0.0, 1.0))
imgplot.set_cmap('gray')
plt.colorbar()
plt.title('Denoising results')
plt.show()
