import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

from proximal import *
from proximal.utils.utils import Impl

# Generate data.
I = scipy.misc.ascent()[:128, :128]
np.random.seed(1)
b = I + 10 * np.random.randn(*I.shape)

H = np.array(
    [
        [-1., 0, I.shape[1]],
        [0, 1., 0],
        [0., 0., 1.],
    ],
    dtype=np.float32,
    order='F',
)

# Construct problem.
impl = Impl['halide']
x = Variable(I.shape)
prob = Problem(sum_squares(
    conv(b / 255., x) - conv(b / 255., x)) +
               .1 * sum_squares(x) + nonneg(x, implem=impl),
               implem=impl)

# Solve problem.
result = prob.solve(verbose=True, solver='ladmm')
print('Optimized cost function value = {}'.format(result))

plt.figure(figsize=(15, 8))
plt.subplot(131)
plt.gray()
plt.imshow(I)
plt.title('Original image')

plt.subplot(132)
plt.gray()
plt.imshow(b, clim=(0, 255))
plt.title('Noisy image')

plt.subplot(133)
plt.gray()
plt.imshow(x.value * 255, clim=(0, 255))
plt.title('Denoising results')
plt.show()
