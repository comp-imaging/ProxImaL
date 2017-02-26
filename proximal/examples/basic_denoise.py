#from proximal import *
import proximal
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
print(dir(proximal))
# Generate data.
I = scipy.misc.ascent()
np.random.seed(1)
n = 10 * np.random.randn(*I.shape)
b = I + n

# Construct problem.
x = proximal.Variable(I.shape)
prob = proximal.Problem(proximal.sum_squares(x - b / 255) + .1 * proximal.norm1(proximal.grad(x)) + proximal.nonneg(x))

# Solve problem.
result = prob.solve(verbose=True)
print(result)

plt.figure(figsize=(15, 8))
plt.subplot(231)
plt.gray()
plt.imshow(I)
plt.title('Original image')

plt.subplot(232)
plt.gray()
plt.imshow(b)
plt.title('Noisy image')

plt.subplot(233)
plt.gray()
plt.imshow(x.value * 255)
plt.title('Denoising results')

plt.subplot(235)
plt.gray()
plt.imshow(x.value * 255 - I)
plt.colorbar()
plt.title('Denoising diff to original')

plt.subplot(236)
plt.gray()
plt.imshow(x.value * 255 - b)
plt.colorbar()
plt.title('Denoising diff to noisy')

plt.subplot(234)
plt.gray()
plt.imshow(n)
plt.colorbar()
plt.title('Noise')


plt.show()
