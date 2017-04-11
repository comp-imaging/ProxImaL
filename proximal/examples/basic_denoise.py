from proximal import *
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

# Generate data.
I = scipy.misc.ascent()
np.random.seed(1)
b = I + 10 * np.random.randn(*I.shape)

# Construct problem.
x = Variable(I.shape)
prob = Problem(sum_squares(x - b / 255) + .1 * norm1(grad(x)) + nonneg(x))

# Solve problem.
result = prob.solve(verbose=True)
print(result)

plt.figure(figsize=(15, 8))
plt.subplot(131)
plt.gray()
plt.imshow(I)
plt.title('Original image')

plt.subplot(132)
plt.gray()
plt.imshow(b)
plt.title('Noisy image')

plt.subplot(133)
plt.gray()
plt.imshow(x.value * 255)
plt.title('Denoising results')
plt.show()
