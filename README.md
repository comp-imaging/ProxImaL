ProxImaL
=====================
[![Build Status](https://travis-ci.org/cvxgrp/cvxpy.png?branch=master)](https://travis-ci.org/comp-imaging/ProxImaL)

ProxImaL is a Python-embedded modeling language for image optimization problems. 
It allows you to express your problem in a natural way that follows the math, 
and automatically determines an efficient method for solving the problem.
ProxImaL makes it easy to experiment with many different priors and other problem reformulations,
without worrying about the details of how the problem is solved.

For example, the following code denoises an image using simple sparse gradient and nonnegativity priors:

```python
from proximal import *
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

# Generate data.
I = scipy.misc.ascent()
np.random.seed(1)
b = I + 10*np.random.randn(*I.shape)

# Construct and solve problem.
x = Variable(I.shape)
prob = Problem(sum_squares(x - b/255) + .1 * norm1(grad(x)) + nonneg(x))
prob.solve()

# Plot the original, noisy, and denoised images.
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
plt.imshow(x.value * 255)  # x.value is the optimal value of x.
plt.title('Denoising results')
plt.show()
```
![Optimization results](https://gist.githubusercontent.com/SteveDiamond/592094bdbd7d9d3f8606383d84db3de5/raw/47ef609f995ee92ab7d9af1d4ad47c60a9764b65/results.png)

The example above uses simple, well-known priors. Much better results can be obtained using the more sophisticated priors provided in ProxImaL.

ProxImaL was designed and implemented by Felix Heide and Steven Diamond.
See [the accompanying paper](http://web.stanford.edu/~stevend2/pdf/proximal.pdf) for a full discussion of ProxImaL's design and examples of state-of-the-art results obtained with ProxImaL.

A tutorial and other documentation can be found at [proximal-lang.org](http://www.proximal-lang.org/).

This git repository holds the latest development version of ProxImaL. For installation instructions, 
see the [install guide](http://www.proximal-lang.org/en/latest/install/index.html) at [proximal-lang.org](http://www.proximal-lang.org/).
