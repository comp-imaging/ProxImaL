'''
    Compute the Horn-Schunck optical flow algorithm.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../..')

import proximal as px

Y, X = np.mgrid[-64:64, -64:64]
R_squared = 35.0**2

################################################################################
# Generate images


def draw_inverted_cone(X, Y, R_squared, offset=0, shift_edge=False):
    image = np.sqrt((X + offset)**2 + Y**2)
    if shift_edge:
        image[(X + offset)**2 + Y**2 >= R_squared] = 0.0
    else:
        image[X**2 + Y**2 >= R_squared] = 0.0

    return np.asfortranarray(image, dtype=np.float32)


# Draw a cone at the center
image1 = draw_inverted_cone(X, Y, R_squared)

# Second image is offset by 0.5 pixel...
image2 = draw_inverted_cone(
    X,
    Y,
    R_squared,
    offset=0.5,
    shift_edge=False,
)

# Add noise to both images
shape = image1.shape
noise_level = image1.max() * 1e-2
image1 += np.random.rand(*shape) * noise_level
image2 += np.random.rand(*shape) * noise_level

################################################################################
# Set up optimization problem

# Compute spatial and temporal derivatives
fy, fx = np.gradient(image1)
ft = image2 - image1

u = px.Variable(shape)
v = px.Variable(shape)

alpha = 2**-1

# Norm-1 regularized
funcs = [
    alpha * px.sum_squares(px.grad(u, implem='halide'), implem='halide'),
    alpha * px.sum_squares(px.grad(v, implem='halide'), implem='halide'),
    px.sum_squares(px.mul_elemwise(fx, u, implem='halide') +
                   px.mul_elemwise(fy, v, implem='halide') + ft,
                   implem='halide'),
]

# Solve the problem using ProxImaL
prob = px.Problem(funcs)
prob.solve(verbose=True, solver='ladmm')

################################################################################
# Plot results

plt.figure(1)
plt.subplot(221)
plt.imshow(image1, cmap='gray')
plt.title('image1')
plt.axis('off')

plt.subplot(222)
plt.imshow(u.value, cmap='gray')
plt.axis('off')
plt.title('u')

plt.subplot(223)
plt.imshow(v.value, cmap='gray')
plt.title('v')

plt.subplot(224)
plt.quiver(u.value[::4, ::4], v.value[::4, ::4])
plt.gca().invert_yaxis()
plt.title('(u, v)')
plt.axis('equal')
plt.axis('off')

plt.show()
