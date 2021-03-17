'''
    Compute the Horn-Schunck optical flow algorithm.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../..')

import proximal as px

################################################################################
# Generate images


def draw_inverted_cone(X, Y, R_squared, offset=0, shift_edge=False):
    image = np.sqrt((X + offset)**2 + Y**2)
    if shift_edge:
        image[(X + offset)**2 + Y**2 >= R_squared] = 0.0
    else:
        image[X**2 + Y**2 >= R_squared] = 0.0

    return np.asfortranarray(image, dtype=np.float32)


Y, X = np.mgrid[-64:64, -64:64]
R_squared = 35.0**2

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

alpha = 0.5

# Formulate the optical flow problem
prob = px.Problem([
    alpha * px.sum_squares(px.grad(u)),
    alpha * px.sum_squares(px.grad(v)),
    px.sum_squares(px.mul_elemwise(fx, u) + px.mul_elemwise(fy, v) + ft,),
])

# Solve the problem using ProxImaL
prob.solve(verbose=True, solver='ladmm', eps_abs=1e-2)

################################################################################
# Plot results

plt.figure(1)

plt.subplot(321)
plt.imshow(image1, cmap='gray')
plt.title('Before')
plt.axis('off')

plt.subplot(322)
plt.imshow(image2, cmap='gray')
plt.title('After')
plt.axis('off')


def draw_bipolar_cmap(field, i: int, title: str):
    peak_value = np.linalg.norm(field.ravel(), np.inf)

    plt.subplot(3, 2, i)
    plt.imshow(field, cmap='seismic', vmin=-peak_value, vmax=peak_value)
    plt.axis('off')
    plt.title(title)


draw_bipolar_cmap(u.value, 3, 'u: horizontal component')
draw_bipolar_cmap(v.value, 4, 'v: vertical component')

plt.subplot(325)
plt.quiver(u.value[::4, ::4], v.value[::4, ::4])
plt.gca().invert_yaxis()
plt.title('(u, v): optical flow')
plt.axis('equal')
plt.axis('off')

plt.show()
