This is a Python framework for matrix-free proximal algorithms.
Currently ADMM, linearized ADMM, Chambolle-Pock, and half-quadratic splitting are implemented.

You form a problem by first declaring a variable.

```python
from proximal import *
X = Variable(10)  # 1D variable in R^10
X = Variable((20, 10))  # 2D variable in R^{20x10}
X = Variable((10, 20, 10))  # 3D variable in R^{10x20x10}
```
Variables can have any number of dimensions. 

Linear operators are created by repeatedly applying a set of library functions.
For example,
```python
X = Variable((20, 10))
lin_op = conv(kernel, subsample(X, (2, 2)))
```
``lin_op`` is a linear operator representing subsampling every other pixel along both axes of X, followed by circular convolution with the given kernel.

To define a new linear operator, simply copy the style of [``proximal/lin_ops/conv.py``](lin_ops/conv.py).
You must define an ``forward(x, y)`` function that applies the forward operator to ``x`` and writes the result to ``y``.
Similarly, you must define an ``adjoint(u, v)`` function that applies the forward operator to ``u`` and writes the result to ``v``.

To define the problem, create a list of proximal operators.
Proximal operators take linear operators as arguments, along with optional arguments such as an offset and scaling.
See ``proximal/prox_fns/prox_fn.py`` for details.
Here we define the operator ``||2*conv(K, X) - B||_1``
```
norm1(2*conv(K, X) - B)
```
Here ``B`` is a numpy  ndarray of the same dimensions as ``conv(K, X)``.


To define more proximal operators, simply copy the style of ``proximal/lin_ops/sum_squares.py``.
You must define a ``_prox(rho, v)`` function that applies the proximal operator.
See [``proximal/prox_fns/prox_fn.py``](prox_fns/prox_fn.py) for details.
