This is a Python framework for matrix-free proximal algorithms.
Currently only ADMM is implemented.

You form a problem by first declaring a variable.

```python
from proximal import *
X = Variable(10)  # 1D variable in R^10
X = Variable((20, 10))  # 2D variable in R^{20x10}
X = Variable((10, 20, 10))  # 3D variable in R^{10x20x10}
```
Variables can have any number of dimensions. I've only tested 1D and 2D variables though.
Each problem can only have 1 variable right now.
Allowing multiple variables makes everything more complicated.

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

I tried to implement these operations so they don't allocate new memory.
This is important for making the framework efficient.
Efficiency of course is not our immediate concern, but I wanted to show what efficient Python code looks like.

There isn't any operator overloading with the linear operators.
Right now the framework assumes that all linear operators are just pipelines (i.e., A_1*A_2*...*A_n*X).
I can make it more flexible later and allow sums and other more complicated expression trees,
but that will be much more involved.

To define the problem, create a list of proximal operators.
Proximal operators take linear operators as arguments, along with optional arguments such as an offset and scaling.
See [``proximal/prox_fns/prox_fn.py``](prox_fns/prox_fn.py) for details.
Here we define the operator ``||2*conv(K, X) - B||_1``
```python
norm1(conv(K, X), beta=2, b=B)
```
Here ``B`` is a numpy  ndarray of the same dimensions as ``conv(K, X)``.


To define more proximal operators, simply copy the style of [``proximal/prox_fns/sum_squares.py``](prox_fns/sum_squares.py).
You must define a ``_prox(rho, v)`` function that applies the proximal operator and ``_eval(v)``  which evaluates the functional.
As with the linear operators, I tried to allocate memory in advance so that applying the proximal operators is more efficient.

The ADMM algorithm can be called using the syntax
```python
admm([prox_fn1, prox_fn2], rho, max_iters)
```
It doesn't check convergence currently. It just runs for max_iters.
The ADMM algorithm is defined in [``proximal/algorithms/admm.py``](algorithms/admm.py).
Unfortunately it's a little complicated.
The main issue is that the LSQR black box wants vector inputs and outputs from the ``matvec`` and ``rmatvec`` functions,
but the linear operators may output ND arrays.
