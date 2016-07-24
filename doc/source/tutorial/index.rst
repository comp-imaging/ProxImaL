.. _tutorial:

Tutorial
========

In this tutorial we cover the linear operators, proxable functions, and solver algorithms available.

Linear Operators
----------------
* ``Variable(shape)``: Creates a variable object.
* ``conv(kernel, arg)``: Convolves ``arg`` with ``kernel``.
* ``subsample(arg, steps)``: Extracts every ``steps[i]`` pixel along axis ``i``, starting with pixel ``steps[i]-1``.
* ``mul_elemwise(weight, arg)``: Element-wise multiplication with a fixed constant weight array.
* ``scale(rho, arg)``: Scale ``arg`` by scalar constant ``rho``.
* ``sum([arg1, arg2, ...])``: Sums list of input expressions into a single linear expression.
* ``vstack([e1, e2, ...])``: Vectorizes and stacks a list of input expressions into a single linear expression.
* ``grad(arg, dims)``: Computes the gradients of ``arg`` across the specified ``dims``, by default across all of its dimensions.
* ``warp(arg, H)``: Interprets ``arg`` as a 2D image and warps it using the homography ``H`` with linear interpolation.
* ``mul_color(arg, C)``: Performs a blockwise 3x3 color transform using the color matrix ``C``, or the predefined opponent (``C="opp"``) and YUV (``C="yuv"``) color spaces.
* ``resize(arg, shape)``: Casts ``arg`` to the given ``shape``.

Proxable Functions
------------------
* ``sum_squares(lin_op)``: The squared L2 norm.
* ``norm1(lin_op)``: The L1 norm.
* ``group_norm1(lin_op, dims)``: The sum of the L2 norm taken across the given dims.
* ``poisson_norm(lin_op, b)``: The maximum-likelihood denoiser for Poisson noise with observations b.
* ``patch_NLM(lin_op)``: A prior based on the nonlocal means denoising algorithm.
* ``nonneg(lin_op)``: A nonnegativity prior.
* ``sum_entries(lin_op)``: Sums over the linear operator. 
* ``diff_fn(lin_op, func, fprime, bounds=None)``: A generic differentiable function with evaluation oracle ``func``, gradient oracle ``fprime``, and box constraints given by ``bounds``.

Solver Algorithms
-----------------
Specify the solver algorithm via ``prob.solve(solver=algorithm key)``.
The algorithm keys are given below.

* ``'pc'``: The Pock-Chambolle algorithm.
* ``'admm'``: The alternating directions method of multipliers (ADMM) algorithm.
* ``'ladmm'``: The linearized ADMM algorithm.
* ``'hqs'``: The half-quadratic splitting algorithm.

All algorithms support the solve method arguments

* ``max_iters`` to set the maximum number of iterations.
* ``eps_abs`` and ``eps_rel`` to set the desired precision.
* ``verbose`` to set whether or not to print debugging information.
* ``x0`` to set the initial value of the variable in the algorithm.
