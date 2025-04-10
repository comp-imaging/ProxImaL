import numpy as np
from numpy import ndarray
from scipy.sparse.linalg import LinearOperator, eigs

from proximal.experimental.emulate import generateImpl
from proximal.experimental.ir.problem import Problem
from proximal.experimental.models import LinOp, LinOpImpl, ProxFn


def generateLinOpImpl(lin_ops: list[LinOp]) -> tuple[LinOpImpl, LinOpImpl]:
    if len(lin_ops) == 0:

        def passthru(x: ndarray) -> ndarray:
            return x

        return passthru, passthru

    for lin_op in lin_ops:
        assert lin_op.input_dims is not None
        assert lin_op.output_dims is not None

    def forward_op(u: ndarray) -> ndarray:
        v = u
        for lin_op in lin_ops:
            assert np.all(np.array(v.shape) == lin_op.input_dims), f"{v.shape} != {lin_op.input_dims}"

            # Retrieve the Python implementation of the function
            forward_func = generateImpl(lin_op)[0]

            # Apply forward linear operator
            v = forward_func(v)
            assert np.all(np.array(v.shape) == lin_op.output_dims), f"{v.shape} != {lin_op.output_dims}"

        return v

    def adjoint_op(v: ndarray) -> ndarray:
        u = v
        for lin_op in reversed(lin_ops):
            assert np.all(np.array(u.shape) == lin_op.output_dims)

            # Retrieve the Python implementation of the function
            adjoint_func = generateImpl(lin_op)[1]

            # Apply forward linear operator
            u = adjoint_func(u)
            assert np.all(np.array(u.shape) == lin_op.input_dims)

        return u

    return forward_op, adjoint_op


def estimateCompGraphNorm(psi_fns: list[ProxFn], dims: tuple[int], tol=1e-3) -> float:
    assert len(psi_fns) > 0

    if np.all([len(fn.lin_ops) for fn in psi_fns] == 0):
        # Pass through function. Return norm = 1.0
        return 1.0

    def K(u: ndarray) -> list[ndarray]:
        v: list[ndarray] = []
        for cost_fn in psi_fns:
            # For each penality term in set Psi, retrieve the linear operator.
            # Generate the implementaiton.
            forward_op, _ = generateLinOpImpl(cost_fn.lin_ops)
            v.append(forward_op(u))

        return v

    def Kt(v: list[ndarray]) -> ndarray:
        assert len(v) == len(psi_fns)

        u = np.zeros(dims, order="F", dtype=np.float32)
        for vi, cost_fn in zip(v, psi_fns):
            _, adjoint_op = generateLinOpImpl(cost_fn.lin_ops)
            u += adjoint_op(vi)

        return u

    def KtK(x: ndarray) -> ndarray:
        u = np.asfortranarray(x).reshape(dims)
        v = K(u)
        u_new = Kt(v)
        return u_new.ravel()

    # Define linear operator
    n_elem = np.prod(dims)
    A = LinearOperator((n_elem, n_elem), KtK, KtK)

    Knorm = np.sqrt(eigs(A, k=1, M=None, sigma=None, which="LM", tol=tol)[0].real)
    return float(Knorm)


def scale(problem: Problem) -> Problem:
    problem.propagateBounds()
    problem.Knorm = estimateCompGraphNorm(problem.psi_fns, problem.u.shape)
    return problem
