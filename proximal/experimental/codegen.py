import numpy as np
from numpy import ndarray

from proximal.experimental.lin_ops import Grad, MultiplyAdd
from proximal.experimental.models import LinOp, LinOpImpl
from proximal.halide.halide import Halide


def GradForward(input: ndarray) -> ndarray:
    dims = input.shape
    input = input.reshape((*dims, 1))
    output = np.empty((*dims, 1, 2), order="F", dtype=np.float32)

    Halide("A_grad").A_grad(input, output)
    return output.reshape((*dims, 2))


def GradAdjoint(input: ndarray) -> ndarray:
    dims = input.shape[:-1]

    input = input.reshape((*dims, 1, 2))
    output = np.empty((*dims, 1), order="F", dtype=np.float32)

    Halide("At_grad").At_grad(input, output)
    return output.reshape(dims)


def generateImpl(lin_op: LinOp) -> tuple[LinOpImpl, LinOpImpl]:
    if isinstance(lin_op, Grad):
        return GradForward, GradAdjoint

    if isinstance(lin_op, MultiplyAdd):

        def multiply_add(x: ndarray) -> ndarray:
            return lin_op.scale * x + lin_op.offset

        return multiply_add, multiply_add

    raise IndexError(f"Linear operator {type(lin_op).__name__} not implemented")
