import numpy as np
from numpy import ndarray

from proximal.experimental.ir.lin_ops import Crop, FFTConv, Grad, MultiplyAdd
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


def CropForward(input: ndarray, crop_args: Crop) -> ndarray:
    return input[crop_args.top : crop_args.top + crop_args.height, crop_args.left : crop_args.left + crop_args.width]


def CropAdjoint(input: ndarray, crop_args: Crop) -> ndarray:
    dims = crop_args.input_dims
    output = np.zeros(dims, order="F", dtype=np.float32)
    output[crop_args.top : crop_args.top + crop_args.height, crop_args.left : crop_args.left + crop_args.width] = input
    return output


def ConvForward(input: ndarray, conv_args: FFTConv) -> ndarray:
    dims = input.shape
    input = input.reshape((*dims, 1))
    output = np.empty((*dims, 1), order="F", dtype=np.float32)

    kernel = conv_args.kernel
    assert kernel.dtype == np.float32
    assert np.isfortran(kernel)

    Halide("A_conv").A_conv(input, kernel, output)
    return output.reshape(dims)


def ConvAdjoint(input: ndarray, conv_args: FFTConv) -> ndarray:
    dims = input.shape
    input = input.reshape((*dims, 1))
    output = np.empty((*dims, 1), order="F", dtype=np.float32)

    kernel = conv_args.kernel
    assert kernel.dtype == np.float32
    assert np.isfortran(kernel)

    Halide("At_conv").At_conv(input, kernel, output)
    return output.reshape(dims)


def generateImpl(lin_op: LinOp) -> tuple[LinOpImpl, LinOpImpl]:
    if isinstance(lin_op, Grad):
        return GradForward, GradAdjoint

    if isinstance(lin_op, Crop):
        return lambda x: CropForward(x, lin_op), lambda x: CropAdjoint(x, lin_op)

    if isinstance(lin_op, FFTConv):
        return lambda x: ConvForward(x, lin_op), lambda x: ConvAdjoint(x, lin_op)

    if isinstance(lin_op, MultiplyAdd):

        def multiply_add(x: ndarray) -> ndarray:
            return lin_op.scale * x + lin_op.offset

        return multiply_add, multiply_add

    raise IndexError(f"Linear operator {type(lin_op).__name__} not implemented")
