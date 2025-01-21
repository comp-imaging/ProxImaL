# from proximal.experimental.types import LinOp
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class MultiplyAdd:
    scale: float
    offset: float

    input_dims: list[int]
    output_dims: list[int]

    latex_notation: str = "\\mathbf{M}"


@dataclass
class Grad:
    input_dims: list[int]
    output_dims: list[int]

    latex_notation: str = "\\nabla"


@dataclass
class FFTConv:
    kernel: ndarray

    input_dims: list[int]
    output_dims: list[int]

    latex_notation: str = "\\mathcal{F}^T \\mathbf{D} \\mathcal{F}"


@dataclass
class Crop:
    left: int
    top: int
    width: int
    height: int

    input_dims: list[int]
    output_dims: list[int]

    latex_notation: str = "\\mathbf{M}"
