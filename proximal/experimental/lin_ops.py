# from proximal.experimental.types import LinOp
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class MultiplyAdd:
    scale: float
    offset: float

    input_dims: list[int]
    output_dims: list[int]

    latex_notation: str = "M"


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

    latex_notation: str = "F^T D F"


@dataclass
class Crop:
    left: int
    top: int
    width: int
    height: int

    input_dims: list[int]
    output_dims: list[int]

    # Use dataclass?
