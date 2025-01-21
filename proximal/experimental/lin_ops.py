# from proximal.experimental.types import LinOp
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class MultiplyAdd:
    scale: float
    offset: float

    input_dims: list[int]
    output_dims: list[int]

    def __init__(self, input_dims: list[int]) -> None:
        self.input_dims = input_dims
        self.output_dims = input_dims


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
