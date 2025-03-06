# from proximal.experimental.types import LinOp
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class MultiplyAdd:
    scale: float
    offset: float | ndarray

    input_dims: list[int] | None = None
    output_dims: list[int] | None = None

    latex_notation: str = "\\eta"

    def queryBounds(self):
        assert self.input_dims is not None
        self.output_dims = self.input_dims


@dataclass
class Grad:
    input_dims: list[int] | None = None
    output_dims: list[int] | None = None

    latex_notation: str = "\\nabla"

    def queryBounds(self):
        assert self.input_dims is not None
        self.output_dims = [*self.input_dims, 2]


@dataclass
class FFTConv:
    kernel: ndarray

    input_dims: list[int] | None = None
    output_dims: list[int] | None = None

    latex_notation: str = "\\mathcal{F}^T \\mathbf{D} \\mathcal{F}"

    def queryBounds(self):
        assert self.input_dims is not None
        self.output_dims = self.input_dims


@dataclass
class Crop:
    left: int
    top: int
    width: int
    height: int

    input_dims: list[int] | None = None
    output_dims: list[int] | None = None

    latex_notation: str = "\\mathbf{M}"

    def queryBounds(self):
        assert self.input_dims is not None
        self.output_dims = [self.width, self.height]
