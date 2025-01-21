from dataclasses import dataclass

from proximal.experimental.models import LinOp


@dataclass
class ProxFnBase:
    lin_ops: list[LinOp]

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0
    c: float = 0.0
    d: float = 0.0


@dataclass
class SumSquares(ProxFnBase):
    def toLatex(self) -> str:

        return "\\Vert v \\Vert_2^2"


@dataclass
class FFTConvSumSquares(ProxFnBase):
    def toLatex(self) -> str:

        return "\\Vert F^T H F v \\Vert_2^2"


@dataclass
class GroupNorm(ProxFnBase):
    def toLatex(self) -> str:

        return "\\Vert v \\Vert_{2,1}"


@dataclass
class Nonneg(ProxFnBase):
    def toLatex(self) -> str:

        return "I_+ (v)"
