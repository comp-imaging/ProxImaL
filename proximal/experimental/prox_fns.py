from dataclasses import dataclass
from math import isclose

from numpy import ndarray

from proximal.experimental.models import LinOp


@dataclass
class ProxFnBase:
    lin_ops: list[LinOp]

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0

    b: ndarray | float = 0.0

    def formatParameters(self) -> tuple[str, str, str, str]:
        _alpha = f"{self.alpha:0.3g}" if not isclose(self.alpha, 1.0) else ""
        _beta = f"{self.beta:0.3g}" if not isclose(self.beta, 1.0) else ""

        if self.gamma == 0.0:
            _gamma = ""
        elif isclose(self.gamma, 1.0):
            _gamma = "+ \\Vert v \\Vert_2^2"
        else:
            _gamma = f"+ {self.gamma:0.3g} \\Vert v \\Vert_2^2"

        if isinstance(self.b, ndarray):
            _b = "- b"
        elif isclose(self.b, 0.0):
            _b = ""
        else:
            _b = f"{self.b:0.3g}" if self.b < 0 else f"+ {self.b:0.3g}"

        return _alpha, _beta, _gamma, _b


@dataclass
class SumSquares(ProxFnBase):
    def toLatex(self) -> str:
        alpha, beta, gamma, b = self.formatParameters()
        return f"{alpha} \\Vert {beta} v {b} \\Vert_2^2 {gamma}"


@dataclass
class FFTConvSumSquares(ProxFnBase):
    def toLatex(self) -> str:

        alpha, beta, gamma, b = self.formatParameters()
        return f"{alpha} \\Vert {beta} F^T H F v {b} \\Vert_2^2 {gamma}"


@dataclass
class GroupNorm(ProxFnBase):
    def toLatex(self) -> str:

        alpha, beta, gamma, b = self.formatParameters()
        return f"{alpha} \\Vert {beta} v {b} \\Vert_{{2, 1}} {gamma}"


@dataclass
class Nonneg(ProxFnBase):
    def toLatex(self) -> str:

        _, beta, gamma, b = self.formatParameters()
        return f"I_+ ({beta} v {b}) {gamma}"
