from typing import Callable, Protocol, runtime_checkable

from numpy import ndarray


@runtime_checkable
class LinOp(Protocol):
    latex_notation: str
    input_dims: list[int] | None
    output_dims: list[int] | None

    def queryBounds(self) -> None:
        pass


class ProxFn(Protocol):
    alpha: float
    beta: float
    gamma: float
    b: ndarray | float

    lin_ops: list[LinOp]

    def toLatex(self) -> str:
        pass


# Generic numpy functions
LinOpImpl = Callable[[ndarray], ndarray]
