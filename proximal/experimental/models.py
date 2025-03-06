from typing import Protocol, runtime_checkable


@runtime_checkable
class LinOp(Protocol):
    latex_notation: str

    def queryBounds(self) -> None:
        pass


class ProxFn(Protocol):
    alpha: float
    beta: float
    gamma: float
    c: float
    d: float

    lin_ops: list[LinOp]

    def toLatex(self) -> str:
        pass
