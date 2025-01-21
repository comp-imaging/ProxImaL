from typing import Protocol


class LinOp(Protocol):
    input_dims: list[int]
    output_dims: list[int]
    latex_notation: str


class ProxFn(Protocol):
    alpha: float
    beta: float
    gamma: float
    c: float
    d: float

    lin_ops: list[LinOp]

    def toLatex(self) -> str:
        pass
