from dataclasses import dataclass

from proximal.experimental.models import LinOp, ProxFn


@dataclass
class CostFnFormatted:
    idx: int
    fn: str
    K: str


def toLatex(lin_ops: list[LinOp]) -> str:
    if len(lin_ops) == 0:
        return "I"

    formatted: str = ""
    for lin_op in lin_ops:
        formatted = f"{lin_op.latex_notation:s} {formatted:s}"

    return formatted


class Problem:
    psi_fns: list[ProxFn]
    omega_fn: ProxFn | None = None

    Knorm: float = 1.0

    def __init__(self, psi_fns: list[ProxFn]) -> None:
        self.psi_fns = psi_fns

    def __repr__(self) -> str:

        formatted = f"""\\begin{{align}}
\hat u &= \\arg \\min_u f(u) + \\sum_{{j=1}}^{len(self.psi_fns)} g_j\\left( K_j u \\right) \\\\
"""
        if self.omega_fn is not None:
            formatted += f"f(u) = {self.omega_fn.toLatex()} \\\\\n"

        equations: list[CostFnFormatted] = []
        for i, psi_fn in enumerate(self.psi_fns):
            equations.append(
                CostFnFormatted(
                    idx=i + 1,
                    fn=psi_fn.toLatex(),
                    K=toLatex(psi_fn.lin_ops),
                )
            )

        for e in equations:
            formatted += f"g_{e.idx}(v) &= {e.fn} & K_{e.idx} &= {e.K} \\\\\n"

        formatted += """\\end{align}
"""

        return formatted
