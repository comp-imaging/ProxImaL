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


@dataclass
class Variable:
    shape: list[int]


class Problem:
    psi_fns: list[ProxFn]
    u: Variable
    omega_fn: ProxFn | None = None

    Knorm: float = 1.0

    def __init__(self, psi_fns: list[ProxFn], u: Variable) -> None:
        self.psi_fns = psi_fns
        self.u = u

    def __repr__(self) -> str:
        formatted = f"""\\begin{{align}}
\\hat u &= \\arg \\min_{{u \\in \\mathbb{{R}}^{len(self.u.shape)} }}
f(u) + \\sum_{{j=1}}^{len(self.psi_fns)} g_j\\left( \\mathbf{{K}}_j u \\right) \\\\
"""
        if self.omega_fn is None:
            formatted += f"f(u) &= \\emptyset \\\\\n"
        else:
            formatted += f"f(u) &= {self.omega_fn.toLatex().replace('v', 'u')} \\\\\n"

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
            formatted += f"g_{e.idx}(v) &= {e.fn} & \\mathbf{{K}}_{e.idx} &= {e.K} \\\\\n"

        formatted += """\\end{align}
"""

        return formatted

    def propagateBounds(self) -> None:
        for fn in self.psi_fns:
            current_dims = self.u.shape

            for lin_op in fn.lin_ops:
                lin_op.input_dims = current_dims
                lin_op.queryBounds()
                current_dims = lin_op.output_dims

        current_dims = self.u.shape
        for lin_op in self.omega_fn.lin_ops:
            lin_op.input_dims = current_dims
            lin_op.queryBounds()
            current_dims = lin_op.output_dims
