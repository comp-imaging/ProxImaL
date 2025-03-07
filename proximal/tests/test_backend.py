import numpy as np

from proximal.experimental.lin_ops import FFTConv, Grad, MultiplyAdd
from proximal.experimental.problem import Problem, Variable
from proximal.experimental.prox_fns import GroupNorm, Nonneg, SumSquares


def test_complex_problem() -> None:
    dims = [512, 512]
    assert Problem(
        u=Variable(shape=dims),
        psi_fns=[
            SumSquares(
                lin_ops=[
                    FFTConv(kernel=np.ones(3)),
                    MultiplyAdd(
                        scale=0.4,
                        offset=-np.ones(dims),
                    ),
                ],
            ),
            GroupNorm(
                alpha=1e-5,
                lin_ops=[
                    Grad(),
                ],
            ),
            SumSquares(
                alpha=1e-3,
                lin_ops=[
                    Grad(),
                ],
            ),
            Nonneg(
                lin_ops=[],
            ),
        ],
    )
