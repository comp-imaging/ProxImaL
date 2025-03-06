import sys

sys.path.append("/home/antony/Projects/ProxImaL/")

import numpy as np

from proximal.experimental.frontend import parse
from proximal.experimental.lin_ops import FFTConv, Grad, MultiplyAdd
from proximal.experimental.optimize.absorb import absorb
from proximal.experimental.optimize.group import group
from proximal.experimental.optimize.split import split
from proximal.experimental.problem import Problem, Variable
from proximal.experimental.prox_fns import GroupNorm, Nonneg, SumSquares
from proximal.experimental.optimize.scale import scale

dims = [512, 512]

problem = Problem(
    u=Variable(shape=dims),
    psi_fns=[
        SumSquares(
            lin_ops=[
                FFTConv(kernel=np.ones(3)),
                # MultiplyAdd(
                #    scale=0.4,
                #    offset=1.0,
                # ),
            ],
            b=np.ones(dims),
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

problem = parse(
    """
sum_squares(conv(u) - b) +
1.0e-5 * group_norm(grad(u)) +
1.0e-3 * sum_squares(grad(u)) +
nonneg(u)
""",
    variable_dims=dims,
    const_buffers={"b": np.ones(dims)},
)

print(
    f"""Before:
{problem}"""
)

problem = split(group(absorb(problem)))

print(
    f"""After:
{problem}"""
)

problem = scale(problem)
print(problem.Knorm)