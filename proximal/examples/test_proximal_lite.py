import sys

sys.path.append("/home/antony/Projects/ProxImaL/")

import numpy as np

from proximal.experimental.lin_ops import FFTConv, Grad, MultiplyAdd
from proximal.experimental.optimize.absorb import absorb
from proximal.experimental.optimize.split import split
from proximal.experimental.problem import Problem, Variable
from proximal.experimental.prox_fns import GroupNorm, Nonneg, SumSquares

dims = [512, 512, 64]

problem = Problem(
    u=Variable(shape=dims),
    psi_fns=[
        SumSquares(
            lin_ops=[
                FFTConv(kernel=np.ones(3), input_dims=dims, output_dims=dims),
                # MultiplyAdd(
                #    scale=0.4,
                #    offset=1.0,
                #    input_dims=dims,
                #    output_dims=dims,
                # ),
            ],
            b=np.ones(dims),
        ),
        GroupNorm(
            alpha=1e-5,
            lin_ops=[
                Grad(input_dims=dims, output_dims=[*dims, 2]),
            ],
        ),
        Nonneg(
            lin_ops=[],
        ),
    ],
)

print(
    f"""Before:
{problem}"""
)

problem = split(absorb(problem))

print(
    f"""After:
{problem}"""
)
