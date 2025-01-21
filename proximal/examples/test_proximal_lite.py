import sys

sys.path.append("/home/antony/Projects/ProxImaL/")

import numpy as np

from proximal.experimental.lin_ops import FFTConv, Grad, MultiplyAdd
from proximal.experimental.optimize.absorb import absorb
from proximal.experimental.problem import Problem
from proximal.experimental.prox_fns import GroupNorm, Nonneg, SumSquares

dims = [512, 512, 64]

problem = Problem(
    psi_fns=[
        SumSquares(
            lin_ops=[
                FFTConv(kernel=np.ones(3), input_dims=dims, output_dims=dims),
                MultiplyAdd(
                    scale=0.4,
                    offset=1.0,
                    input_dims=dims,
                    output_dims=dims,
                ),
            ],
        ),
        GroupNorm(
            lin_ops=[
                Grad(input_dims=dims, output_dims=[*dims, 2]),
            ]
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

problem = absorb(problem)

print(
    f"""After:
{problem}"""
)
