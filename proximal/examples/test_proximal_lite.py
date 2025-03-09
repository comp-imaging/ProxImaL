import sys

sys.path.append("/home/antony/Projects/ProxImaL/")

import numpy as np

from proximal.experimental.frontend import parse
from proximal.experimental.lin_ops import Crop
from proximal.experimental.optimize.absorb import absorb
from proximal.experimental.optimize.group import group
from proximal.experimental.optimize.split import split
from proximal.experimental.optimize.scale import scale
from proximal.experimental.codegen import LinearizedADMM

dims = [512, 512]
out_dims = [128, 128]
N = 5

problem = parse(
    """
sum_squares(conv(k, u) - b) +
1.0e-5 * group_norm(grad(u)) +
1.0e-3 * sum_squares(grad(u)) +
nonneg(u)
""",
    variable_dims=dims,
    const_buffers={"b": np.ones(out_dims),
                   'k': np.ones((N, N), order='F', dtype=np.float32) / (N*N)},
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
print(f"Knorm = {problem.Knorm}")

print(LinearizedADMM(problem).generateProblemDefinition())