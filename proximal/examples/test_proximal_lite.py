import sys

sys.path.append("/home/antony/Projects/ProxImaL/")

import numpy as np

from proximal.experimental.frontend import parse
from proximal.experimental.optimize.absorb import absorb
from proximal.experimental.optimize.group import group
from proximal.experimental.optimize.split import split
from proximal.experimental.optimize.scale import scale

dims = [512, 512]

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