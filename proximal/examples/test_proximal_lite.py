import numpy as np

from proximal.experimental.codegen import LinearizedADMM
from proximal.experimental.frontend import parse
from proximal.experimental.optimize.absorb import absorb
from proximal.experimental.optimize.group import group
from proximal.experimental.optimize.scale import scale
from proximal.experimental.optimize.split import split

input_width = 512
output_width = 256
kernel_size = 5

offset = (input_width - output_width) // 2
assert offset >= 0

problem = parse(
    f"""
sum_squares(conv(k, u)[
    {offset}:{offset + output_width}, {offset}:{offset + output_width}
] - b) +
1.0e-5 * group_norm(grad(u)) +
1.0e-3 * sum_squares(grad(u)) +
nonneg(u)
""",
    variable_dims=[input_width, input_width],
    const_buffers={
        "b": np.ones(
            (output_width, output_width),
            order="F",
            dtype=np.float32,
        ),
        "k": np.ones(
            (kernel_size, kernel_size),
            order="F",
            dtype=np.float32,
        )
        / (kernel_size * kernel_size),
    },
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