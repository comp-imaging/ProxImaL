import numpy as np

from proximal.experimental.frontend import parse
from proximal.experimental.ir.lin_ops import FFTConv, MultiplyAdd


def test_single_fn() -> None:
    assert parse("nonneg(u)")


def test_multiply_add() -> None:
    assert parse("nonneg(u + 1)")
    assert parse("nonneg(u - 1)")
    assert parse("nonneg(3 * u + 1)")
    assert parse("nonneg(3 * u - 1)")
    assert parse("nonneg(3.0 * u)")
    assert parse("nonneg(3.0 * u + 1.0e-3)")
    assert parse("nonneg(3.0 * u - 1.0e-3)")


def test_scaled_fn() -> None:
    assert parse("1.0 * nonneg(u)")
    assert parse("1 * nonneg(u)")
    assert parse("2.3e-3 * nonneg(u)")
    assert parse("2.3e3 * nonneg(u)")


def test_buffer() -> None:
    assert parse("sum_squares(u - b)", const_buffers={"b": np.ones(3)})


def test_conv_op() -> None:
    assert parse("sum_squares(conv(k, u))", const_buffers={"k": np.ones((3, 3))})


def test_complex_problem() -> None:
    blur_kernel = np.ones((3, 3))
    assert parse(
        """
sum_squares(conv(k, u) - 1) +
2.0e-3 * group_norm(grad(u)) +
0.1 * sum_squares(grad(u)) +
nonneg(3 * u - 1)
""",
        const_buffers={"k": blur_kernel},
    )

    dims = [5, 5, 5]
    assert parse(
        """
sum_squares(conv(k, u) - b) +
1.0e-5 * group_norm(grad(u)) +
1.0e-3 * sum_squares(grad(u)) +
nonneg(u)
""",
        variable_dims=dims,
        const_buffers={"k": blur_kernel, "b": np.ones(dims)},
    )


def test_conv_only() -> None:
    dims = [5, 5]
    problem = parse(
        "sum_squares(conv(k, u) - b)",
        variable_dims=dims,
        const_buffers={"k": np.ones((3, 3)), "b": np.ones(dims)},
    )

    assert len(problem.psi_fns) == 1

    prox_fn = problem.psi_fns[0]
    assert len(prox_fn.lin_ops) == 2
    assert isinstance(prox_fn.lin_ops[0], FFTConv)
    assert isinstance(prox_fn.lin_ops[-1], MultiplyAdd)
