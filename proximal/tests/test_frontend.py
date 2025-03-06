from proximal.experimental.frontend import parse
import numpy as np


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


def test_complex_problem() -> None:
    assert parse(
        """
sum_squares(conv(u) - 1) +
2.0e-3 * group_norm(grad(u)) +
0.1 * sum_squares(grad(u)) +
nonneg(3 * u - 1)
"""
    )

    dims = [5, 5, 5]
    assert parse(
        """
sum_squares(conv(u) - b) +
1.0e-5 * group_norm(grad(u)) +
1.0e-3 * sum_squares(grad(u)) +
nonneg(u)
""",
        variable_dims=dims,
        const_buffers={"b": np.ones(dims)},
    )
