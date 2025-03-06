from proximal.experimental.frontend import parse


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


def test_complex_problem() -> None:
    assert parse(
        """
sum_squares(conv(u) - 1) +
2.0e-3 * group_norm(grad(u)) +
0.1 * sum_squares(grad(u)) +
nonneg(3 * u - 1)
"""
    )
