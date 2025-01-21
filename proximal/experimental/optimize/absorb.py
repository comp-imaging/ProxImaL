from proximal.experimental.lin_ops import FFTConv, MultiplyAdd
from proximal.experimental.models import ProxFn
from proximal.experimental.problem import Problem
from proximal.experimental.prox_fns import LeastSquaresFFT, SumSquares


def absorbFFTConv(prox_fn: SumSquares) -> LeastSquaresFFT:
    is_lin_ops_empty: bool = len(prox_fn.lin_ops) == 0
    has_fft_conv: bool = isinstance(prox_fn.lin_ops[-1], FFTConv)
    if is_lin_ops_empty or not has_fft_conv:
        # Only FFTConv is supported. Skipping...
        return prox_fn

    return LeastSquaresFFT(
        alpha=prox_fn.alpha,
        gamma=prox_fn.gamma,
        # todo: pre-compute FFT
        freq_diag=prox_fn.lin_ops[-1].kernel * prox_fn.beta,
        new_b=prox_fn.b,
        lin_ops=prox_fn.lin_ops[:-1],
    )


def absorbMultiplyAdd(prox_fn: ProxFn) -> ProxFn:
    """Absorb (a * x + b) into the proximal function."""

    if len(prox_fn.lin_ops) == 0 or not isinstance(prox_fn.lin_ops[-1], MultiplyAdd):
        return prox_fn

    scale = prox_fn.lin_ops[-1].scale
    offset = prox_fn.lin_ops[-1].offset
    prox_fn.beta *= scale
    prox_fn.b -= prox_fn.beta * offset
    prox_fn.lin_ops = prox_fn.lin_ops[:-1]

    return prox_fn


def absorb(problem: Problem) -> Problem:
    assert problem.omega_fn is None, "Problem is already split, why?"

    for i, psi_fn in enumerate(problem.psi_fns):
        problem.psi_fns[i] = absorbMultiplyAdd(psi_fn)

        if isinstance(psi_fn, SumSquares):
            problem.psi_fns[i] = absorbFFTConv(psi_fn)

    return problem
