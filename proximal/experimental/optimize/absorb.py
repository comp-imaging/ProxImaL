from proximal.experimental.ir.lin_ops import Crop, FFTConv, MultiplyAdd
from proximal.experimental.ir.problem import Problem
from proximal.experimental.ir.prox_fns import LeastSquaresFFT, SumSquares, WeightedLeastSquares
from proximal.experimental.models import ProxFn


def absorbFFTConv(prox_fn: SumSquares) -> tuple[bool, SumSquares | LeastSquaresFFT]:
    is_lin_ops_empty: bool = len(prox_fn.lin_ops) == 0
    if is_lin_ops_empty:
        # Nothing to absorb
        return False, prox_fn

    lin_op = prox_fn.lin_ops[-1]
    has_fft_conv: bool = isinstance(lin_op, FFTConv)
    if not has_fft_conv:
        # Only FFTConv is supported. Skipping...
        return False, prox_fn

    # A trick to force the static analyzer to recognize the FFTConv type
    assert isinstance(lin_op, FFTConv)

    return True, LeastSquaresFFT(
        alpha=prox_fn.alpha,
        gamma=prox_fn.gamma,
        # todo: pre-compute FFT
        freq_diag=lin_op.kernel * prox_fn.beta,
        new_b=prox_fn.b,
        lin_ops=prox_fn.lin_ops[:-1],
    )


def absorbMultiplyAdd(prox_fn: ProxFn) -> tuple[bool, ProxFn]:
    """Absorb (a * x + b) into the proximal function."""

    if len(prox_fn.lin_ops) == 0 or not isinstance(prox_fn.lin_ops[-1], MultiplyAdd):
        return False, prox_fn

    scale = prox_fn.lin_ops[-1].scale
    offset = prox_fn.lin_ops[-1].offset
    prox_fn.beta *= scale
    prox_fn.b = prox_fn.b - prox_fn.beta * offset
    prox_fn.lin_ops = prox_fn.lin_ops[:-1]

    return True, prox_fn


def absorbCrop(prox_fn: SumSquares) -> tuple[bool, ProxFn]:
    """sum_square(Crop(u)) -> WeighteddLeastSquare(u)."""

    if len(prox_fn.lin_ops) == 0 or not isinstance(prox_fn.lin_ops[-1], Crop):
        return False, prox_fn

    # Generate the values of the binary mask representing the crop
    crop_op: Crop = prox_fn.lin_ops[-1]

    def mask(x: int, y: int) -> float:
        return (crop_op.left <= x < crop_op.left + crop_op.width) and (crop_op.top <= x < crop_op.top + crop_op.height)

    return True, WeightedLeastSquares(
        lin_ops=prox_fn.lin_ops[:-1],
        alpha=prox_fn.alpha,
        beta=prox_fn.beta,
        gamma=prox_fn.gamma,
        b=prox_fn.b,
        weights=mask,
    )


def absorb(problem: Problem) -> Problem:
    assert problem.omega_fn is None, "Problem is already split, why?"

    for i in range(len(problem.psi_fns)):
        should_retry = True
        while should_retry:
            psi_fn = problem.psi_fns[i]
            should_retry, psi_fn = absorbMultiplyAdd(psi_fn)

            if isinstance(psi_fn, SumSquares):
                is_success, psi_fn = absorbFFTConv(psi_fn)
                should_retry = should_retry or is_success

            if isinstance(psi_fn, SumSquares):
                is_success, psi_fn = absorbCrop(psi_fn)
                should_retry = should_retry or is_success

            problem.psi_fns[i] = psi_fn

    return problem
