from proximal.experimental.lin_ops import MultiplyAdd
from proximal.experimental.models import ProxFn
from proximal.experimental.problem import Problem


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

    for psi_fn in problem.psi_fns:
        problem.psi_fn = absorbMultiplyAdd(psi_fn)

    return problem
