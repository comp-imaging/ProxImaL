from proximal.experimental.models import ProxFn
from proximal.experimental.problem import Problem
from proximal.experimental.prox_fns import SumSquares


def findSimpleSumSquares(prox_fns: list[ProxFn]) -> list[int]:
    """Find all the L2-norms having no linear operators."""
    idx: list[int] = []

    for i, prox_fn in enumerate(prox_fns):
        if isinstance(prox_fn, SumSquares) and len(prox_fn.lin_ops) == 0:
            idx.append(i)

    return idx


def findFirstSimpleProxFn(prox_fns: list[ProxFn]) -> int | None:
    """Find the proximal function having no linear operators."""
    for i, prox_fn in enumerate(prox_fns):
        if len(prox_fn.lin_ops) == 0:
            return i

    return None


def split(problem: Problem) -> Problem:
    sum_squares_idx = findSimpleSumSquares(problem.psi_fns)

    if len(sum_squares_idx) > 0:
        # todo: merge multiple SumSquares into one single LeastSquares
        first_term = sum_squares_idx[0]
        problem.omega_fn = problem.psi_fns[first_term]

        new_psi_fns: list[ProxFn] = []
        for i, psi_fn in enumerate(problem.psi_fns):
            if i in sum_squares_idx:
                continue
            new_psi_fns.append(psi_fn)

        problem.psi_fns = new_psi_fns
        return problem

    simple_prox_fn_idx = findFirstSimpleProxFn(problem.psi_fns)
    if simple_prox_fn_idx is not None:
        problem.omega_fn = problem.psi_fns[simple_prox_fn_idx]
        problem.psi_fns.pop(simple_prox_fn_idx)

        return problem

    return problem
