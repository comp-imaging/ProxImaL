from proximal.experimental.ir.problem import Problem
from proximal.experimental.ir.prox_fns import SumSquares
from proximal.experimental.models import LinOp, ProxFn


def hash(lin_ops: list[LinOp]) -> str:
    return " ".join([type(x).__name__ for x in lin_ops])


def group(problem: Problem) -> Problem:
    assert problem.omega_fn is None

    new_prox_fns: list[ProxFn] = []

    for prox_fn in problem.psi_fns:
        if isinstance(prox_fn, SumSquares):
            new_prox_fns.append(prox_fn)
            continue

        current_lin_ops = hash(prox_fn.lin_ops)
        for j, prox_fn2 in enumerate(problem.psi_fns):
            candidate_lin_ops = hash(prox_fn2.lin_ops)

            is_equivalent_lin_ops: bool = candidate_lin_ops == current_lin_ops
            is_sum_squares: bool = isinstance(prox_fn2, SumSquares)
            is_zero_offset: bool = isinstance(prox_fn2.b, float) and prox_fn2.b == 0.0
            scale_is_one: bool = prox_fn2.beta == 1.0

            if (not is_equivalent_lin_ops) or (not is_sum_squares) or (not is_zero_offset) or (not scale_is_one):
                continue

            prox_fn.gamma = prox_fn2.alpha
            problem.psi_fns.pop(j)

        new_prox_fns.append(prox_fn)

    problem.psi_fns = new_prox_fns
    return problem
