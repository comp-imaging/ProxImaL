from proximal.experimental.ir.problem import Problem
from proximal.experimental.ir.prox_fns import SumSquares
from proximal.experimental.models import LinOp, ProxFn


def hash(lin_ops: list[LinOp]) -> str:
    return " ".join([type(x).__name__ for x in lin_ops])


def group(problem: Problem) -> Problem:
    assert problem.omega_fn is None

    new_prox_fns: list[ProxFn] = []
    absorbed_list: list[int] = []

    for i, prox_fn in enumerate(problem.psi_fns):
        if i in absorbed_list:
            continue

        current_lin_ops = hash(prox_fn.lin_ops)
        for j, prox_fn2 in enumerate(problem.psi_fns):
            if i == j or j in absorbed_list:
                continue

            candidate_lin_ops = hash(prox_fn2.lin_ops)

            is_equivalent_lin_ops: bool = candidate_lin_ops == current_lin_ops
            is_sum_squares: bool = isinstance(prox_fn2, SumSquares)
            is_zero_offset: bool = isinstance(prox_fn2.b, float) and prox_fn2.b == 0.0
            scale_is_one: bool = prox_fn2.beta == 1.0

            if is_equivalent_lin_ops and is_sum_squares and is_zero_offset and scale_is_one:
                prox_fn.gamma = prox_fn2.alpha
                absorbed_list.append(j)
                break

        new_prox_fns.append(prox_fn)

    problem.psi_fns = new_prox_fns
    return problem
