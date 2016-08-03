# Merge proximal operators together.

from proximal.prox_fns import sum_squares, zero_prox
import numpy as np


def merge_all(prox_fns):
    """Merge as many prox functions as possible.
    """
    while True:
        merged = []
        new_prox_fns = []
        no_merges = True
        for i in range(len(prox_fns)):
            for j in range(i + 1, len(prox_fns)):
                if prox_fns[i] not in merged and prox_fns[j] not in merged and \
                   can_merge(prox_fns[i], prox_fns[j]):
                    no_merges = False
                    merged += [prox_fns[i], prox_fns[j]]
                    new_prox_fns.append(merge_fns(prox_fns[i], prox_fns[j]))
        if no_merges:
            break
        prox_fns = new_prox_fns + [fn for fn in prox_fns if fn not in merged]

    return prox_fns


def can_merge(lh_prox, rh_prox):
    """Can lh_prox and rh_prox be merged into a single function?
    """
    # Lin ops must be the same.
    if lh_prox.lin_op == rh_prox.lin_op:
        if type(lh_prox) == zero_prox or type(rh_prox) == zero_prox:
            return True
        elif type(lh_prox) == sum_squares or type(rh_prox) == sum_squares:
            return True

    return False


def merge_fns(lh_prox, rh_prox):
    """Merge the two functions into a single function.
    """
    assert can_merge(lh_prox, rh_prox)
    new_c = lh_prox.c + rh_prox.c
    new_gamma = lh_prox.gamma + rh_prox.gamma
    new_d = lh_prox.d + rh_prox.d
    args = [lh_prox, rh_prox]
    arg_types = [type(lh_prox), type(rh_prox)]
    # Merge a linear term into the other proxable function.
    if zero_prox in arg_types:
        to_copy = args[1 - arg_types.index(zero_prox)]
        return to_copy.copy(c=new_c, gamma=new_gamma, d=new_d)
    # Merge a sum squares term into the other proxable function.
    elif sum_squares in arg_types:
        idx = arg_types.index(sum_squares)
        sq_fn = args[idx]
        to_copy = args[1 - idx]
        coeff = sq_fn.alpha * sq_fn.beta
        return to_copy.copy(c=new_c - 2 * coeff * sq_fn.b,
                            gamma=new_gamma + coeff * sq_fn.beta,
                            d=new_d + np.square(sq_fn.b).sum())
    else:
        raise ValueError("Unknown merge strategy.")
