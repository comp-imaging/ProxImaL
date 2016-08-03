# Absorb linear operators into proximal operators.

from proximal.lin_ops import Variable, scale, mul_elemwise, Constant
from proximal.prox_fns import (nonneg, weighted_nonneg, norm1, weighted_norm1, poisson_norm,
                               weighted_poisson_norm, sum_squares, weighted_sum_squares,
                               group_norm1, weighted_group_norm1, zero_prox)
import numpy as np
import copy as cp

WEIGHTED = {nonneg: weighted_nonneg,
            norm1: weighted_norm1,
            sum_squares: weighted_sum_squares,
            poisson_norm: weighted_poisson_norm,
            group_norm1: weighted_group_norm1}


def absorb_all_lin_ops(prox_funcs):
    """Repeatedy absorb lin ops.
    """
    new_proxes = []
    ready = prox_funcs[:]
    while len(ready) > 0:
        curr = ready.pop(0)
        absorbed = absorb_lin_op(curr)
        if len(absorbed) == 1 and absorbed[0] == curr:
            new_proxes.append(absorbed[0])
        else:
            ready += absorbed
    return new_proxes


def absorb_lin_op(prox_fn):
    """If possible moves the top level lin op argument
       into the prox operator.

       For example, elementwise multiplication can be folded into
       a separable function's prox.
    """
    # Never eliminate variables.
    if isinstance(prox_fn.lin_op, Variable):
        return [prox_fn]
    # Absorb a lin op into sum_entries/zero.
    if type(prox_fn) == zero_prox and prox_fn.gamma == 0:
        outputs = []
        inputs = [prox_fn.c]
        for arg in prox_fn.lin_op.input_nodes:
            outputs.append(np.zeros(arg.shape))
        prox_fn.lin_op.adjoint(inputs, outputs)
        new_proxes = []
        for output, arg in zip(outputs, prox_fn.lin_op.input_nodes):
            new_proxes.append(prox_fn.copy(arg, c=output))
        return new_proxes
    # Fold scaling into the function.
    if isinstance(prox_fn.lin_op, mul_elemwise):
        op_weight = prox_fn.lin_op.weight

        def get_new_prox(prox_type, args):
            new_prox = prox_type(*args)
            copy_prox_fn(new_prox, prox_fn)
            idxs = op_weight != 0
            new_prox.b[idxs] = prox_fn.b[idxs] / op_weight[idxs]
            new_prox.c = prox_fn.c * op_weight
            return [new_prox]

        if type(prox_fn) in WEIGHTED.keys() and prox_fn.gamma == 0:
            args = [prox_fn.lin_op.input_nodes[0]] + prox_fn.get_data() + \
                   [op_weight]
            return get_new_prox(WEIGHTED[type(prox_fn)], args)
        elif type(prox_fn) in WEIGHTED.values() and prox_fn.gamma == 0:
            args = [prox_fn.lin_op.input_nodes[0]] + prox_fn.get_data()
            args[-1] = args[-1] * op_weight
            return get_new_prox(type(prox_fn), args)
    # Fold scalar into the function.
    if isinstance(prox_fn.lin_op, scale):
        scalar = prox_fn.lin_op.scalar
        new_prox = prox_fn.copy(prox_fn.lin_op.input_nodes[0],
                                beta=prox_fn.beta * scalar, b=prox_fn.b / scalar,
                                c=prox_fn.c * scalar,
                                gamma=prox_fn.gamma * scalar**2)
        return [new_prox]
    # No change.
    return [prox_fn]


def copy_prox_fn(dst_prox, src_prox):
    """Copy the optional parameters from src_prox to dst_prox.
    """
    dst_prox.alpha = src_prox.alpha
    dst_prox.beta = src_prox.beta
    dst_prox.gamma = src_prox.gamma
    dst_prox.b = src_prox.b
    dst_prox.c = src_prox.c
    dst_prox.d = src_prox.d


def copy_non_var(lin_op):
    """If not a variable, returns a shallow copy.
    """
    if isinstance(lin_op, Variable):
        return lin_op
    else:
        return cp.copy(lin_op)


def absorb_offset(prox_fn):
    """Absorb the constant offset into the b term and zero out constants in lin op.
    """
    # Short circuit if no constant leaves.
    if len(prox_fn.lin_op.constants()) == 0:
        return prox_fn
    new_b = -prox_fn.lin_op.get_offset()
    # Zero out constants.
    new_lin_op = copy_non_var(prox_fn.lin_op)
    ready = [new_lin_op]
    while len(ready) > 0:
        curr = ready.pop(0)
        for idx, arg in enumerate(curr.input_nodes):
            if isinstance(arg, Constant):
                curr.input_nodes[idx] = Constant(np.zeros(arg.shape))
            # Don't copy variables.
            else:
                curr.input_nodes[idx] = copy_non_var(arg)
            ready.append(curr.input_nodes[idx])
    return prox_fn.copy(new_lin_op, b=new_b + prox_fn.b)
