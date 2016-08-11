# Utilities for getting the inverse of lin ops.
from __future__ import print_function
import numpy as np
from proximal.prox_fns import least_squares, sum_squares
from proximal.lin_ops import vstack
from proximal.utils import Impl


def get_dims(lin_ops):

    dims = -1
    for op in lin_ops:
        # Recurse on lin op tree.
        while True:
            if hasattr(op, 'dims'):
                if dims == -1:
                    dims = op.dims
                elif dims != op.dims:
                    raise Exception('Dims not consistent')
                    return -1

            if len(op.input_nodes) > 0:
                op = op.input_nodes[0]
            else:
                break

    if dims == -1:
        dims = None

    return dims


def get_implem(lin_ops):

    implem = Impl['numpy']
    for op in lin_ops:
        # Recurse on lin op tree.
        while True:
            if op.implementation != Impl['numpy']:
                implem = op.implementation
                break

            if len(op.input_nodes) > 0:
                op = op.input_nodes[0]
            else:
                break

    return implem


def get_diag_quads(prox_fns, freq):
    """Returns all the quadratic functions that are Gram (freq) diagonal.
    """
    quad_funcs = [fn for fn in prox_fns if isinstance(fn, sum_squares)]
    if freq:
        return [fn for fn in quad_funcs if fn.lin_op.is_diag(freq=True) and
                type(fn) == sum_squares]
    else:
        return [fn for fn in quad_funcs if fn.lin_op.is_diag(freq=False)]


def max_diag_set(prox_fns):
    """Return a maximal cardinality set of quadratic functions with
       diagonalizable lin ops.
    """
    freq_diag = get_diag_quads(prox_fns, True)
    spatial_diag = get_diag_quads(prox_fns, False)
    if len(spatial_diag) >= len(freq_diag):
        return spatial_diag
    else:
        return freq_diag


def get_least_squares_inverse(op_list, b, try_freq_diagonalize=True, verbose=False):
    if len(op_list) == 0:
        return None
    # Are all the operators diagonal?
    stacked = vstack(op_list)
    if stacked.is_gram_diag(freq=False):
        if verbose:
            print('Optimized for diagonal inverse')

        diag = list(stacked.get_diag(freq=False).values())[0]
        diag = diag * np.conj(diag)
        x_update = least_squares(stacked, b, diag=diag)

    # Are all the operators diagonal in the frequency domain?
    elif try_freq_diagonalize and stacked.is_gram_diag(freq=True):

        diag = list(stacked.get_diag(freq=True).values())[0]
        diag = diag * np.conj(diag)
        dims = get_dims(op_list)
        implem = get_implem(op_list)  # If any freqdiag is halide, solve with halide

        if verbose:
            dimstr = (' with dimensionality %d' % dims) if dims is not None else ''
            print('Optimized for diagonal frequency inverse' + dimstr)

        x_update = least_squares(stacked, b,
                                 freq_diag=diag, freq_dims=dims, implem=implem)
    else:

        x_update = least_squares(stacked, b)

    return x_update
