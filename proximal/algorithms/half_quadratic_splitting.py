from __future__ import print_function
import numpy as np
import math
from proximal.lin_ops import CompGraph, scale, vstack
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from .invert import get_least_squares_inverse, get_diag_quads


def partition(prox_fns, try_diagonalize=True):
    """Divide the proxable functions into sets Psi and Omega.
    """
    # Merge these quadratic functions with the v update.
    quad_funcs = []
    # All lin ops must be gram diagonal for the least squares problem
    # to be diagonal.
    func_opts = {True: [], False: []}
    for freq in [True, False]:
        if all([fn.lin_op.is_gram_diag(freq) for fn in prox_fns]):
            func_opts[freq] = get_diag_quads(prox_fns, freq)
    # Quad funcs is the max cardinality set.
    if len(func_opts[True]) >= len(func_opts[False]):
        quad_funcs = func_opts[True]
    else:
        quad_funcs = func_opts[False]
    psi_fns = [fn for fn in prox_fns if fn not in quad_funcs]
    return psi_fns, quad_funcs


def solve(psi_fns, omega_fns,
          rho_0=1.0, rho_scale=math.sqrt(2.0) * 2.0, rho_max=2**8,
          max_iters=-1, max_inner_iters=100, x0=None,
          eps_rel=1e-3, eps_abs=1e-3,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, scaled=False, try_fast_norm=False,
          metric=None, convlog=None, verbose=0):
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    # Rescale so (1/2)||x - b||^2_2
    rescaling = np.sqrt(2.)
    quad_ops = []
    quad_weights = []
    const_terms = []
    for fn in omega_fns:
        fn = fn.absorb_params()
        quad_ops.append(scale(rescaling * fn.beta, fn.lin_op))
        quad_weights.append(rescaling * fn.beta)
        const_terms.append(fn.b.flatten() * rescaling)

    # Get optimize inverse (tries spatial and frequency diagonalization)
    op_list = [func.lin_op for func in psi_fns] + quad_ops
    stacked_ops = vstack(op_list)
    x_update = get_least_squares_inverse(op_list, None,
                                         try_diagonalize, verbose)

    # Initialize
    if x0 is not None:
        x = np.reshape(x0, K.input_size)
    else:
        x = np.zeros(K.input_size)

    Kx = np.zeros(K.output_size)
    w = Kx.copy()

    # Temporary iteration counts
    x_prev = x.copy()

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("HQS iteration")
    inner_iter_timing = TimingsEntry("HQS inner iteration")
    # Convergence log for initial iterate
    if convlog is not None:
        K.update_vars(x)
        objval = sum([func.value for func in prox_fns])
        convlog.record_objective(objval)
        convlog.record_timing(0.0)

    # Rho scedule
    rho = rho_0
    i = 0
    while rho < rho_max and i < max_iters:
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()

        # Update rho for quadratics
        for idx, op in enumerate(quad_ops):
            op.scalar = quad_weights[idx] / np.sqrt(rho)
        x_update = get_least_squares_inverse(op_list, CompGraph(stacked_ops),
                                             try_diagonalize, verbose)

        for ii in range(max_inner_iters):
            inner_iter_timing.tic()
            # Update Kx.
            K.forward(x, Kx)

            # Prox update to get w.
            offset = 0
            w_prev = w.copy()
            for fn in psi_fns:
                slc = slice(offset, offset + fn.lin_op.size, None)
                # Apply and time prox.
                prox_log[fn].tic()
                w[slc] = fn.prox(rho, np.reshape(Kx[slc], fn.lin_op.shape), ii).flatten()
                prox_log[fn].toc()
                offset += fn.lin_op.size

            # Update x.
            x_prev[:] = x
            tmp = np.hstack([w] + [cterm / np.sqrt(rho) for cterm in const_terms])
            x = x_update.solve(tmp, x_init=x, lin_solver=lin_solver, options=lin_solver_options)

            # Very basic convergence check.
            r_x = np.linalg.norm(x_prev - x)
            eps_x = eps_rel * np.prod(K.input_size)

            r_w = np.linalg.norm(w_prev - w)
            eps_w = eps_rel * np.prod(K.output_size)

            # Convergence log
            if convlog is not None:
                convlog.toc()
                K.update_vars(x)
                objval = sum([fn.value for fn in prox_fns])
                convlog.record_objective(objval)

            # Show progess
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    K.update_vars(x)
                    objstr = ", obj_val = %02.03e" % sum([fn.value for fn in prox_fns])

                # Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format(metric.message(x))
                print("iter [%02d (rho=%2.1e) || %02d]:"
                      "||w - w_prev||_2 = %02.02e (eps=%02.03e)"
                      "||x - x_prev||_2 = %02.02e (eps=%02.03e)%s%s"
                      % (i, rho, ii, r_x, eps_x, r_w, eps_w, objstr, metstr))

            inner_iter_timing.toc()
            if r_x < eps_x and r_w < eps_w:
                break

        # Update rho
        rho = np.minimum(rho * rho_scale, rho_max)
        i += 1
        iter_timing.toc()

    # Print out timings info.
    if verbose > 0:
        print(iter_timing)
        print(inner_iter_timing)
        print("prox funcs:")
        print(prox_log)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    K.update_vars(x)

    # Return optimal value.
    return sum([fn.value for fn in prox_fns])
