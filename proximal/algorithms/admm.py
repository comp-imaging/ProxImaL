from __future__ import division, print_function
from proximal.lin_ops import CompGraph, scale, vstack
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from .invert import get_least_squares_inverse, get_diag_quads
import numpy as np


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


def solve(psi_fns, omega_fns, rho=1.0,
          max_iters=1000, eps_abs=1e-3, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, try_fast_norm=False,
          scaled=True,
          metric=None, convlog=None, verbose=0):
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    # Rescale so (rho/2)||x - b||^2_2
    rescaling = np.sqrt(2. / rho)
    quad_ops = []
    const_terms = []
    for fn in omega_fns:
        fn = fn.absorb_params()
        quad_ops.append(scale(rescaling * fn.beta, fn.lin_op))
        const_terms.append(fn.b.flatten() * rescaling)
    # Check for fast inverse.
    op_list = [func.lin_op for func in psi_fns] + quad_ops
    stacked_ops = vstack(op_list)

    # Get optimize inverse (tries spatial and frequency diagonalization)
    v_update = get_least_squares_inverse(op_list, None, try_diagonalize, verbose)

    # Initialize everything to zero.
    v = np.zeros(K.input_size)
    z = np.zeros(K.output_size)
    u = np.zeros(K.output_size)

    # Initialize
    if x0 is not None:
        v[:] = np.reshape(x0, K.input_size)
        K.forward(v, z)

    # Buffers.
    Kv = np.zeros(K.output_size)
    KTu = np.zeros(K.input_size)
    s = np.zeros(K.input_size)

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("ADMM iteration")
    # Convergence log for initial iterate
    if convlog is not None:
        K.update_vars(v)
        objval = sum([func.value for func in prox_fns])
        convlog.record_objective(objval)
        convlog.record_timing(0.0)

    for i in range(max_iters):
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()

        z_prev = z.copy()

        # Update v.
        tmp = np.hstack([z - u] + const_terms)
        v = v_update.solve(tmp, x_init=v, lin_solver=lin_solver, options=lin_solver_options)

        # Update z.
        K.forward(v, Kv)
        Kv_u = Kv + u
        offset = 0
        for fn in psi_fns:
            slc = slice(offset, offset + fn.lin_op.size, None)
            Kv_u_slc = np.reshape(Kv_u[slc], fn.lin_op.shape)
            # Apply and time prox.
            prox_log[fn].tic()
            z[slc] = fn.prox(rho, Kv_u_slc, i).flatten()
            prox_log[fn].toc()
            offset += fn.lin_op.size
        # Update u.
        u += Kv - z
        K.adjoint(u, KTu)

        # Check convergence.
        r = Kv - z
        K.adjoint(rho * (z - z_prev), s)
        eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * \
            max([np.linalg.norm(Kv), np.linalg.norm(z)])
        eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / rho

        # Convergence log
        if convlog is not None:
            convlog.toc()
            K.update_vars(v)
            objval = sum([fn.value for fn in prox_fns])
            convlog.record_objective(objval)

        # Show progess
        if verbose > 0:
            # Evaluate objective only if required (expensive !)
            objstr = ''
            if verbose == 2:
                K.update_vars(v)
                objstr = ", obj_val = %02.03e" % sum([fn.value for fn in prox_fns])

            # Evaluate metric potentially
            metstr = '' if metric is None else ", {}".format(metric.message(v))
            print("iter %d: ||r||_2 = %.3f, eps_pri = %.3f, ||s||_2 = %.3f, eps_dual = %.3f%s%s" % (
                i, np.linalg.norm(r), eps_pri, np.linalg.norm(s), eps_dual, objstr, metstr))

        iter_timing.toc()
        # Exit if converged.
        if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
            break

    # Print out timings info.
    if verbose > 0:
        print(iter_timing)
        print("prox funcs:")
        print(prox_log)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    K.update_vars(v)
    # Return optimal value.
    return sum([fn.value for fn in prox_fns])
