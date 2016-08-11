from __future__ import division, print_function
from proximal.lin_ops import CompGraph, est_CompGraph_norm, Variable, vstack
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from .invert import get_least_squares_inverse, max_diag_set
import numpy as np
import warnings


def partition(prox_fns, try_diagonalize=True):
    """Divide the proxable functions into sets Psi and Omega.
    """
    # Omega must be a single function.
    # Merge quadratic functions into the x update.
    # Automatically try to split the problem.
    quad_fns = max_diag_set(prox_fns)
    split_fn = []
    omega_fns = []
    if len(quad_fns) == 0:
        for fn in prox_fns:
            if type(fn.lin_op) == Variable:
                split_fn = [fn]
                break
        omega_fns = split_fn
    else:
        # Proximal solve for:
        # G(x) + 1/(2*tau) * ||x - v||^2_2, with G containing all quadratics
        quad_ops = []
        const_terms = []
        for fn in quad_fns:
            fn = fn.absorb_params()
            quad_ops.append(fn.beta * fn.lin_op)
            const_terms.append(fn.b.flatten())

        b = np.hstack(const_terms)
        # Get optimize inverse (tries spatial and frequency diagonalization)
        x_update = get_least_squares_inverse(quad_ops, b, try_diagonalize)
        omega_fns = [x_update]

    psi_fns = [func for func in prox_fns if func not in split_fn + quad_fns]
    return psi_fns, omega_fns


def solve(psi_fns, omega_fns, lmb=1.0, mu=None, quad_funcs=None,
          max_iters=1000, eps_abs=1e-3, eps_rel=1e-3,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, try_fast_norm=True, scaled=False,
          metric=None, convlog=None, verbose=0):

    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    # Select optimal parameters if wanted
    if lmb is None or mu is None:
        lmb, mu = est_params_lin_admm(K, lmb, verbose, scaled, try_fast_norm)

    # Initialize everything to zero.
    v = np.zeros(K.input_size)
    z = np.zeros(K.output_size)
    u = np.zeros(K.output_size)

    # Buffers.
    Kv = np.zeros(K.output_size)
    KTu = np.zeros(K.input_size)
    s = np.zeros(K.input_size)

    Kvzu = np.zeros(K.output_size)
    v_prev = np.zeros(K.input_size)
    z_prev = np.zeros(K.output_size)

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("LIN-ADMM iteration")
    # Convergence log for initial iterate
    if convlog is not None:
        K.update_vars(v)
        objval = sum([fn.value for fn in prox_fns])
        convlog.record_objective(objval)
        convlog.record_timing(0.0)

    for i in range(max_iters):
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()

        v_prev[:] = v
        z_prev[:] = z

        # Update v
        K.forward(v, Kv)
        Kvzu[:] = Kv - z + u
        K.adjoint(Kvzu, v)
        v[:] = v_prev - (mu / lmb) * v

        if len(omega_fns) > 0:
            v[:] = omega_fns[0].prox(1.0 / mu, v, x_init=v_prev.copy(),
                                     lin_solver=lin_solver, options=lin_solver_options)

        # Update z.
        K.forward(v, Kv)
        Kv_u = Kv + u
        offset = 0
        for fn in psi_fns:
            slc = slice(offset, offset + fn.lin_op.size, None)
            Kv_u_slc = np.reshape(Kv_u[slc], fn.lin_op.shape)
            # Apply and time prox.
            prox_log[fn].tic()
            z[slc] = fn.prox(1.0 / lmb, Kv_u_slc, i).flatten()
            prox_log[fn].toc()
            offset += fn.lin_op.size

        # Update u.
        u += Kv - z
        K.adjoint(u, KTu)

        # Check convergence.
        r = Kv - z
        K.adjoint((1.0 / lmb) * (z - z_prev), s)
        eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * \
            max([np.linalg.norm(Kv), np.linalg.norm(z)])
        eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / (1.0 / lmb)

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


def est_params_lin_admm(K, lmb=None, verbose=True, scaled=False, try_fast_norm=False):

    # Select lambda
    lmb = 1.0 if lmb is None else np.maximum(lmb, 1e-5)

    # Warn user
    if lmb > 1.0:
        warnings.warn("Large lambda value given by user.")

    # Estimate Lipschitz constant and comput tau
    if scaled:
        L = 1
    else:
        L = est_CompGraph_norm(K, try_fast_norm)
    mu = lmb / (L**2)

    if verbose:
        print("Estimated params [lambda = %3.3f | mu = %3.3f | L_est = %3.4f]" % (lmb, mu, L))

    return lmb, mu
