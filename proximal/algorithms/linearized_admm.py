from __future__ import division, print_function
from proximal.lin_ops import CompGraph, est_CompGraph_norm, Variable, vstack
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from .invert import get_least_squares_inverse, max_diag_set
import numpy as np
import numexpr as ne
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
            if isinstance(fn.lin_op, Variable):
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


def solve(
    psi_fns,
    omega_fns,
    lmb=1.0,
    mu=None,
    quad_funcs=None,
    max_iters=1000,
    eps_abs=1e-3,
    eps_rel=1e-3,
    lin_solver="cg",
    lin_solver_options=None,
    implem=None,
    try_diagonalize=True,
    try_fast_norm=True,
    scaled=False,
    metric=None,
    convlog=None,
    verbose=0,
    conv_check=20,
):

    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops, implem=implem)
    # Select optimal parameters if wanted
    if lmb is None or mu is None:
        lmb, mu = est_params_lin_admm(K, lmb, verbose, scaled, try_fast_norm)

    # Initialize everything to zero.
    v = np.zeros(K.input_size, dtype=np.float32, order='F')
    z = np.zeros(K.output_size, dtype=np.float32, order='F')
    u = np.zeros(K.output_size, dtype=np.float32, order='F')

    # Buffers.
    Kv = np.zeros(K.output_size, dtype=np.float32, order='F')
    KTu = np.zeros(K.input_size, dtype=np.float32, order='F')
    s = np.zeros(K.input_size, dtype=np.float32, order='F')

    Kvzu = np.zeros(K.output_size, dtype=np.float32, order='F')
    v_prev = np.zeros(K.input_size, dtype=np.float32, order='F')
    z_prev = np.zeros(K.output_size, dtype=np.float32, order='F')

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
        ne.evaluate('Kv - z + u', out=Kvzu)
        K.adjoint(Kvzu, v)
        ne.evaluate('v_prev - (mu / lmb) * v', out=v, casting='unsafe')

        if len(omega_fns) > 0:
            prox_log[omega_fns[0]].tic()
            v_shape = omega_fns[0].lin_op.shape
            v[:] = omega_fns[0].prox(1.0 / mu, np.asfortranarray(v.reshape(v_shape)), x_init=v_prev.copy(),
                                     lin_solver=lin_solver, options=lin_solver_options).ravel()
            prox_log[omega_fns[0]].toc()

        # Update z.
        K.forward(v, Kv)
        Kv_u = ne.evaluate('Kv + u')
        offset = 0
        for fn in psi_fns:
            slc = slice(offset, offset + fn.lin_op.size, None)
            Kv_u_slc = np.asfortranarray(np.reshape(Kv_u[slc], fn.lin_op.shape))
            # Apply and time prox.
            prox_log[fn].tic()
            z[slc] = fn.prox(1.0 / lmb, Kv_u_slc, i).ravel()
            prox_log[fn].toc()
            offset += fn.lin_op.size

        # Update u.
        ne.evaluate('u + Kv - z', out=u)
        K.adjoint(u, KTu)

        # Convergence log
        if convlog is not None:
            convlog.toc()
            K.update_vars(v)
            objval = sum([fn.value for fn in prox_fns])
            convlog.record_objective(objval)

        should_check_convergence: bool = i % conv_check == 0
        if should_check_convergence:
            # Check convergence.
            r = ne.evaluate("Kv - z")
            ztmp = ne.evaluate("(z - z_prev) / lmb")
            K.adjoint(ztmp, s)

            eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * max(
                [
                    np.linalg.norm(Kv.astype(np.float64)),
                    np.linalg.norm(z.astype(np.float64)),
                ]
            )
            eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(
                KTu.astype(np.float64)
            ) / (1.0 / lmb)

        # Show progess
        if verbose > 0 and should_check_convergence:
            # Evaluate objective only if required (expensive !)
            objstr = ''
            if verbose == 2:
                K.update_vars(v)
                objstr = f", obj_val = {sum([fn.value for fn in prox_fns]):02.03e}"

            # Evaluate metric potentially
            metstr = "" if metric is None else f", {metric.message(v)}"
            print(
                f"iter {i:d}: ||r||_2 = {np.linalg.norm(r):.3g}, eps_pri = {eps_pri:.3g}, "
                f"||s||_2 = {np.linalg.norm(s):.3g}, eps_dual = {eps_dual:.3g}{objstr:s}{metstr:s}"
            )

        iter_timing.toc()
        if (
            i >= 1
            and should_check_convergence
            and np.linalg.norm(r) <= eps_pri
            and np.linalg.norm(s) <= eps_dual
        ):
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
