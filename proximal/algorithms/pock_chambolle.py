from __future__ import print_function
from proximal.lin_ops import (CompGraph, est_CompGraph_norm, Variable,
                              vstack)
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from .invert import get_least_squares_inverse, max_diag_set
import numpy as np


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


def solve(psi_fns, omega_fns, tau=None, sigma=None, theta=None,
          max_iters=1000, eps_abs=1e-3, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, try_fast_norm=False, scaled=True,
          metric=None, convlog=None, verbose=0):
    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    v = np.zeros(K.input_size)
    # Select optimal parameters if wanted
    if tau is None or sigma is None or theta is None:
        tau, sigma, theta = est_params_pc(K, tau, sigma, verbose, scaled, try_fast_norm)

    # Initialize
    x = np.zeros(K.input_size)
    y = np.zeros(K.output_size)
    xbar = np.zeros(K.input_size)
    u = np.zeros(K.output_size)
    z = np.zeros(K.output_size)

    if x0 is not None:
        x[:] = np.reshape(x0, K.input_size)
        K.forward(x, y)
        xbar[:] = x

    # Buffers.
    Kxbar = np.zeros(K.output_size)
    Kx = np.zeros(K.output_size)
    KTy = np.zeros(K.input_size)
    KTu = np.zeros(K.input_size)
    s = np.zeros(K.input_size)

    prev_x = x.copy()
    prev_Kx = Kx.copy()
    prev_z = z.copy()
    prev_u = u.copy()

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("PC iteration")

    # Convergence log for initial iterate
    if convlog is not None:
        K.update_vars(x)
        objval = sum([fn.value for fn in prox_fns])
        convlog.record_objective(objval)
        convlog.record_timing(0.0)

    for i in range(max_iters):
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()

        # Keep track of previous iterates
        np.copyto(prev_x, x)
        np.copyto(prev_z, z)
        np.copyto(prev_u, u)
        np.copyto(prev_Kx, Kx)

        # Compute z
        K.forward(xbar, Kxbar)
        z = y + sigma * Kxbar

        # Update y.
        offset = 0
        for fn in psi_fns:
            slc = slice(offset, offset + fn.lin_op.size, None)
            z_slc = np.reshape(z[slc], fn.lin_op.shape)

            # Moreau identity: apply and time prox.
            prox_log[fn].tic()
            y[slc] = (z_slc - sigma * fn.prox(sigma, z_slc / sigma, i)).flatten()
            prox_log[fn].toc()
            offset += fn.lin_op.size
        y[offset:] = 0

        # Update x
        K.adjoint(y, KTy)
        x -= tau * KTy

        if len(omega_fns) > 0:
            xtmp = np.reshape(x, omega_fns[0].lin_op.shape)
            x[:] = omega_fns[0].prox(1.0 / tau, xtmp, x_init=prev_x,
                                     lin_solver=lin_solver, options=lin_solver_options).flatten()

        # Update xbar
        np.copyto(xbar, x)
        xbar += theta * (x - prev_x)

        # Convergence log
        if convlog is not None:
            convlog.toc()
            K.update_vars(x)
            objval = sum([fn.value for fn in prox_fns])
            convlog.record_objective(objval)

        """ Old convergence check
        #Very basic convergence check.
        r_x = np.linalg.norm(x - prev_x)
        r_xbar = np.linalg.norm(xbar - prev_xbar)
        r_ybar = np.linalg.norm(y - prev_y)
        error = r_x + r_xbar + r_ybar
        """

        # Residual based convergence check
        K.forward(x, Kx)
        u = 1.0 / sigma * y + theta * (Kx - prev_Kx)
        z = prev_u + prev_Kx - 1.0 / sigma * y

        # Iteration order is different than
        # lin-admm (--> start checking at iteration 1)
        if i > 0:

            # Check convergence
            r = prev_Kx - z
            K.adjoint(sigma * (z - prev_z), s)
            eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * \
                max([np.linalg.norm(prev_Kx), np.linalg.norm(z)])

            K.adjoint(u, KTu)
            eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / sigma

            # Progress
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    K.update_vars(x)
                    objstr = ", obj_val = %02.03e" % sum([fn.value for fn in prox_fns])

                """ Old convergence check
                #Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format( metric.message(x.copy()) )
                print "iter [%04d]:" \
                      "||x - x_prev||_2 = %02.02e " \
                      "||xbar - xbar_prev||_2 = %02.02e " \
                      "||y - y_prev||_2 = %02.02e " \
                      "SUM = %02.02e (eps=%02.03e)%s%s" \
                        % (i, r_x, r_xbar, r_ybar, error, eps, objstr, metstr)
                """

                # Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format(metric.message(v))
                print(
                    "iter %d: ||r||_2 = %.3f, eps_pri = %.3f, ||s||_2 = %.3f, eps_dual = %.3f%s%s"
                    % (i, np.linalg.norm(r), eps_pri, np.linalg.norm(s), eps_dual, objstr, metstr)
                )

            iter_timing.toc()
            if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
                break

        else:
            iter_timing.toc()

        """ Old convergence check
        if error <= eps:
            break
        """

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
    K.update_vars(x)

    # Return optimal value.
    return sum([fn.value for fn in prox_fns])


def est_params_pc(K, tau=None, sigma=None, verbose=True, scaled=False, try_fast_norm=False):

    # Select theta constant and sigma larger 0
    theta = 1.0
    sigma = 1.0 if sigma is None else sigma

    # Estimate Lipschitz constant and comput tau
    if scaled:
        L = 1
    else:
        L = est_CompGraph_norm(K, try_fast_norm)
    tau = 1.0 / (sigma * L**2)

    if verbose:
        print("Estimated params [sigma = %3.3f | tau = %3.3f | theta = %3.3f | L_est = %3.4f]"
              % (sigma, tau, theta, L))

    return tau, sigma, theta
