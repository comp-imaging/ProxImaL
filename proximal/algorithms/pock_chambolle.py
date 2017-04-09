from __future__ import print_function
import tempfile
import os
from proximal.lin_ops import (CompGraph, est_CompGraph_norm, Variable,
                              vstack)
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from proximal.utils.utils import graph_visualize
from proximal.utils import matlab_support
from .invert import get_least_squares_inverse, max_diag_set
import numpy as np

class PCUniformConvexGorF:
    """This class can be used to implement algorithm 2 of the Pock Chambolle 
    paper for 1/N^2 convergence rate if either Psi or Omega is uniformly convex.
    """
    def __init__(self, gamma, tau0):
        self._lastIt = 0
        self._gamma = gamma
        self._tau = tau0
        self._theta = 1.0 / np.sqrt(1 + 2*gamma*tau0)
        
    def _recalculate(self, it, L):
        self._lastIt = it
        self._tau = self._theta*self._tau
        self._theta = 1.0 / np.sqrt(1 + 2*self._gamma*self._tau)
        
    def tau(self, it, L):
        if self._lastIt != it:
            self._recalculate(it,L)
        return self._tau
            
    def sigma(self, it, L):
        if self._lastIt != it:
            self._recalculate(it,L)
        return 1.0/((L**2)*self._tau)
    
    def theta(self, it, L):
        if self._lastIt != it:
            self._recalculate(it,L)
        return self._theta

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

dsp_cnt = 0
def display_matrix(M):
    import pickle
    global dsp_cnt
    f = "/tmp/matrix%d.pickle" % dsp_cnt
    pickle.dump(M, open(f, "wb"))
    dsp_cnt = (dsp_cnt + 1) % 30
    # display the corners of M and a point in the middle
    print(f)
    idx = [0] * len(M.shape)
    while 1:
        print(("%3d "*len(idx)) % tuple(idx), "-> %+02.03e" % M.item(*idx))
        ok = False
        i = -1
        while not ok:
            if idx[i] == 0 and M.shape[i] > 1:
                ok = True
                idx[i] = M.shape[i]-1
            else:
                idx[i] = 0
            i = i-1
            if -i > len(idx):
                break
        if not ok:
            break
    idx = [s//2 for s in M.shape]
    print(("%3d "*len(idx)) % tuple(idx), "-> %+02.03e" % M.item(*idx))

def solve(psi_fns, omega_fns, tau=None, sigma=None, theta=None,
          max_iters=1000, eps_abs=1e-3, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None, conv_check=100,
          try_diagonalize=True, try_fast_norm=False, scaled=True,
          metric=None, convlog=None, verbose=0, callback=None, use_matlab=False, use_cuda=False, show_graph = False):
    
    if use_matlab:
        return solve_matlab(psi_fns, omega_fns, tau, sigma, theta,
              max_iters, eps_abs, eps_rel, x0,
              lin_solver, lin_solver_options, conv_check,
              try_diagonalize, try_fast_norm, scaled,
              metric, convlog, verbose, callback)
    if use_cuda:
        return solve_cuda(psi_fns, omega_fns, tau, sigma, theta,
              max_iters, eps_abs, eps_rel, x0,
              lin_solver, lin_solver_options, conv_check,
              try_diagonalize, try_fast_norm, scaled,
              metric, convlog, verbose, callback, show_graph)
    
    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    v = np.zeros(K.input_size)
    # Select optimal parameters if wanted
    if tau is None or sigma is None or theta is None:
        tau, sigma, theta = est_params_pc(K, tau, sigma, verbose, scaled, try_fast_norm)
    elif callable(tau) or callable(sigma) or callable(theta):
        if scaled:
            L = 1
        else:
            L = est_CompGraph_norm(K, try_fast_norm)
    
    if verbose > 0:
        print("psi_fns:", [str(f) for f in psi_fns])
        print("omega_fns:", [str(f) for f in omega_fns])
    if show_graph:
        print("Computational graph after optimizing:")
        graph_visualize(prox_fns)

    # Initialize
    x = np.zeros(K.input_size)
    y = np.zeros(K.output_size)
    xbar = np.zeros(K.input_size)
    u = np.zeros(K.output_size)
    z = np.zeros(K.output_size)

    if x0 is not None:
        x[:] = np.reshape(x0, K.input_size)
    else:
        x[:] = K.x0()

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
        objval = 0.0
        for f in prox_fns:
            evp = f.value
            #print(str(f), '->', f.value)
            objval += evp
        print("Initial objval: ", objval)
        if convlog is not None:
            convlog.record_objective(objval)
            convlog.record_timing(0.0)

    for i in range(max_iters):
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()

        if callable(sigma):
            csigma = sigma(i, L)
        else:
            csigma = sigma
        if callable(tau):
            ctau = tau(i, L)
        else:
            ctau = tau
        if callable(theta):
            ctheta = theta(i, L)
        else:
            ctheta = theta
            
        # Keep track of previous iterates
        np.copyto(prev_x, x)
        np.copyto(prev_z, z)
        np.copyto(prev_u, u)
        np.copyto(prev_Kx, Kx)

        # Compute z
        K.forward(xbar, Kxbar)
        z = y + csigma * Kxbar

        # Update y.
        #print("--------------- psi_fns --------------------")
        offset = 0
        for fn in psi_fns:
            slc = slice(offset, offset + fn.lin_op.size, None)
            z_slc = np.reshape(z[slc], fn.lin_op.shape)

            #print("prox", offset)
            #display_matrix(z_slc / csigma)

            # Moreau identity: apply and time prox.
            prox_log[fn].tic()
            y[slc] = (z_slc - csigma * fn.prox(csigma, z_slc / csigma, i)).flatten()
            prox_log[fn].toc()
            #print("y_slc", fn)
            #display_matrix(y[slc])
            offset += fn.lin_op.size
        y[offset:] = 0

        # Update x
        K.adjoint(y, KTy)
        x -= ctau * KTy

        if len(omega_fns) > 0:
            xtmp = np.reshape(x, omega_fns[0].lin_op.shape)
            x[:] = omega_fns[0].prox(1.0 / ctau, xtmp, x_init=prev_x,
                                     lin_solver=lin_solver, options=lin_solver_options).flatten()

        # Update xbar
        np.copyto(xbar, x)
        xbar += ctheta * (x - prev_x)

        # Convergence log
        if convlog is not None:
            convlog.toc()
            K.update_vars(x)
            objval = list([fn.value for fn in prox_fns])
            objval = sum(objval)
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
        u = 1.0 / csigma * y + ctheta * (Kx - prev_Kx)
        z = prev_u + prev_Kx - 1.0 / csigma * y

        # Iteration order is different than
        # lin-admm (--> start checking at iteration 1)
        if i % conv_check == 0:

            # Check convergence
            r = prev_Kx - z
            K.adjoint(csigma * (z - prev_z), s)
            eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * \
                max([np.linalg.norm(prev_Kx), np.linalg.norm(z)])

            K.adjoint(u, KTu)
            eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / csigma

            if not callback is None or verbose == 2:
                K.update_vars(x)
            if not callback is None:
                callback(x)
            
            # Progress
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    ov = list([fn.value for fn in prox_fns])
                    objval = sum(ov)
                    objstr = ", obj_val = %02.03e [%s] " % (objval, ", ".join("%02.03e" % x for x in ov))

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
    if not callback is None:
        callback(x)
    # Return optimal value.
    return sum([fn.value for fn in prox_fns])

def solve_cuda(psi_fns, omega_fns, tau=None, sigma=None, theta=None,
               max_iters=1000, eps_abs=1e-3, eps_rel=1e-3, x0=None,
               lin_solver="cg", lin_solver_options=None, conv_check=100,
               try_diagonalize=True, try_fast_norm=False, scaled=True,
               metric=None, convlog=None, verbose=0, callback=None, show_graph=False):
    import pycuda.gpuarray as gpuarray
    
    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    v = np.zeros(K.input_size)
    # Select optimal parameters if wanted
    if tau is None or sigma is None or theta is None:
        tau, sigma, theta = est_params_pc(K, tau, sigma, verbose, scaled, try_fast_norm)
    elif callable(tau) or callable(sigma) or callable(theta):
        if scaled:
            L = 1
        else:
            L = est_CompGraph_norm(K, try_fast_norm)
        
    if verbose > 0:
        print("psi_fns:", [str(f) for f in psi_fns])
        print("omega_fns:", [str(f) for f in omega_fns])

    if show_graph:
        print("Computational graph after optimizing:")
        graph_visualize(prox_fns)
        
        
    # Initialize
    x = gpuarray.to_gpu(np.zeros(K.input_size, dtype=np.float32))
    y = gpuarray.to_gpu(np.zeros(K.output_size, dtype=np.float32))
    xbar = gpuarray.to_gpu(np.zeros(K.input_size, dtype=np.float32))
    u = gpuarray.to_gpu(np.zeros(K.output_size, dtype=np.float32))
    z = gpuarray.to_gpu(np.zeros(K.output_size, dtype=np.float32))

    if x0 is not None:
        x[:] = np.reshape(x0, K.input_size)
    else:
        x[:] = K.x0().astype(np.float32)

    K.forward_cuda(x, y)
    xbar[:] = x

    # Buffers.
    Kxbar = gpuarray.to_gpu(np.zeros(K.output_size, dtype=np.float32))
    Kx = gpuarray.to_gpu(np.zeros(K.output_size, dtype=np.float32))
    KTy = gpuarray.to_gpu(np.zeros(K.input_size, dtype=np.float32))
    KTu = gpuarray.to_gpu(np.zeros(K.input_size, dtype=np.float32))
    s = gpuarray.to_gpu(np.zeros(K.input_size, dtype=np.float32))

    prev_x = x.copy()
    prev_Kx = Kx.copy()
    prev_z = z.copy()
    prev_u = u.copy()

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    prox_log_tot = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("PC iteration")
    add_timing = TimingsLog(["copyprev", "calcz", "calcx", "omega_fn", "xbar", "conv_check"])

    # Convergence log for initial iterate
    if 1: #convlog is not None:
        K.update_vars(x.get())
        objval = 0.0
        for f in prox_fns:
            evp = f.value
            #print(str(f), '->', f.value)
            objval += evp
        print("Initial objval: ", objval)
        if convlog is not None:
            convlog.record_objective(objval)
            convlog.record_timing(0.0)

    for i in range(max_iters):
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()
        add_timing["copyprev"].tic()
        # Keep track of previous iterates
        prev_x[:] = x
        prev_z[:] = z
        prev_u[:] = u
        prev_Kx[:] = Kx
        add_timing["copyprev"].toc()

        if callable(sigma):
            csigma = np.float32(sigma(i, L))
        else:
            csigma = np.float32(sigma)
        if callable(tau):
            ctau = np.float32(tau(i, L))
        else:
            ctau = np.float32(tau)
        if callable(theta):
            ctheta = np.float32(theta(i, L))
        else:
            ctheta = np.float32(theta)
            
        # Compute z
        add_timing["calcz"].tic()
        K.forward_cuda(xbar, Kxbar)
        z = y + csigma * Kxbar
        add_timing["calcz"].toc()

        # Update y.
        #print("--------------- psi_fns --------------------")
        offset = 0
        for fn in psi_fns:
            prox_log_tot[fn].tic()
            slc = slice(offset, offset + fn.lin_op.size, None)
            z_slc = gpuarray.reshape(z[slc], fn.lin_op.shape)

            #print("prox", offset)
            #display_matrix((z_slc / sigma).get())

            # Moreau identity: apply and time prox.
            prox_log[fn].tic()
            y[slc] = gpuarray.reshape((z_slc - csigma * fn.prox_cuda(csigma, z_slc / csigma, i)), int(np.prod(z_slc.shape)) )
            prox_log[fn].toc()
            #print("y_slc", fn)
            #display_matrix(y[slc].get())
            offset += fn.lin_op.size
            prox_log_tot[fn].toc()
        #y[offset:] = 0

        # Update x
        add_timing["calcx"].tic()
        K.adjoint_cuda(y, KTy)
        #print("KTy")
        #display_matrix(KTy.get())
        x -= ctau * KTy
        add_timing["calcx"].toc()

        add_timing["omega_fn"].tic()
        if len(omega_fns) > 0:
            xtmp = np.reshape(x, omega_fns[0].lin_op.shape)
            x[:] = omega_fns[0].prox_cuda(np.float32(1.0) / ctau, xtmp, x_init=prev_x,
                                         lin_solver=lin_solver, options=lin_solver_options).flatten()
        add_timing["omega_fn"].toc()

        add_timing["xbar"].tic()
        # Update xbar
        xbar[:] = x
        #np.copyto(xbar, x)
        xbar += ctheta * (x - prev_x)
        add_timing["xbar"].toc()

        # Convergence log
        if convlog is not None:
            convlog.toc()
            K.update_vars(x.get())
            objval = list([fn.value for fn in prox_fns])
            objval = sum(objval)
            convlog.record_objective(objval)

        """ Old convergence check
        #Very basic convergence check.
        r_x = np.linalg.norm(x - prev_x)
        r_xbar = np.linalg.norm(xbar - prev_xbar)
        r_ybar = np.linalg.norm(y - prev_y)
        error = r_x + r_xbar + r_ybar
        """

        if i % conv_check in [0, conv_check-1]:
            add_timing["conv_check"].tic()
            # Residual based convergence check
            K.forward_cuda(x, Kx)
            u = np.float32(1.0) / csigma * y + ctheta * (Kx - prev_Kx)
            z = prev_u + prev_Kx - np.float32(1.0) / csigma * y
            add_timing["conv_check"].toc()

        # Iteration order is different than
        # lin-admm (--> start checking at iteration 1)
        if i % conv_check == 0:

            # Check convergence
            r = prev_Kx - z
            K.adjoint_cuda(csigma * (z - prev_z), s)
            eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * \
                max([np.linalg.norm(prev_Kx.get()), np.linalg.norm(z.get())])

            K.adjoint_cuda(u, KTu)
            eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(KTu.get()) / csigma

            if not callback is None or verbose == 2:
                K.update_vars(x.get())
            if not callback is None:
                callback(x.get())
            
            # Progress
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    ov = list([fn.value for fn in prox_fns])
                    objval = sum(ov)
                    objstr = ", obj_val = %02.03e [%s] " % (objval, ", ".join("%02.03e" % x for x in ov))

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
                    % (i, np.linalg.norm(r.get()), eps_pri, np.linalg.norm(s.get()), eps_dual, objstr, metstr)
                )

            iter_timing.toc()
            if np.linalg.norm(r.get()) <= eps_pri and np.linalg.norm(s.get()) <= eps_dual:
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
        print("additional PC profiling:")
        print(add_timing)
        print("prox funcs total:")
        print(prox_log_tot)
        print("prox funcs:")
        print(prox_log)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    K.update_vars(x.get())
    if not callback is None:
        callback(x.get())
    # Return optimal value.
    return sum([fn.value for fn in prox_fns])


def solve_matlab(psi_fns, omega_fns, tau=None, sigma=None, theta=None,
          max_iters=1000, eps_abs=1e-3, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None, conv_check=100,
          try_diagonalize=True, try_fast_norm=False, scaled=True,
          metric=None, convlog=None, verbose=0, callback=None):
    eng = matlab_support.engine()
    eng.run("clear all")
    
    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    mlclass = matlab_support.MatlabClass("prox_pc")
    K = CompGraph(stacked_ops, implem='matlab')
    K.genmatlab(mlclass)
    
    v = np.zeros(K.input_size)
    # Select optimal parameters if wanted
    if tau is None or sigma is None or theta is None:
        tau, sigma, theta = est_params_pc(K, tau, sigma, verbose, scaled, try_fast_norm)
        
    if verbose > 0:
        print("psi_fns:", [str(f) for f in psi_fns])
        print("omega_fns:", [str(f) for f in omega_fns])

    # generate code for psi_fns and omega_fns
    for i,f in enumerate(prox_fns):
        f.genmatlab('proxfn_%d' % i, mlclass)

    # Initialize
    x = np.zeros(K.input_size, np.float32)

    if x0 is not None:
        x[:] = np.reshape(x0, K.input_size, order='F')
    else:
        x[:] = K.x0(order='F')
    
    Kforward = "obj." + K.matlab_forward_script;
    Kadjoint = "obj." + K.matlab_adjoint_script;
    
    sigmaf = '' if sigma == 1 else '%(sigma)f *' % locals()
    
    pciter = """
function [obj,x,y,xbar,u,z,Kxbar,Kx,KTy,KTu,s,prev_x,prev_Kx,prev_z,prev_u] = pciter(obj, x,y,xbar,u,z,Kxbar,Kx,KTy,KTu,s,prev_x,prev_Kx,prev_z,prev_u)
    prev_x = x;
    prev_z = z;
    prev_u = u;
    prev_Kx = Kx;
    
    [obj, Kxbar] = %(Kforward)s(xbar);
    z = y + %(sigmaf)s Kxbar;

    %%display('--------------- psi_fns --------------------')
""" % locals()
    offset = 0 
    for fn in psi_fns:
        linop_size = fn.lin_op.size
        slc = "(%(offset)d+1):(%(offset)d + %(linop_size)d)" % locals()
        
        shape = list(fn.lin_op.shape)
        numdims = len(shape)
        script = fn.matlab_prox_script
        
        if sigma == 1:
            pciter += """
        z_slc = reshape( z(%(slc)s), %(shape)s );
        %%display('z_slc %(offset)d');
        %%obj.display_matrix(z_slc)
        prox_out = obj.%(script)s(z_slc, 1);
        y(%(slc)s) = (z_slc - %(sigma)f * prox_out);

""" % locals()
            
        else:
            pciter += """
        z_slc = reshape( z(%(slc)s), %(shape)s );
        rho = %(sigma)f;
        prox_out = obj.%(script)s(z_slc ./ rho, rho);
        y(%(slc)s) = (z_slc - %(sigma)f * prox_out);

""" % locals()
        offset += fn.lin_op.size
    
    tauf = '' if tau == 1 else "%(tau)f *" % locals()
    pciter += """
    y(%(offset)d + 1:end) = 0;
    
    [obj, KTy] = %(Kadjoint)s(y);
    
    x = x - %(tauf)s KTy;
""" % locals()
    if len(omega_fns) > 0:
        shape = list(omega_fns[0].lin_op.shape)
        numdims = len(shape)
        script = omega_fns[0].matlab_prox_script
        pciter += """
        prox_in = reshape(x, %(shape)s);
        rho = 1./%(tau)f;
        x(1:end) = script(prox_in, rho);
""" % locals()

    thetaf = '' if theta == 1 else '%(theta)f *' % locals()

    pciter += """
    xbar = x + %(thetaf)s (x - prev_x);
end""" % locals()
    
    mlclass.add_method(pciter)
    mlclass.add_method("""
function display_matrix(obj, M)
    % display the corners of M and a point in the middle
    idx = num2cell(ones(length(size(M)), 1));
    while 1
        display([sprintf('%3d ', cell2mat(idx)) sprintf(' -> %+02.03e', M(sub2ind(size(M), idx{:})))]);
        ok = 0;
        i = length(idx);
        while ~ok
            if idx{i} == 1 && size(M,i) > 1
                ok = 1;
                idx{i} = size(M,i);
            else
                idx{i} = 1;
            end
            i = i-1;
            if i < 1
                break
            end
        end
        if ~ok
            break
        end
    end
    idx = num2cell(floor(size(M)/2) + 1);
    display([sprintf('%3d ', cell2mat(idx)) sprintf(' -> %+02.03e', M(sub2ind(size(M), idx{:})))]);
end
""")
    mlclass.generate()

    matlab_support.put_array("x0", x)
    eng.run("x = gpuArray(x0);")

    #K.forward(x, y)
    eng.run("[" + mlclass.instancename + ", y] = " + mlclass.instancename + "." + K.matlab_forward_script + "(x);")
    
    # Buffers.
    #xbar = np.zeros(K.input_size)
    #u = np.zeros(K.output_size)
    #z = np.zeros(K.output_size)
    #Kxbar = np.zeros(K.output_size)
    #Kx = np.zeros(K.output_size)
    #KTy = np.zeros(K.input_size)
    #KTu = np.zeros(K.input_size)
    #s = np.zeros(K.input_size)
    eng.run("xbar  = zeros(%d, 1, 'single', 'gpuArray');" % K.input_size)
    eng.run("u     = zeros(%d, 1, 'single', 'gpuArray');" % K.output_size)
    eng.run("z     = zeros(%d, 1, 'single', 'gpuArray');" % K.output_size)    
    eng.run("Kxbar = zeros(%d, 1, 'single', 'gpuArray');" % K.output_size)
    eng.run("Kx    = zeros(%d, 1, 'single', 'gpuArray');" % K.output_size)
    eng.run("KTy   = zeros(%d, 1, 'single', 'gpuArray');" % K.input_size)
    eng.run("KTu   = zeros(%d, 1, 'single', 'gpuArray');" % K.input_size)
    eng.run("s     = zeros(%d, 1, 'single', 'gpuArray');" % K.input_size)

    #xbar[:] = x
    eng.run("xbar = x;")

    #prev_x = x.copy()
    #prev_Kx = Kx.copy()
    #prev_z = z.copy()
    #prev_u = u.copy()
    eng.run("prev_x = x; prev_Kx = Kx; prev_z = z; prev_u = u;")

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("PC iteration")

    # Convergence log for initial iterate
    if 1: # convlog is not None:
        K.update_vars_matlab("x")
        objval = 0
        for f in prox_fns:
            n = K.get_node_output_name(f.lin_op)
            n = n.replace("obj.", mlclass.instancename + ".")
            eng.run("evp = " + mlclass.instancename + "." + f.matlab_eval_script + "(%(n)s);"%locals())
            evp = matlab_support.get_array('evp')
            #print(str(f), '->', evp)
            objval += evp
        if convlog is not None:
            convlog.record_objective(objval)
            convlog.record_timing(0.0)
        #print(objval)
    
    matlab_pc_script = "[" + mlclass.instancename + ", x,y,xbar,u,z,Kxbar,Kx,KTy,KTu,s,prev_x,prev_Kx,prev_z,prev_u] = " + mlclass.instancename + ".pciter(x,y,xbar,u,z,Kxbar,Kx,KTy,KTu,s,prev_x,prev_Kx,prev_z,prev_u);"

    for i in range(max_iters):
        iter_timing.tic()
        eng.run(matlab_pc_script + ';')

        if i == max_iters - 1 or i % conv_check == 0:
            K.update_vars_matlab("x")        
            ov = []
            for f in prox_fns:
                n = K.get_node_output_name(f.lin_op)
                n = n.replace("obj.", mlclass.instancename + ".")                
                eng.run("evp = " + mlclass.instancename + "." + f.matlab_eval_script + "(%(n)s);"%locals() )
                evp = matlab_support.get_array('evp')
                ov += [evp]
            objval = sum(ov)
            if convlog is not None:
                convlog.record_objective(objval)
                convlog.record_timing(0.0)
            if verbose > 0:
                obj_str = ''
                if verbose == 2:
                    objstr = ", obj_val = %02.03e [%s] " % (objval, ", ".join("%02.03e" % x for x in ov))
                else:
                    objstr = ""

                # Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format(metric.message(v))
                print(
                    "iter %d: ||r||_2 = %.3f, eps_pri = %.3f, ||s||_2 = %.3f, eps_dual = %.3f%s%s"
                    % (i, 0.0, 0.0, 0.0, 0.0, objstr, metstr)
                )
                
        iter_timing.toc()

    # Print out timings info.
    if verbose > 0:
        print(iter_timing)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    eng.run('xcpu = gather(x);')
    x = matlab_support.get_array('xcpu')
    K.update_vars(x, order='F')
    if not callback is None:
        callback(x)
    # Return optimal value.
    res = sum([fn.value for fn in prox_fns])
    return res

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
