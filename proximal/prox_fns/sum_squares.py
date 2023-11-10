from __future__ import print_function
from .prox_fn import ProxFn
from proximal.lin_ops import CompGraph, mul_elemwise
import numpy as np
import numexpr as ne
from proximal.utils.utils import Impl, fftd, ifftd
from scipy.sparse.linalg import lsqr, LinearOperator
from proximal.halide.halide import Halide
from proximal.utils.memoized_expr import memoized_expr


class sum_squares(ProxFn):
    """The function ||x||_2^2.
    """

    def absorb_params(self):
        """Returns an equivalent sum_squares with alpha = 1.0,
           gamma = 0, and c = 0.
        """
        new_beta = np.sqrt(self.alpha * self.beta**2 + self.gamma)
        new_b = (self.alpha * self.beta * self.b - self.c / 2) / new_beta
        return sum_squares(self.lin_op, beta=new_beta, b=new_b)

    def _prox(self, rho, v, *args, **kwargs):
        """x = rho/(2+rho)*v.
        """
        ne.evaluate('v * rho / (rho + 2)', out=v, casting='unsafe')
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        if v.dtype == np.complex64 or v.dtype == np.complex128:
            return ne.evaluate('sum(real(v * conj(v)))')
        else:
            return ne.evaluate('sum(v * v)')


class weighted_sum_squares(sum_squares):
    """The function ||W.*x||_2^2.
    """

    def __init__(self, lin_op, weight, **kwargs):
        self.weight = weight
        super(weighted_sum_squares, self).__init__(lin_op, **kwargs)

    def absorb_params(self):
        """Returns an equivalent sum_squares with alpha = 1.0,
           gamma = 0, and c = 0.
        """
        new_lin_op = mul_elemwise(self.weight, self.lin_op)
        new_b = mul_elemwise(self.weight, self.b).value
        return sum_squares(new_lin_op,
                           alpha=self.alpha,
                           beta=self.beta,
                           b=new_b,
                           c=self.c,
                           gamma=self.gamma).absorb_params()

    def _prox(self, rho, v, *args, **kwargs):
        """x = (rho/weight)/(2+(rho/weight))*v.
        """
        ne.evaluate('where(w == 0, v, v * (rho / w**2) / (rho / w**2 + 2))', {
            'w': self.weight,
            'v': v,
            'rho': rho,
        },
                    out=v,
                    casting='unsafe')
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return super(weighted_sum_squares, self)._eval(self.weight * v)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.weight]


class least_squares(sum_squares):
    """The function ||K*x||_2^2.

       Here K is a computation graph (vector to vector lin op).
    """

    def __init__(self,
                 lin_op,
                 offset,
                 diag=None,
                 freq_diag=None,
                 freq_dims=None,
                 implem=Impl['numpy'],
                 **kwargs):
        self.K = CompGraph(lin_op)
        self.offset = offset
        self.diag = diag
        # TODO: freq diag is supposed to be True/False. What is going on below?
        self.freq_diag = freq_diag
        self.orig_freq_diag = freq_diag
        self.freq_dims = freq_dims
        self.orig_freq_dims = freq_dims
        # Get shape for frequency inversion var
        if self.freq_diag is not None:
            if len(self.K.orig_end.variables()) > 1:
                raise Exception(
                    "Diagonal frequency inversion supports only one var currently."
                )

            self.freq_shape = self.K.orig_end.variables()[0].shape
            self.freq_diag = np.reshape(self.freq_diag, self.freq_shape)
            if implem == Impl['halide'] and \
                    (len(self.freq_shape) == 2 or (len(self.freq_shape) == 2 and
                                                   self.freq_dims == 2)):
                # TODO: FIX REAL TO IMAG
                hsize = self.freq_shape if len(
                    self.freq_shape) == 3 else (self.freq_shape[0],
                                                self.freq_shape[1], 1)
                hsizehalide = (int((hsize[0] + 1) / 2) + 1, hsize[1], hsize[2])

                self.freq_diag = np.asfortranarray(
                    self.freq_diag[0:hsizehalide[0], ...].reshape(hsizehalide), dtype=np.complex64)

        super(least_squares, self).__init__(lin_op, implem=implem, **kwargs)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [
            self.offset, self.diag, self.orig_freq_diag, self.orig_freq_dims
        ]

    def _prox(self, rho, v, b=None, lin_solver="cg", *args, **kwargs):
        """x = argmin_x ||K*x - self.offset - b||_2^2 + (rho/2)||x-v||_2^2.
        """

        # Note(Antony): Memorized expression is implemented at
        # Halide-accelerated modules already. The following code bloat to be
        # eliminated.
        self.offset.setflags(write=False)
        if b is None:
            offset = memoized_expr("offset", {'offset':self.offset}, self.offset.shape)
            hash = self.offset.__array_interface__['data'][0]
        else:
            b.setflags(write=False)
            offset = memoized_expr("offset + b", {'offset': self.offset, 'b': b}, self.offset.shape)
            hash = b.__array_interface__['data'][0]

        return self.solve(offset,
                          rho=rho,
                          v=v,
                          lin_solver=lin_solver,
                          hash=hash,
                          *args,
                          **kwargs)

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        Kv = np.zeros(self.K.output_size)
        self.K.forward(v.ravel(), Kv)
        return super(least_squares, self)._eval(Kv - self.offset)

    def solve(self, b: memoized_expr, rho=None, v=None, lin_solver="lsqr", hash=None, *args, **kwargs):
        if self.diag is not None or self.freq_diag is not None:
            # TODO(Antony): Move the cache machanism to Halide.
            is_cache_miss: bool = (hash is None or
                                   not hasattr(self, 'Ktb') or
                                   not hasattr(self, 'b_hash') or
                                   self.b_hash is None or
                                   self.b_hash != hash)

            if is_cache_miss:
                self.Ktb = np.empty(self.K.input_size, dtype=np.float32, order='F')
                self.K.adjoint(b.evaluate(), self.Ktb)
                self.Ktb.setflags(write=False)

                self.b_hash = hash

        # KtK Operator is diagonal
        if self.diag is not None:

            u = np.empty(self.Ktb.shape, dtype=np.float32, order='F')

            if rho is None:
                ne.evaluate('Ktb / diag', {'Ktb': self.Ktb, 'diag': self.diag}, out=u, casting='unsafe')
            else:
                ne.evaluate(
                    '(Ktb + v * half_rho) / (d + half_rho)',
                    {
                        'Ktb': self.Ktb,
                        'half_rho': rho * 0.5,
                        'v': 0.0 if v is None else v,
                        'd': self.diag,
                    },
                    out=u,
                    casting='unsafe',
                )

            return u

        # KtK operator is diagonal in frequency domain.
        elif self.freq_diag is not None:
            # Frequency inversion
            if self.implementation == Impl['halide'] and \
                    (len(self.freq_shape) == 2 or
                     (len(self.freq_shape) == 2 and self.freq_dims == 2)):
                
                ftmp_halide_out = np.empty(self.freq_shape, dtype=np.float32, order='F')

                if rho is None:
                    Halide('prox_L2_ignore_offset').prox_L2_ignore_offset(
                        self.Ktb.reshape(self.freq_shape),
                        self.freq_diag,
                        ftmp_halide_out,
                    )
                else:
                    Halide('prox_L2').prox_L2(
                        self.Ktb.reshape(self.freq_shape),
                        float(rho),
                        np.reshape(v, self.freq_shape),
                        self.freq_diag,
                        ftmp_halide_out,
                    )

                return ftmp_halide_out.ravel()

            else:

                # General frequency inversion
                Ktb = fftd(np.reshape(self.Ktb, self.freq_shape), self.freq_dims)

                if rho is None:
                    Ktb /= self.freq_diag
                else:
                    Ktb *= 2.0 / rho
                    Ktb += fftd(np.reshape(v, self.freq_shape), self.freq_dims)
                    Ktb /= (2.0 / rho * self.freq_diag + 1.0)

                return (ifftd(Ktb, self.freq_dims).real).ravel()

        elif lin_solver == "lsqr":
            return self.solve_lsqr(b.evaluate(), rho, v, *args, **kwargs)
        elif lin_solver == "cg":
            return self.solve_cg(b.evaluate(), rho, v, *args, **kwargs)
        else:
            raise Exception("Unknown least squares solver.")

    def solve_lsqr(self, b, rho=None, v=None, x_init=None, options=None):
        """Solve ||K*x - b||^2_2 + (rho/2)||x-v||_2^2.
        """

        # Add additional linear terms for the rho terms
        sizev = 0
        if rho is not None:
            vf = v.flatten() * np.sqrt(rho / 2.0)
            sizeb = self.K.input_size
            sizev = np.prod(v.shape)
            b = np.hstack((b, vf))

        input_data = np.zeros(self.K.input_size)
        output_data = np.zeros(self.K.output_size + sizev)

        def matvec(x, output_data):
            if rho is None:
                # Traverse compgraph
                self.K.forward(x, output_data)
            else:
                # Compgraph and additional terms
                self.K.forward(x, output_data[0:0 + sizeb])
                np.copyto(output_data[sizeb:sizeb + sizev],
                          x * np.sqrt(rho / 2.0))

            return output_data

        def rmatvec(y, input_data):
            if rho is None:
                self.K.adjoint(y, input_data)
            else:
                self.K.adjoint(y[0:0 + sizeb], input_data)
                input_data += y[sizeb:sizeb + sizev] * np.sqrt(rho / 2.0)

            return input_data

        # Define linear operator
        def matvecComp(x):
            return matvec(x, output_data)

        def rmatvecComp(y):
            return rmatvec(y, input_data)

        K = LinearOperator((self.K.output_size + sizev, self.K.input_size),
                           matvecComp, rmatvecComp)

        # Options
        if options is None:
            # Default options
            return lsqr(K, b)[0]
        else:
            if not isinstance(options, lsqr_options):
                raise Exception("Invalid LSQR options.")
            return lsqr(K,
                        b,
                        atol=options.atol,
                        btol=options.btol,
                        show=options.show,
                        iter_lim=options.iter_lim)[0]

    def solve_cg(self, b, rho=None, v=None, x_init=None, options=None):
        """Solve ||K*x - b||^2_2 + (rho/2)||x-v||_2^2.
        """
        output_data = np.zeros(self.K.output_size)

        def KtK(x, r):
            self.K.forward(x, output_data)
            self.K.adjoint(output_data, r)
            if rho is not None:
                r += rho * x
            return r

        # Compute Ktb
        Ktb = np.zeros(self.K.input_size)
        self.K.adjoint(b, Ktb)
        if rho is not None:
            Ktb += rho * v

        # Options
        if options is None:
            # Default options
            options = cg_options()
        elif not isinstance(options, cg_options):
            raise Exception("Invalid CG options.")

        return cg(KtK, Ktb, options.tol, options.num_iters, options.verbose,
                  x_init, self.implementation)


class lsqr_options:

    def __init__(self, atol=1e-6, btol=1e-6, num_iters=50, verbose=False):
        self.atol = atol
        self.btol = btol
        self.iter_lim = num_iters
        self.show = verbose


class cg_options:

    def __init__(self, tol=1e-6, num_iters=50, verbose=False):
        self.tol = tol
        self.num_iters = num_iters
        self.verbose = verbose


def cg(KtKfun, b, tol, num_iters, verbose, x_init=None, implem=Impl['numpy']):

    # Solves KtK x = b with
    # KtKfun being a function that computes the matrix vector product KtK x

    # TODO: Fix halide later
    assert implem == Impl['numpy']

    if implem == Impl['halide']:
        output = np.array([0.0], dtype=np.float32)
        hl_norm2 = Halide('A_norm_L2.cpp',
                          generator_name="normL2_1DImg",
                          func="A_norm_L2_1D").A_norm_L2_1D
        hl_dot = Halide('A_dot_prod.cpp',
                        generator_name="dot_1DImg",
                        func="A_dot_1D").A_dot_1D

        # Temp vars
        x = np.zeros(b.shape, dtype=np.float32, order='F')
        r = np.zeros(b.shape, dtype=np.float32, order='F')
        Ap = np.zeros(b.shape, dtype=np.float32, order='F')

    else:
        # Temp vars
        x = np.zeros(b.shape)
        r = np.zeros(b.shape)
        Ap = np.zeros(b.shape)

    # Initialize x
    # Initialize everything to zero.
    if x_init is not None:
        x = x_init

    # Compute residual
    # r = b - KtKfun(x)
    KtKfun(x, r)
    r *= -1.0
    r += b

    # Do cg iterations
    if implem == Impl['halide']:
        hl_norm2(b.ravel().astype(np.float32), output)
        cg_tol = tol * output[0]
    else:
        cg_tol = tol * np.linalg.norm(b.ravel(), 2)  # Relative tol

    # CG iteration
    gamma_1 = p = None
    cg_iter = np.minimum(num_iters, np.prod(b.shape))
    for iter in range(cg_iter):
        # Check for convergence

        if implem == Impl['halide']:
            hl_norm2(r.ravel(), output)
            normr = output[0]
        else:
            normr = np.linalg.norm(r.ravel(), 2)

        # Check for convergence
        if normr <= cg_tol:
            break

        # gamma = r'*r;
        if implem == Impl['halide']:
            hl_norm2(r.ravel(), output)
            gamma = output[0]
            gamma *= gamma
        else:
            gamma = np.dot(r.ravel().T, r.ravel())

        # direction vector
        if iter > 0:
            beta = gamma / gamma_1
            p = r + beta * p
        else:
            p = r

        # Compute Ap
        KtKfun(p, Ap)

        # Cg update
        q = Ap

        # alpha = gamma / (p'*q);
        if implem == Impl['halide']:
            hl_dot(p.ravel(), q.ravel(), output)
            alpha = gamma / output[0]
        else:
            alpha = gamma / np.dot(p.ravel().T, q.ravel())

        x = x + alpha * p  # update approximation vector
        r = r - alpha * q  # compute residual

        gamma_1 = gamma

        # Iterate
        if verbose:
            print("CG Iter %03d" % iter)

    return x
