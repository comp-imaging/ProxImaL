import sys
import numpy as np

try:
    from pycuda import gpuarray
    from proximal.utils.cuda_codegen import PyCudaAdapter
except ImportError:
    pass

import pytest

from proximal.tests.base_test import BaseTest
import proximal as px
from proximal.lin_ops.vstack import vstack
from proximal.algorithms import admm, pc, hqs, ladmm, absorb_offset
import cvxpy as cvx


class TestAlgs(BaseTest):

    def test_admm_single_prox_fn(self):
        """Test ADMM algorithm, single prox function only.
        """
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5)) * 1.

        prox_fns = [px.sum_squares(X, b=B)]
        sltn = admm.solve(prox_fns, [], 1.0, eps_abs=1e-4, eps_rel=1e-4)
        self.assertItemsAlmostEqual(X.value, B, eps=2e-2)
        self.assertAlmostEqual(sltn, 0)

        prox_fns = [px.norm1(X, b=B, beta=2)]
        sltn = admm.solve(prox_fns, [], 1.0)
        self.assertItemsAlmostEqual(X.value, B / 2., eps=2e-2)
        self.assertAlmostEqual(sltn, 0)

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_admm_two_prox_fn(self):
        """Test ADMM algorithm, two prox functions.
        """
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5)) * 1.

        prox_fns = [px.norm1(X), px.sum_squares(X, b=B)]
        sltn = admm.solve(prox_fns, [], 1.0, eps_rel=1e-5, eps_abs=1e-5)

        cvx_X = cvx.Variable((10, 5))
        cost = cvx.sum_squares(cvx_X - B) + cvx.norm(cvx_X, 1)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-1)
        self.assertAlmostEqual(sltn, prob.value)

        psi_fns, omega_fns = admm.partition(prox_fns)
        sltn = admm.solve(psi_fns, omega_fns, 1.0, eps_rel=1e-5, eps_abs=1e-5)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value)

        prox_fns = [px.norm1(X)]
        quad_funcs = [px.sum_squares(X, b=B)]
        sltn = admm.solve(prox_fns, quad_funcs, 1.0, eps_rel=1e-5, eps_abs=1e-5)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value)

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_admm_extra_param(self):
        ''' With parameters for px.sum_squares '''
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5)) * 1.

        prox_fns = [px.norm1(X)]
        quad_funcs = [px.sum_squares(X, b=B, alpha=0.1, beta=2., gamma=1, c=B)]
        sltn = admm.solve(prox_fns, quad_funcs, 1.0, eps_rel=1e-5, eps_abs=1e-5)

        cvx_X = cvx.Variable((10, 5))
        cost = 0.1 * cvx.sum_squares(2 * cvx_X - B) + cvx.sum_squares(cvx_X) + \
            cvx.norm(cvx_X, 1) + cvx.trace(B.T @ cvx_X)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value, eps=1e-3)

        prox_fns = [px.norm1(X)]
        quad_funcs = [px.sum_squares(X - B, alpha=0.1, beta=2., gamma=1, c=B)]
        quad_funcs[0] = absorb_offset(quad_funcs[0])
        sltn = admm.solve(prox_fns, quad_funcs, 1.0, eps_rel=1e-5, eps_abs=1e-5)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value, eps=1e-3)

        prox_fns = [px.norm1(X)]
        cvx_X = cvx.Variable((10, 5))
        # With linear operators.
        kernel = np.array([1, 2, 3])
        x = px.Variable(3)
        b = np.array([-41, 413, 2])
        prox_fns = [px.nonneg(x), px.sum_squares(px.conv(kernel, x), b=b)]
        sltn = admm.solve(prox_fns, [], 1.0, eps_abs=1e-5, eps_rel=1e-5)

        kernel_mat = np.matrix("2 1 3; 3 2 1; 1 3 2")
        cvx_X = cvx.Variable(3)
        cost = cvx.norm(kernel_mat * cvx_X - b)
        prob = cvx.Problem(cvx.Minimize(cost), [cvx_X >= 0])
        prob.solve()
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(np.sqrt(sltn), prob.value, eps=2e-2)

        prox_fns = [px.nonneg(x)]
        quad_funcs = [px.sum_squares(px.conv(kernel, x), b=b)]
        sltn = admm.solve(prox_fns, quad_funcs, 1.0, eps_abs=1e-5, eps_rel=1e-5)
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(np.sqrt(sltn), prob.value, eps=2e-2)

    @pytest.mark.skipif('pycuda' not in sys.modules,
                        reason="requires the pycuda library")
    def test_pock_chambolle_cuda(self):
        self._test_pock_chambolle('pycuda')

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_pock_chambolle_numpy(self):
        self._test_pock_chambolle('numpy')

    def _test_pock_chambolle(self, impl):
        """Test pock chambolle algorithm.
        """
        #print()
        #print("----------------------",impl,"-------------------------")
        if impl == 'pycuda':
            kw = {'adapter': PyCudaAdapter()}
        else:
            kw = {}
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5))
        prox_fns = [px.sum_squares(X, b=B)]
        sltn = pc.solve(prox_fns, [],
                        1.0,
                        1.0,
                        1.0,
                        eps_rel=1e-5,
                        eps_abs=1e-5,
                        **kw)
        self.assertItemsAlmostEqual(X.value, B, eps=2e-2)
        self.assertAlmostEqual(sltn, 0)

        prox_fns = [px.norm1(X, b=B, beta=2)]
        sltn = pc.solve(prox_fns, [],
                        1.0,
                        1.0,
                        1.0,
                        eps_rel=1e-5,
                        eps_abs=1e-5,
                        **kw)
        self.assertItemsAlmostEqual(X.value, B / 2., eps=2e-2)
        self.assertAlmostEqual(sltn, 0, eps=2e-2)

        prox_fns = [px.norm1(X), px.sum_squares(X, b=B)]
        #print("----------------------------------------------------")
        #print("----------------------------------------------------")
        sltn = pc.solve(prox_fns, [],
                        0.5,
                        1.0,
                        1.0,
                        eps_rel=1e-5,
                        eps_abs=1e-5,
                        conv_check=1,
                        **kw)

        cvx_X = cvx.Variable((10, 5))
        cost = cvx.sum_squares(cvx_X - B) + cvx.norm(cvx_X, 1)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value)

        psi_fns, omega_fns = pc.partition(prox_fns)
        sltn = pc.solve(psi_fns,
                        omega_fns,
                        0.5,
                        1.0,
                        1.0,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        **kw)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value)

        # With linear operators.
        kernel = np.array([1, 2, 3])
        kernel_mat = np.matrix("2 1 3; 3 2 1; 1 3 2")
        x = px.Variable(3)
        b = np.array([-41, 413, 2])
        prox_fns = [px.nonneg(x), px.sum_squares(px.conv(kernel, x), b=b)]
        sltn = pc.solve(prox_fns, [],
                        0.1,
                        0.1,
                        1.0,
                        max_iters=3000,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        **kw)
        cvx_X = cvx.Variable(3)
        cost = cvx.norm(kernel_mat * cvx_X - b)
        prob = cvx.Problem(cvx.Minimize(cost), [cvx_X >= 0])
        prob.solve()
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=2e-2)

        psi_fns, omega_fns = pc.partition(prox_fns)
        sltn = pc.solve(psi_fns,
                        omega_fns,
                        0.1,
                        0.1,
                        1.0,
                        max_iters=3000,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        **kw)
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=2e-2)

        # TODO
        # Multiple variables.
        x = px.Variable(1)
        y = px.Variable(1)
        prox_fns = [
            px.nonneg(x),
            px.sum_squares(vstack([x, y]), b=np.arange(2))
        ]
        sltn = pc.solve(prox_fns, [prox_fns[-1]],
                        0.1,
                        0.1,
                        1.0,
                        max_iters=3000,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        try_diagonalize=False)
        self.assertItemsAlmostEqual(x.value, [0])
        self.assertItemsAlmostEqual(y.value, [1])

        sltn = pc.solve(prox_fns, [prox_fns[-1]],
                        0.1,
                        0.1,
                        1.0,
                        max_iters=3000,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        try_diagonalize=True)
        self.assertItemsAlmostEqual(x.value, [0])
        self.assertItemsAlmostEqual(y.value, [1])

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_half_quadratic_splitting_single_prox_fn(self):
        """Test half quadratic splitting.
        """
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5))
        prox_fns = [px.sum_squares(X, b=B)]
        sltn = hqs.solve(prox_fns, [],
                         eps_rel=1e-4,
                         max_iters=100,
                         max_inner_iters=50)
        self.assertItemsAlmostEqual(X.value, B, eps=2e-2)
        self.assertAlmostEqual(sltn, 0)

        prox_fns = [px.norm1(X, b=B, beta=2)]
        sltn = hqs.solve(prox_fns, [],
                         eps_rel=1e-4,
                         max_iters=100,
                         max_inner_iters=50)
        self.assertItemsAlmostEqual(X.value, B / 2., eps=2e-2)
        self.assertAlmostEqual(sltn, 0)

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_half_quadratic_splitting_two_prox_fn(self):
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5))

        prox_fns = [px.norm1(X), px.sum_squares(X, b=B)]
        sltn = hqs.solve(prox_fns, [],
                         eps_rel=1e-7,
                         rho_0=1.0,
                         rho_scale=np.sqrt(2.0) * 2.0,
                         rho_max=2**16,
                         max_iters=20,
                         max_inner_iters=500)

        cvx_X = cvx.Variable((10, 5))
        cost = cvx.sum_squares(cvx_X - B) + cvx.norm(cvx_X, 1)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value, eps=1e-3)

        psi_fns, omega_fns = hqs.partition(prox_fns)
        sltn = hqs.solve(psi_fns,
                         omega_fns,
                         eps_rel=1e-7,
                         rho_0=1.0,
                         rho_scale=np.sqrt(2.0) * 2.0,
                         rho_max=2**16,
                         max_iters=20,
                         max_inner_iters=500)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value, eps=1e-3)

        # With linear operators.
        kernel = np.array([1, 2, 3])
        kernel_mat = np.matrix("2 1 3; 3 2 1; 1 3 2")
        x = px.Variable(3)
        b = np.array([-41, 413, 2])
        prox_fns = [px.nonneg(x), px.sum_squares(px.conv(kernel, x), b=b)]
        hqs.solve(prox_fns, [],
                  eps_rel=1e-9,
                  rho_0=4,
                  rho_scale=np.sqrt(2.0) * 1.0,
                  rho_max=2**16,
                  max_iters=30,
                  max_inner_iters=500)

        cvx_X = cvx.Variable(3)
        cost = cvx.norm(kernel_mat * cvx_X - b)
        prob = cvx.Problem(cvx.Minimize(cost), [cvx_X >= 0])
        prob.solve()
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=1e-0)

        psi_fns, omega_fns = hqs.partition(prox_fns)
        hqs.solve(psi_fns,
                  omega_fns,
                  eps_rel=1e-9,
                  rho_0=4,
                  rho_scale=np.sqrt(2.0) * 1.0,
                  rho_max=2**16,
                  max_iters=30,
                  max_inner_iters=500)
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=1e-0)

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_lin_admm_single_prox_fn(self):
        """Test linearized admm. algorithm.
        """
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5))
        prox_fns = [px.sum_squares(X, b=B)]
        sltn = ladmm.solve(prox_fns, [],
                           0.1,
                           max_iters=500,
                           eps_rel=1e-5,
                           eps_abs=1e-5)
        self.assertItemsAlmostEqual(X.value, B, eps=2e-2)
        self.assertAlmostEqual(sltn, 0)

        prox_fns = [px.norm1(X, b=B, beta=2)]
        sltn = ladmm.solve(prox_fns, [],
                           0.1,
                           max_iters=500,
                           eps_rel=1e-5,
                           eps_abs=1e-5)
        self.assertItemsAlmostEqual(X.value, B / 2., eps=2e-2)
        self.assertAlmostEqual(sltn, 0, eps=2e-2)

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_lin_admm_two_prox_fn(self):
        X = px.Variable((10, 5))
        B = np.reshape(np.arange(50), (10, 5))
        prox_fns = [px.norm1(X), px.sum_squares(X, b=B)]
        sltn = ladmm.solve(prox_fns, [],
                           0.1,
                           max_iters=500,
                           eps_rel=1e-5,
                           eps_abs=1e-5)

        cvx_X = cvx.Variable((10, 5))
        cost = cvx.sum_squares(cvx_X - B) + cvx.norm(cvx_X, 1)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value)

        psi_fns, omega_fns = ladmm.partition(prox_fns)
        sltn = ladmm.solve(psi_fns,
                           omega_fns,
                           0.1,
                           max_iters=500,
                           eps_rel=1e-5,
                           eps_abs=1e-5)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, eps=2e-2)
        self.assertAlmostEqual(sltn, prob.value)

        # With linear operators.
        kernel = np.array([1, 2, 3])
        kernel_mat = np.matrix("2 1 3; 3 2 1; 1 3 2")
        x = px.Variable(3)
        b = np.array([-41, 413, 2])
        prox_fns = [px.nonneg(x), px.sum_squares(px.conv(kernel, x), b=b)]
        sltn = ladmm.solve(prox_fns, [],
                           0.1,
                           max_iters=3000,
                           eps_abs=1e-5,
                           eps_rel=1e-5)

        cvx_X = cvx.Variable(3)
        cost = cvx.norm(kernel_mat * cvx_X - b)
        prob = cvx.Problem(cvx.Minimize(cost), [cvx_X >= 0])
        prob.solve()
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=2e-2)

        psi_fns, omega_fns = ladmm.partition(prox_fns)
        sltn = ladmm.solve(psi_fns,
                           omega_fns,
                           0.1,
                           max_iters=3000,
                           eps_abs=1e-5,
                           eps_rel=1e-5)
        self.assertItemsAlmostEqual(x.value, cvx_X.value, eps=2e-2)

    @pytest.mark.skip(reason="algorithm failed to converge")
    def test_equil(self):
        """Test equilibration.
        """
        from proximal.algorithms.equil import newton_equil
        np.random.seed(1)
        kernel = np.array([1, 1, 1]) / np.sqrt(3)
        kernel_mat = np.ones((3, 3)) / np.sqrt(3)
        x = px.Variable(3)
        wr = np.array([10, 5, 7])
        K = px.mul_elemwise(wr, x)
        K = px.conv(kernel, K)
        wl = np.array([100, 50, 3])
        K = px.mul_elemwise(wl, K)
        K = px.CompGraph(K)

        # Equilibrate
        gamma = 1e-1
        d, e = px.equil(K, 1000, gamma=gamma, M=5)
        tmp = d * wl * kernel_mat * wr * e
        u, v = np.log(d), np.log(e)
        obj_val = np.square(tmp).sum() / 2 - u.sum() - v.sum() + \
            gamma * (np.linalg.norm(v)**2 + np.linalg.norm(u)**2)

        d, e = newton_equil(wl * kernel_mat * wr, gamma, 100)
        tmp = d * wl * kernel_mat * wr * e
        u, v = np.log(d), np.log(e)
        sltn_val = np.square(tmp).sum() / 2 - u.sum() - v.sum() + \
            gamma * (np.linalg.norm(v)**2 + np.linalg.norm(u)**2)
        self.assertAlmostEqual((obj_val - sltn_val) / sltn_val, 0, eps=1e-3)
