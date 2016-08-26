from proximal.tests.base_test import BaseTest
from proximal.lin_ops import Variable, mul_elemwise, subsample
from proximal.prox_fns import norm1, sum_squares
from proximal.algorithms import Problem
from proximal.utils.utils import Impl
import cvxpy as cvx
import numpy as np


class TestProblem(BaseTest):

    def test_problem(self):
        """Test problem object.
        """
        X = Variable((4, 2))
        B = np.reshape(np.arange(8), (4, 2)) * 1.
        prox_fns = [norm1(X), sum_squares(X, b=B)]
        prob = Problem(prox_fns)
        # prob.partition(quad_funcs = [prox_fns[0], prox_fns[1]])
        prob.set_automatic_frequency_split(False)
        prob.set_absorb(False)
        prob.set_implementation(Impl['halide'])
        prob.set_solver('admm')
        prob.solve()

        true_X = norm1(X).prox(2, B.copy())
        self.assertItemsAlmostEqual(X.value, true_X, places=2)
        prob.solve(solver="pc")
        self.assertItemsAlmostEqual(X.value, true_X, places=2)
        prob.solve(solver="hqs", eps_rel=1e-6,
                   rho_0=1.0, rho_scale=np.sqrt(2.0) * 2.0, rho_max=2**16,
                   max_iters=20, max_inner_iters=500, verbose=False)
        self.assertItemsAlmostEqual(X.value, true_X, places=2)

        # CG
        prob = Problem(prox_fns)
        prob.set_lin_solver("cg")
        prob.solve(solver="admm")
        self.assertItemsAlmostEqual(X.value, true_X, places=2)
        prob.solve(solver="hqs", eps_rel=1e-6,
                   rho_0=1.0, rho_scale=np.sqrt(2.0) * 2.0, rho_max=2**16,
                   max_iters=20, max_inner_iters=500, verbose=False)
        self.assertItemsAlmostEqual(X.value, true_X, places=2)

        # Quad funcs.
        prob = Problem(prox_fns)
        prob.solve(solver="admm")
        self.assertItemsAlmostEqual(X.value, true_X, places=2)

        # Absorbing lin ops.
        prox_fns = [norm1(5 * mul_elemwise(B, X)), sum_squares(-2 * X, b=B)]
        prob = Problem(prox_fns)
        prob.set_absorb(True)
        prob.solve(solver="admm", eps_rel=1e-6, eps_abs=1e-6)

        cvx_X = cvx.Variable(4, 2)
        cost = cvx.sum_squares(-2 * cvx_X - B) + cvx.norm(5 * cvx.mul_elemwise(B, cvx_X), 1)
        cvx_prob = cvx.Problem(cvx.Minimize(cost))
        cvx_prob.solve(solver=cvx.SCS)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, places=2)

        prob.set_absorb(False)
        prob.solve(solver="admm", eps_rel=1e-6, eps_abs=1e-6)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, places=2)

        # Constant offsets.
        prox_fns = [norm1(5 * mul_elemwise(B, X)), sum_squares(-2 * X - B)]
        prob = Problem(prox_fns)
        prob.solve(solver="admm", eps_rel=1e-6, eps_abs=1e-6)
        self.assertItemsAlmostEqual(X.value, cvx_X.value, places=2)

    def test_single_func(self):
        """Test problems with only a single function to minimize.
        """
        X = Variable((4, 2))
        B = np.reshape(np.arange(8), (4, 2)) * 1.
        prox_fns = [sum_squares(X - B)]
        prob = Problem(prox_fns[0])
        prob.solve(solver="admm", eps_rel=1e-6, eps_abs=1e-6)
        self.assertItemsAlmostEqual(X.value, B, places=2)

    def test_multiple_vars(self):
        """Test problems with multiple variables."""
        x = Variable(3)
        y = Variable(6)
        rhs = np.array([1, 2, 3])
        prob = Problem([sum_squares(x - rhs),
                        sum_squares(subsample(y, [2]) - x)])
        prob.solve(solver="admm", eps_rel=1e-6, eps_abs=1e-6)
        self.assertItemsAlmostEqual(x.value, [1, 2, 3], places=3)
        self.assertItemsAlmostEqual(y.value, [1, 0, 2, 0, 3, 0], places=3)
