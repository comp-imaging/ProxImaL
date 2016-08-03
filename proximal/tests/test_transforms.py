from proximal.tests.base_test import BaseTest
from proximal.prox_fns import sum_squares, norm1, sum_entries, nonneg
from proximal.lin_ops import Variable, grad, mul_elemwise, sum
from proximal.algorithms import can_merge, absorb_offset, merge_fns, absorb_lin_op
from proximal.algorithms.merge import merge_all
import numpy as np
import cvxpy as cvx
import proximal.algorithms.absorb as absorb


class TestTransforms(BaseTest):

    def test_absorb_lin_op(self):
        """Test absorb lin op operator.
        """
        # norm1.
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0

        fn = norm1(mul_elemwise(-v, tmp), alpha=5.)
        rho = 2
        new_prox = absorb_lin_op(fn)[0]
        x = new_prox.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.sign(v) * np.maximum(np.abs(v) - 5. *
                                                               np.abs(v) / rho, 0))

        fn = norm1(mul_elemwise(-v, mul_elemwise(2 * v, tmp)), alpha=5.)
        rho = 2
        new_prox = absorb_lin_op(fn)[0]
        x = new_prox.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.sign(v) * np.maximum(np.abs(v) - 5. *
                                                               np.abs(v) / rho, 0))
        new_prox = absorb_lin_op(new_prox)[0]
        x = new_prox.prox(rho, v.copy())
        new_v = 2 * v * v
        self.assertItemsAlmostEqual(x, np.sign(new_v) *
                                    np.maximum(np.abs(new_v) - 5. * np.abs(new_v) / rho, 0))

        # nonneg.
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0

        fn = nonneg(mul_elemwise(-v, tmp), alpha=5.)
        rho = 2
        new_prox = absorb_lin_op(fn)[0]
        x = new_prox.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, fn.prox(rho, -np.abs(v)))

        # sum_squares.
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0

        alpha = 5.
        val = np.arange(10)
        fn = sum_squares(mul_elemwise(-v, tmp), alpha=alpha, c=val)
        rho = 2
        new_prox = absorb_lin_op(fn)[0]
        x = new_prox.prox(rho, v.copy())

        cvx_x = cvx.Variable(10)
        prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(cvx_x - v) * (rho / 2) +
                                        5 * cvx.sum_squares(cvx.mul_elemwise(-v,
                                                            cvx_x)) + (val * -v).T * cvx_x
                                        ))
        prob.solve()
        self.assertItemsAlmostEqual(x, cvx_x.value, places=3)

        # Test scale.
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0

        fn = norm1(10 * tmp)
        rho = 2
        new_prox = absorb_lin_op(fn)[0]
        x = new_prox.prox(rho, v.copy())
        cvx_x = cvx.Variable(10)
        prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(cvx_x - v) + cvx.norm(10 * cvx_x, 1)))
        prob.solve()
        self.assertItemsAlmostEqual(x, cvx_x.value, places=3)

        val = np.arange(10)
        fn = norm1(10 * tmp, c=val, b=val, gamma=0.01)
        rho = 2
        new_prox = absorb_lin_op(fn)[0]
        x = new_prox.prox(rho, v.copy())
        cvx_x = cvx.Variable(10)
        prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(cvx_x - v) +
                                        cvx.norm(10 * cvx_x - val, 1) + 10 * val.T * \
                                                 cvx_x + cvx.sum_squares(cvx_x)
                                        ))
        prob.solve()
        self.assertItemsAlmostEqual(x, cvx_x.value, places=2)

        # sum_entries
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0

        fn = sum_entries(sum([10 * tmp, mul_elemwise(v, tmp)]))

        funcs = absorb.absorb_all_lin_ops([fn])
        c = __builtins__['sum']([func.c for func in funcs])
        self.assertItemsAlmostEqual(c, v + 10, places=3)

    def test_merge(self):
        """Test merging functions.
        """
        # sum_entries
        x = Variable(10)
        fn1 = sum_entries(x, gamma=1.0)
        fn2 = norm1(x)
        assert can_merge(fn1, fn2)
        merged = merge_fns(fn1, fn2)
        v = np.arange(10) * 1.0 - 5.0
        prox_val1 = merged.prox(1.0, v.copy())
        tmp = norm1(x, c=np.ones(10), gamma=1.0)
        prox_val2 = tmp.prox(1.0, v.copy())
        self.assertItemsAlmostEqual(prox_val1, prox_val2)

        # sum_squares
        x = Variable(10)
        val = np.arange(10)
        fn1 = sum_squares(x, gamma=1.0, beta=2.0, alpha=3.0, b=val)
        fn2 = norm1(x)
        assert can_merge(fn1, fn2)
        merged = merge_fns(fn1, fn2)
        v = np.arange(10) * 1.0 - 5.0
        prox_val1 = merged.prox(1.0, v.copy())
        tmp = norm1(x, c=-12 * val, gamma=1.0 + 12, d=val.dot(val))
        prox_val2 = tmp.prox(1.0, v.copy())
        self.assertItemsAlmostEqual(prox_val1, prox_val2)

    def test_merge_all(self):
        """Test function to merge all prox operators possible.
        """
        # merge all
        x = Variable(10)
        lin_op = grad(x)
        fns = [sum_squares(lin_op), sum_entries(lin_op), nonneg(lin_op)]
        merged = merge_all(fns)
        assert len(merged) == 1
        v = np.reshape(np.arange(10) * 1.0 - 5.0, (10, 1))
        prox_val1 = merged[0].prox(1.0, v.copy())
        tmp = nonneg(lin_op, c=np.ones((10, 1)), gamma=1.0)
        prox_val2 = tmp.prox(1.0, v.copy())
        self.assertItemsAlmostEqual(prox_val1, prox_val2)

    def test_const_val(self):
        """Test obtaining the constant offset.
        """
        x = Variable(10)
        b = np.arange(10)
        expr = x - b
        self.assertItemsAlmostEqual(-b, expr.get_offset())
        fn = sum_squares(expr)
        new_fn = absorb_offset(fn)
        self.assertItemsAlmostEqual(b, new_fn.b)
