from proximal.tests.base_test import BaseTest

from proximal.prox_fns.group_norm1 import group_norm1
from proximal.lin_ops.variable import Variable

import logging

from numpy import random
import numpy as np

class TestCudaProxFn(BaseTest):

    def test_group_norm1(self):
        random.seed(0)

        x = Variable((10,10,2,3))
        f = group_norm1(x, [2,3])

        v = np.reshape(np.arange(10*10*2*3), (10,10,2,3)).astype(np.float32)
        xhat1 = f.prox(1, v.copy())
        xhat2 = f.prox_cuda(1, v.copy()).get()

        if not np.all(np.abs(xhat1 - xhat2) < 1e-4):
            logging.error(f.cuda_code)
            logging.error("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
        self.assertTrue(np.all(np.abs(xhat1 - xhat2) < 1e-4))

        eps = 1e-5
        maxeps = 0
        for i in range(50):
            v = random.rand(10,10,2,3).astype(np.float32)
            rho = np.abs(random.rand(1))
            xhat1 = f.prox(rho, v.copy())
            xhat2 = f.prox_cuda(rho, v.copy()).get()

            err = np.amax(np.abs(xhat1 - xhat2))
            if not err < eps:
                logging.error(f.cuda_code)
                logging.error("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
            self.assertTrue(err < eps)
            maxeps = max(err,maxeps)

        for i in range(50):
            v = random.rand(10,10,2,3).astype(np.float32)
            rho = np.abs(random.rand(1))
            alpha = np.abs(random.rand(1))
            beta = np.abs(random.rand(1))
            gamma = np.abs(random.rand(1))
            c = np.abs(random.rand(*f.c.shape))
            b = np.abs(random.rand(*f.b.shape))

            xhat1 = f.prox(rho, v.copy(), alpha=alpha, beta=beta, gamma=gamma, c=c, b=b)
            xhat2 = f.prox_cuda(rho, v.copy()).get()

            err = np.amax(np.abs(xhat1 - xhat2))
            if not err < eps:
                logging.error(f.cuda_code)
                logging.error("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
            self.assertTrue(err < eps)
            maxeps = max(err,maxeps)

        logging.info("Max proxfn error: %.2e" % maxeps)

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)
    import unittest
    unittest.main()