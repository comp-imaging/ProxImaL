from proximal.prox_fns.group_norm1 import group_norm1
from proximal.lin_ops.variable import Variable

from numpy import random
import numpy as np

import pycuda.compiler

pycuda.compiler.DEFAULT_NVCC_FLAGS.extend(['-ccbin', 'clang-3.8', '--prec-sqrt=true'])

def check_group_norm1(eps = 1e-4):
    x = Variable((10,10,2,3))
    f = group_norm1(x, [2,3])
    
    v = np.reshape(np.arange(10*10*2*3), (10,10,2,3)).astype(np.float32)
    xhat1 = f.prox(1, v.copy())
    xhat2 = f.prox_cuda(1, v.copy()).get()
    
    if not np.all(np.abs(xhat1 - xhat2) < eps):
        print(f.cuda_code)
        print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
        assert(False)
        
    maxeps = 0
    for i in range(50):
        v = random.rand(10,10,2,3).astype(np.float32)
        rho = np.abs(random.rand(1))
        xhat1 = f.prox(rho, v.copy())
        xhat2 = f.prox_cuda(rho, v.copy()).get()
        
        err = np.amax(np.abs(xhat1 - xhat2))
        if not err < eps:
            print(f.cuda_code)
            print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
            assert(False)
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
            print(f.cuda_code)
            print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
            assert(False)
        maxeps = max(err,maxeps)

    print("Max error: %.4e" % maxeps)
    
if __name__ == "__main__":
    check_group_norm1()
    