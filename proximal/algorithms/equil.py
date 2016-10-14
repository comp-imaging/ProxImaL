
import numpy as np
import math


def equil(K, iters=50, gamma=1e-1, M=math.log(1e4)):
    """Computes diagonal D, E so that DKE is approximately equilibrated.
    """
    m = K.output_size
    n = K.input_size
    alpha, beta = get_alpha_beta(m, n)

    u = np.zeros(m)
    v = np.zeros(n)
    ubar = u.copy()
    vbar = v.copy()

    in_buf = np.zeros(n)
    out_buf = np.zeros(m)

    # Main loop.
    for t in range(1, iters + 1):
        step_size = 2 / (gamma * (t + 1))
        # u grad estimate.
        s = np.random.choice([-1, 1], size=n)
        K.forward(np.exp(v) * s, out_buf)
        u_grad = np.exp(2 * u) * np.square(out_buf) - alpha**2 + gamma * u

        # v grad estimate.
        w = np.random.choice([-1, 1], size=m)
        K.adjoint(np.exp(u) * w, in_buf)
        v_grad = np.exp(2 * v) * np.square(in_buf) - beta**2 + gamma * v

        u = project(u - step_size * u_grad, M)
        v = project(v - step_size * v_grad, M)
        # Update averages.
        ubar = 2 * u / (t + 2) + t * ubar / (t + 2)
        vbar = 2 * v / (t + 2) + t * vbar / (t + 2)

    return np.exp(ubar), np.exp(vbar)


def get_alpha_beta(m, n):
    return (n / m)**(0.25), (m / n)**(0.25)


def project(x, M):
    """Project x onto [-M, M]^n.
    """
    return np.minimum(M, np.maximum(x, -M, out=x), out=x)

# Comparison method.


def f(A, u, v, gamma, p=2):
    m, n = A.shape
    alpha, beta = get_alpha_beta(m, n)
    total = (1. / p) * np.exp(p * u).T.dot(np.power(np.abs(A), p)).dot(np.exp(p * v))
    total += -alpha**p * u.sum() - beta**p * v.sum() + (gamma / 2) * ((u * u).sum() + (v * v).sum())
    return np.sum(total)


def get_grad(A, u, v, gamma, p=2):
    m, n = A.shape
    alpha, beta = get_alpha_beta(m, n)

    tmp = np.diag(np.exp(p * u)).dot((A * A)).dot(np.exp(p * v))
    grad_u = tmp - alpha**p + gamma * u
    du = -grad_u / (2 * tmp + gamma)

    tmp = np.diag(np.exp(p * v)).dot((A.T * A.T)).dot(np.exp(p * u))
    grad_v = tmp - beta**p + gamma * v
    dv = -grad_v / (2 * tmp + gamma)

    return du, dv, grad_u, grad_v


def newton_equil(A, gamma, max_iters):
    alpha = 0.25
    beta = 0.5
    m, n = A.shape
    u = np.zeros(m)
    v = np.zeros(n)
    for i in range(max_iters):
        du, dv, grad_u, grad_v = get_grad(A, u, v, gamma)
        # Backtracking line search.
        t = 1
        obj = f(A, u, v, gamma)
        grad_term = np.sum(alpha * (grad_u.dot(du) + grad_v.dot(dv)))
        while True:
            new_obj = f(A, u + t * du, v + t * dv, gamma)
            if new_obj > obj + t * grad_term:
                t = beta * t
            else:
                u = u + t * du
                v = v + t * dv
                break
    return np.exp(u), np.exp(v)
