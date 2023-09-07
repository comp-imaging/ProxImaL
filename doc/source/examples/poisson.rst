Total variation denoising with multiplicative Poisson noise
################################################################

Signal distortion model:

.. math::
    \min_u \| u , b \|_\mathtt{Poisson} + \lambda \| \nabla u \|_1

Textual representation in ProxImaL:

.. code-block:: python
    
    prob = Problem([
        poisson_norm(conv(K, u), b),
        lambda_tv * group_norm1(grad(u, dims=2), [2])
    ])

Example code: https://github.com/comp-imaging/ProxImaL/blob/master/proximal/examples/test_poisson.py

Expected output:

.. image:: ../files/poisson.png