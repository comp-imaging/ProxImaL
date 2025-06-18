TV-regularized image deconvolution
######################################

Signal distortion model:

.. math::
    \min_u \| \mathtt{conv}(k, u)  - b \|_2^2 + \lambda \| \nabla u \|_1

Textual representation in ProxImaL:

.. code-block:: python
    
    prob = Problem(
        sum_squares(conv(K, u, dims=2) - b) +
        lambda_tv * group_norm1(grad(u, dims=2), [2]),
    )

Example code: https://github.com/comp-imaging/ProxImaL/blob/master/proximal/examples/test_deconv.py

Expected output:

.. figure:: ../files/deconv.png