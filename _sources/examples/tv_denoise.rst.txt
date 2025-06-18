Total variation denoising
############################

Signal distortion model:

.. math::
    \min_u \| u  - b \|_2^2 + 0.1 \| \nabla_x u \|_1 + 0.1 \| \nabla_y u \|_1  + \mathtt{nonneg}(u)

Textual representation in ProxImaL:

.. code-block:: python
    
    prob = Problem(
        sum_squares(u - b / 255) + .1 * norm1(grad(u)) + nonneg(u))

Example code: https://github.com/comp-imaging/ProxImaL/blob/master/proximal/examples/basic_denoise.py

Expected output:

.. image:: ../files/hello_world_results.png