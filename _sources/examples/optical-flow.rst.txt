Horn-Schunck optical flow algorithm
######################################

Signal distortion model:

.. math::
    \min_{\{u, v\}}
    \| \mathtt{Diag}(f_x) u  + \mathtt{Diag}(f_y) v - f_t \|_2^2
    + \alpha \| \nabla u \|_2^2
    + \alpha \| \nabla v \|_2^2.

Textual representation in ProxImaL:

.. code-block:: python
    
    prob = px.Problem([
        alpha * px.sum_squares(px.grad(u)),
        alpha * px.sum_squares(px.grad(v)),
        px.sum_squares(px.mul_elemwise(fx, u) + px.mul_elemwise(fy, v) + ft,),
    ])

Example code: https://github.com/comp-imaging/ProxImaL/blob/master/proximal/examples/test_optical_flow.py

Expected output:

.. figure:: ../files/optical_flow.png