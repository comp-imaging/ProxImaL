Image deconvolution with spatial varying point spread functions (SV-PSFs)
############################################################################

References: Denis, L., Thiébaut, E., Soulez, F. et al. Fast Approximations of
Shift-Variant Blur. Int J Comput Vis 115, 253--278 (2015).
https://doi.org/10.1007/s11263-015-0817-x

Signal distortion model:

.. math::
    \min_u \left\Vert
    \sum_{i=1}^N
    \mathtt{conv}[h_i, \mathrm{Diag}(w_i) u]
     - b \right\Vert_2^2 +
    \mu \alpha \Vert \nabla u \Vert_1 +
    \mu (1 - \alpha) \Vert \nabla u \Vert_2^2.

Textual representation in ProxImaL:

.. code-block:: python

    grad_term = grad(u)

    prob = Problem([
        sum_squares(
            sum([
                conv(psf_modes[..., i], mul_elemwise(weights[..., i], u))
                for i in range(n_psf)
            ]) - raw_image
        ),
        lambda_tv * alpha * group_norm1(grad_term, group_dims=[2]),
        lambda_tv * (1.0 - alpha) * sum_squares(grad_term),
    ])

Example code: https://github.com/comp-imaging/ProxImaL/blob/master/proximal/examples/test_deconv_sv_psf.py

Expected output:

.. figure:: ../files/devonv-sv-psf-inputs.png

.. figure:: ../files/deconv-sv-psf-closeup.png