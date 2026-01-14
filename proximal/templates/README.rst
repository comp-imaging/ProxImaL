ProxImaL-to-Halide code generation templates
==============================================

Overview
----------

The folder ``proximal/templates/*.j2`` contains code templates to translate the
ProxImaL-optimized inverse problem formulation from Python to Halide/C++17 code
generator. The templates are written in Jinja2 language.

Usage::

    from jinja2 import Environment, PackageLoader

    # Loads the template/ folder in proximal package
    env = Environment(
        loader=PackageLoader('proximal')
    )

    # Substitute values and export the problem input and output image sizes.
    template = env.get_template('problem-config.h.j2')
    with open('proximal/halide/src/user-problem/problem-config.h') as f:
        f.write(template.render(psi_fns=psi_fns, omega_fns=omega_fns, K=K))

    # Export the proximal functions and mapping functions after absortion,
    # splitting, and scaling by ProxImaL core functions.
    template = env.get_template('problem-definition.h.j2')
    with open('proximal/halide/src/user-problem/problem-definition.h') as f:
        f.write(template.render(psi_fns=psi_fns, omega_fns=omega_fns, K=K,
            # ... and other necessary components ...
        ))

After that, invoke ``ninja -C proximal/halide/build`` to generate the
Halide-optimized (L-)ADMM algorithm solver of the user-defined problem family.

Working principle
------------------

The ProxImaL compiler is composed of two software stacks: the problem
formulation language as the design frontend, and the code generator as the
hardware specific backend. The two stacks meet at the middle, as illustrated in
the hourglass model of the following figure.

It is expected that various applications, ranging from consumer-oriented imaging
(e.g. Bayer raw image demoasicking) to research-oriented imaging (e.g.
Ptychographic phase retrieval and 2D/3D deconvolution) is mapped to
hardware-accelerated compute platforms (e.g. ``x86-64`` / ``aarch64`` CPUs,
``OpenCL`` / ``NvPTX`` GPUs) through a unifying interface, illustrated as the
"narrow waist" in the hourglass model.

.. figure:: files/hourglass.svg

    The hourglass architecture of the ProxImaL project. The application-driven
    problem formulation (e.g. Bayer raw demoasicing, 2D/3D deconvolution,
    Ptychographic phase retrieval) and the architectures (e.g. multi-core CPUs,
    GPUs) meets at a narrow waist in the middle.

    Figure adapted from Lee, Edward A., Stephen Neuendorffer, and Michael J.
    Wirthlin. "Actor-Oriented Design of Embedded Hardware and Software Systems."
    Journal of Circuits, Systems and Computers, vol. 12, no. 3, 2003, pp.
    231-260. https://doi.org/10.1142/S0218126603000751

On the other hand, it is expected that various problem solvers (e.g. Linearized
ADMM, Pock-Chambolle, ADMM), and various hardware-accelerated compute
architectures (e.g. Intel/AMD CPUs, ARM CPUs, Nvidia/AMD GPUs) can be exposed
bottom-up, again meeting in the middle of the "hourglass" through the code
generation templates.