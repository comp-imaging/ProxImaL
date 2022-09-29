# Halide-codegen templates

Halide/C++17 templates to export ProxImaL-optmized problem to code generator,
written in Jinja2 language.

This folder contains all the necessary boilerplates in `proximal/src/halide/user-problem/`, in order to control how Halide should parse, optimize, and export (L-)ADMM algorithms of the user defined problem family.

Usage:
```python
from jinja2 import Environment, PackageLoader

# Loads the template/ folder in proximal package
env = Environment(
    loader=PackageLoader('proximal')
)

# Substitute values and export the problem input and output image sizes
template = env.get_template('problem-config.h.j2')
with open('proximal/halide/src/user-problem/problem-config.h') as f:
    f.write(template.render(psi_fns=psi_fns, omega_fns=omega_fns, K=K))

# Export the proximal functions and mapping functions after absortion, splitting, and scaling by ProxImaL core functions
template = env.get_template('problem-definition.h.j2')
with open('proximal/halide/src/user-problem/problem-definition.h') as f:
    f.write(template.render(psi_fns=psi_fns, omega_fns=omega_fns, K=K,
        # ... and other necessary components ...
    ))
```

After that, invoke `ninja -C proximal/halide/build` to generate the Halide-optimized (L-)ADMM algorithm solver of the user-defined problem family.
