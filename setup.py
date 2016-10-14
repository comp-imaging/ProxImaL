from setuptools import setup

setup(
    name='proximal',
    version='0.1.6',
    packages=['proximal',
              'proximal.prox_fns',
              'proximal.lin_ops',
              'proximal.algorithms',
              'proximal.utils',
              'proximal.halide',
              'proximal.tests',
              'proximal.tests.data'],
    package_dir={'proximal': 'proximal'},
    package_data={'proximal.tests.data': ['angela.jpg'],
                  'proximal.halide': ['src/*.cpp', 'src/core/*', 'src/external/*', 'src/fft/*']},
    url='http://github.com/comp-imaging/ProxImaL/',
    install_requires=["numpy >= 1.9",
                      "scipy >= 0.15",
                      "Pillow",
                      "opencv-python"],
    use_2to3=True,
)
