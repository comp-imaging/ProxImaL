.. _install:

Install Guide
=============

Dependencies
-------------

ProxImaL has the following dependencies:

* Python 3.9
* `NumPy`_ <= 1.26.0
* `SciPy`_ >= 0.15
* Numexpr
* `Pillow`_
* `cv2`_

`Halide`_ compiler toolchain installation is fully automated as long as the
installation path is not root/admin restricted.

To test the ProxImaL installation, you additionally need `Pytest`_ and `CVXPY`_.

Install from source
-------------------

Installation is easiest with `UV`_ the pip accelerator. First, clone the
`ProxImaL git repository <https://github.com/comp-imaging/ProxImaL>`_.

Next, navigate to the top-level of the cloned directory and create a virtual
environment by::

   cd path/to/ProxImaL/
   uv venv --python=3.9
   source .venv/bin/activate

Now, install the project to the virtual environment::

   source .venv/bin/activate
   uv pip install -U -e .

Unit testing
------------

To run the unit test cases, reinstall the project by::

   cd path/to/ProxImaL/
   source .venv/bin/activate
   uv pip install -U -e .[test]

This action installs the `CVXPY`_ and the `Pytest`_ framework.

After installation, run the unit tests::

   pytest -rx ./proximal/tests/

Expected test report::

   collected 61 items

   proximal/tests/test_algs.py ...s.......                                  [ 18%]
   proximal/tests/test_cuda_comp_graph.py sssssssssssss                     [ 39%]
   proximal/tests/test_cuda_prox_fn.py s                                    [ 40%]
   proximal/tests/test_halide.py .......                                    [ 52%]
   proximal/tests/test_lin_ops.py ............                              [ 72%]
   proximal/tests/test_problem.py ss..                                      [ 78%]
   proximal/tests/test_prox_fn.py .........                                 [ 93%]
   proximal/tests/test_transforms.py s...                                   [100%]
   =========== 43 passed, 18 skipped, 14 warnings in 137.02s (0:02:17) ============

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _NumPy: https://www.numpy.org/
.. _SciPy: https://scipy.org/
.. _Pytest: https://pytest.org/
.. _Pillow: https://python-pillow.github.io/
.. _cv2: https://opencv.org/
.. _Halide: https://halide-lang.org/
.. _CVXPY: https://www.cvxpy.org/
.. _UV: https://docs.astral.sh/uv/#installation