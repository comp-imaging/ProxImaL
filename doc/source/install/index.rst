.. _install:

Install Guide
=============

Anaconda install
----------------

The easiest way to install ProxImaL and its dependencies is using `Anaconda`_.
To install ProxImaL without `Anaconda`_, see the section on installation from source.

1. Install `Anaconda`_. On Linux and Mac OS X, install all Python dependencies by running

   ::

      conda install numpy scipy pil pip opencv

   On Windows, install all Python dependencies by running

   ::

     conda install numpy scipy pil pip
     conda install -c https://conda.binstar.org/menpo opencv

   If you are using Python 3, install dependencies by running

   ::

     conda install numpy scipy pil pip
     conda install -c https://conda.binstar.org/menpo opencv3


2. (Optional) Install `Halide`_ and define the ``HALIDE_PATH`` environment variable to point to the installation. Also, add the `Halide`_ binary to your ``LD_LIBRARY_PATH`` for Linux or ``DYLD_LIBRARY_PATH`` for Mac OS X. 

3. Install ``proximal`` with ``pip`` from the command-line.

   ::

       pip install proximal 

4. (Optional) Test the installation with ``nose``. The tests require that you have `CVXPY`_ installed.

  ::

       conda install nose
       nosetests proximal 

Install from source
-------------------

ProxImaL has the following dependencies:

* Python 2.7
* `setuptools`_ >= 1.4
* `NumPy`_ >= 1.10
* `SciPy`_ >= 0.15
* `PIL`_
* `cv2`_

`Halide`_ installation is optional, but necessary for the best performance. 
To test the ProxImaL installation, you additionally need `Nose`_ and `CVXPY`_.

Once you've installed the dependencies, installing ProxImaL from source is simple:

1. Clone the `ProxImaL git repository <https://github.com/comp-imaging/ProxImaL>`_.
2. Navigate to the top-level of the cloned directory and run

   ::

       python setup.py install

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _website: https://store.continuum.io/cshop/anaconda/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _multiprocess: https://github.com/uqfoundation/multiprocess/
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _Nose: http://nose.readthedocs.org
.. _PIL: http://www.pythonware.com/products/pil/
.. _cv2: http://opencv.org/
.. _Halide: http://halide-lang.org/
.. _CVXPY: http://www.cvxpy.org/
