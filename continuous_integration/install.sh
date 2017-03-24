#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

sudo apt-get update -qq
if [[ "$DISTRIB" == "conda" ]]; then

    sudo apt-get install -qq libatlas-base-dev gfortran
    export ATLAS="/usr/lib/atlas-base/libatlas.so"
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda2/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy scipy pillow libgcc
    source activate testenv
    # if [[ "$PYTHON_VERSION" == "3.5" ]]; then
    #     conda install --yes -c https://conda.binstar.org/menpo opencv3
    # else
    #     conda install --yes opencv
    # fi
    conda install --yes -c cvxgrp scs multiprocess cvxcanon ecos
    pip install flake8
    pip install cvxpy
    pip install opencv-python
    # if [[ "$INSTALL_MKL" == "true" ]]; then
    #     # Make sure that MKL is used
    #     conda install --yes mkl
    # else
    #     # Make sure that MKL is not used
    #     conda remove --yes --features mkl || echo "MKL not installed"
    # fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo apt-get update -qq
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-pip python-scipy python-numpy
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
