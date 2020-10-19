#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

python3 --version
python3 -c "import numpy; print('numpy %s' % numpy.__version__)"
python3 -c "import scipy; print('scipy %s' % scipy.__version__)"
python3 setup.py install

if [[ "$COVERAGE" == "true" ]]; then
    export WITH_COVERAGE="--cov=."
else
    export WITH_COVERAGE=""
fi
python3 -m pytest ${WITH_COVERAGE} proximal/tests/
