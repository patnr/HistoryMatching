#!/usr/bin/env bash

# Colab doesn't provide functionality for working with full packages/repos, including
# - defining python environments (e.g. requirements.txt)
# - pre-loading data/scripts other than what's in the notebook.
# We therefore make this script for bootstrapping the notebooks by cloning the full repo.
#!wget -qO- $URL | bash -s -- --debug


setup () {
    set -e

    # Install requirements
    URL=https://github.com/patricknraanes/HistoryMatching.git
    if [[ ! -d REPO ]]; then git clone --depth=1 $URL REPO; fi
    pip install -r REPO/requirements.txt

    # Install dependency from source -- NB: no versioning!
    # coz colab'p pip couldn't find package,
    # even though it was on pypi (uploaded 15 min earlier).
    # Could be that Colab/pypi needs some time to fetch NEW packages.
    URL=https://github.com/patricknraanes/pylib.git
    if [[ ! -d patlib ]]; then git clone --depth=1 $URL patlib; fi
    pip install -e patlib
}

# Only run if we're on colab
if python -c "import colab"; then
    # Quiet execution
    if echo $@ | grep -- '--debug' > /dev/null ; then
        setup
    else
        setup > /dev/null 2>&1
    fi
    echo "Initialization for Colab done."
fi
