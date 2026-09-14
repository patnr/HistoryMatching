#!/usr/bin/env bash

# Colab doesn't provide
# - Auto-installing dependencies (neither pyproject.toml nor requirements.txt)
# - Pre-loading data/modules (aside from the notebook itself)
# This script takes care of the above by cloning the full (shallow) repo.

main () {
    set -e

    # Clear any existing REPO for a fresh git clone
    rm -rf REPO

    # Download repo
    URL=https://github.com/patnr/HistoryMatching.git
    if [[ ! -d REPO ]]; then git clone --depth=1 $URL REPO; fi

    # https://pythonspeed.com/articles/upgrade-pip/
    pip install --upgrade pip

    # Install dependencies (declared in REPO/pyproject.toml)
    pip install ./REPO

    # Put repo contents (including hidden files) in PWD
    shopt -s dotglob
    cp -r REPO/notebooks/* ./
}

# Only run if we're on colab
if python -c "import google.colab" 2>/dev/null; then

    # Use `bash -s -- --debug` to get verbose output
    if echo $@ | grep -E -- '(--debug|-v)' > /dev/null ; then
        main
    else
        # Quiet
        main > /dev/null 2>&1
    fi

    echo "Initialization for Colab done."
else
    echo "Not running on Colab => Didn't do anything."
fi
