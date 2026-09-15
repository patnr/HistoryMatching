#!/usr/bin/env bash

# Bootstrap for Google Colab, invoked from the first code cell of each notebook:
#
#     !wget -qO- https://raw.githubusercontent.com/patnr/HistoryMatching/master/colab_bootstrap.sh | bash -s
#
# Colab does not auto-install a project's dependencies (no pyproject/requirements
# support), nor does it fetch anything but the notebook itself. So this script
# 1. shallow-clones the repo (or copies it from `--repo=PATH`, for tests),
# 2. installs ONLY the packages Colab lacks, with `--no-deps`, so that nothing
#    preinstalled (and already imported) gets touched, which would otherwise
#    require a runtime restart. See requirements-colab.txt and pyproject.toml.
# 3. copies the notebooks dir (incl. `tools/`) into the working directory.
#
# Options: `--debug`/`-v` for verbose output; `--branch=NAME` (default: master).

main () {
    set -e

    # Clear any existing REPO for a fresh copy
    rm -rf REPO

    if [[ -n "$SRC" ]]; then
        # Local checkout (used by tests/colab/smoke.sh)
        mkdir REPO
        tar -C "$SRC" --exclude=.git -cf - . | tar -C REPO -xf -
    else
        git clone --depth=1 --branch "$BRANCH" https://github.com/patnr/HistoryMatching.git REPO
    fi

    # Install what Colab lacks. NB: no pip upgrade, no dependency resolution.
    pip install --no-deps -r REPO/requirements-colab.txt

    # Put notebooks (including hidden files) in PWD
    shopt -s dotglob
    cp -r REPO/notebooks/* ./
}

# Parse args
SRC=""
BRANCH="master"
DEBUG=""
for arg in "$@"; do
    case "$arg" in
        --repo=*)   SRC="${arg#--repo=}" ;;
        --branch=*) BRANCH="${arg#--branch=}" ;;
        --debug|-v) DEBUG=1 ;;
    esac
done

# Only run if we're on Colab (the env var is also set in Colab's runtime image)
if [[ -n "$COLAB_RELEASE_TAG" ]] || python -c "import google.colab" 2>/dev/null; then
    if [[ -n "$DEBUG" ]]; then
        main
    else
        main > /dev/null 2>&1
    fi
    echo "Initialization for Colab done."
else
    echo "Not running on Colab => Didn't do anything."
fi
