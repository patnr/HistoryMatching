#!/usr/bin/env bash
# Runs INSIDE the Colab runtime image, either on GitHub Actions
# (.github/workflows/colab-compat.yml) or locally via smoke.sh.
#
# Env: REPO_DIR (checkout of this repo, default /repo),
#      SMOKE_MODE=local|remote (default local), SMOKE_BRANCH (remote mode),
#      SMOKE_NOTEBOOKS (default: both), SMOKE_TIMEOUT (per cell, default 1800).
set -euo pipefail
REPO_DIR=${REPO_DIR:-/repo}

MODE=${SMOKE_MODE:-local}
NOTEBOOKS=${SMOKE_NOTEBOOKS:-"HistoryMatch.ipynb Optimise.ipynb"}
TIMEOUT=${SMOKE_TIMEOUT:-1800}

mkdir -p /content && cd /content
echo "== Image: ${COLAB_RELEASE_TAG:-unknown}  |  $(python --version)  |  $(which python) $(which pip)"
echo "== Mode: $MODE"

pip freeze > /tmp/before.txt

echo "== 1/3 bootstrap"
if [[ "$MODE" == remote ]]; then
    remote=https://raw.githubusercontent.com/patnr/HistoryMatching
    wget -qO- "$remote/${SMOKE_BRANCH:-master}/colab_bootstrap.sh" | bash -s -- --debug --branch="${SMOKE_BRANCH:-master}"
else
    bash "$REPO_DIR/colab_bootstrap.sh" --debug --repo="$REPO_DIR"
fi

echo "== 2/3 preinstalled packages untouched?"
pip freeze > /tmp/after.txt
python "$REPO_DIR/tests/colab/check_freeze.py" /tmp/before.txt /tmp/after.txt

echo "== 3/3 execute notebooks"
skip=""; [[ "$MODE" == local ]] && skip="--skip-bootstrap"
status=0
for nb in $NOTEBOOKS; do
    python "$REPO_DIR/tests/colab/run_nb.py" "$nb" --timeout "$TIMEOUT" $skip || status=1
done
exit $status
