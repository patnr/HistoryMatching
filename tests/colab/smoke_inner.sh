#!/usr/bin/env bash
# Runs INSIDE the Colab runtime image, either on GitHub Actions
# (.github/workflows/colab-compat.yml) or locally via smoke.sh.
#
# Env: REPO_DIR (checkout of this repo, default /repo),
#      SMOKE_MODE=local|remote (default local; remote = what students get, i.e.
#      the bootstrap fetched from GitHub master, notebook cell included),
#      SMOKE_NOTEBOOKS (default: both), SMOKE_TIMEOUT (per cell, default 1800),
#      SMOKE_OUT (dir for the executed notebooks, default /content/executed).
set -euo pipefail
REPO_DIR=${REPO_DIR:-/repo}

MODE=${SMOKE_MODE:-local}
NOTEBOOKS=${SMOKE_NOTEBOOKS:-"HistoryMatch.ipynb Optimise.ipynb"}
TIMEOUT=${SMOKE_TIMEOUT:-1800}
OUT=${SMOKE_OUT:-/content/executed}

mkdir -p /content && cd /content
echo "== Image: ${COLAB_RELEASE_TAG:-unknown}  |  $(python --version)  |  $(which python) $(which pip)"
echo "== Mode: $MODE"

pip freeze > /tmp/before.txt

echo "== 1/3 bootstrap"
if [[ "$MODE" == remote ]]; then
    remote=https://raw.githubusercontent.com/patnr/HistoryMatching
    wget -qO- "$remote/master/colab_bootstrap.sh" | bash -s -- --debug
else
    bash "$REPO_DIR/colab_bootstrap.sh" --debug --repo="$REPO_DIR"
fi

# The bootstrap is a silent no-op outside Colab: make sure it actually ran.
test -f HistoryMatch.ipynb -a -d tools || { echo "FAIL: bootstrap did not populate $PWD"; exit 1; }

echo "== 2/3 preinstalled packages untouched?"
pip freeze > /tmp/after.txt
python "$REPO_DIR/tests/colab/check_freeze.py" /tmp/before.txt /tmp/after.txt

echo "== 3/3 execute notebooks"
skip=""; [[ "$MODE" == local ]] && skip="--skip-bootstrap"
mkdir -p "$OUT"
status=0
for nb in $NOTEBOOKS; do
    python "$REPO_DIR/tests/colab/run_nb.py" "$nb" --timeout "$TIMEOUT" $skip --out "$OUT/$nb" || status=1
done
exit $status
