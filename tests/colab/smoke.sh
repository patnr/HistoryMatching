#!/usr/bin/env bash
# Smoke-test the tutorials inside Google's published Colab runtime image,
# LOCALLY. The same test runs monthly on GitHub Actions
# (.github/workflows/colab-compat.yml), which is the primary way to run it:
# native x86-64 and a fast network. Use this script when you need to iterate
# without pushing.
#
#   tests/colab/smoke.sh                  # local checkout, both notebooks
#   tests/colab/smoke.sh --remote         # what students get: bootstrap from GitHub master
#   SMOKE_NOTEBOOKS=HistoryMatch.ipynb tests/colab/smoke.sh
#   COLAB_IMAGE=us-docker.pkg.dev/colab-images/public/runtime:<tag> tests/colab/smoke.sh
#
# Checks (see smoke_inner.sh): the bootstrap runs, it modifies no preinstalled
# package (=> no "restart runtime"), and every notebook executes without error.
#
# The image is x86-64 and ~23 GB compressed (it is the GPU image: CUDA, torch,
# TF, ...), so the first run takes a while; later runs use the cached image.
# On Apple silicon it runs emulated (Rosetta via `podman machine`), i.e. slowly.
# Tags: `latest`, or dated ones like `release-colab_20250925-060051_RC00`;
# list with `podman search --list-tags us-docker.pkg.dev/colab-images/public/runtime`.
# Note the public image lags Colab's actual runtime by a few weeks; for what
# students *currently* have, see https://github.com/googlecolab/backend-info .
set -euo pipefail

IMAGE=${COLAB_IMAGE:-us-docker.pkg.dev/colab-images/public/runtime:latest}
ENGINE=${CONTAINER_ENGINE:-$(command -v podman || command -v docker)}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

MODE=local
for arg in "$@"; do
    case "$arg" in
        --remote) MODE=remote ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

# The repo is copied into the container rather than bind-mounted: it is a few
# MB, and bind mounts are unreliable in some `podman machine` setups
# (e.g. "statfs ...: connection refused" with the libkrun provider on macOS).
name=hm-colab-smoke-$$
staging=$(mktemp -d)
trap '"$ENGINE" rm -f "$name" >/dev/null 2>&1; rm -rf "$staging"' EXIT

mkdir "$staging/repo"
tar -C "$REPO" --exclude=.git --exclude=.claude --exclude=.ipynb_checkpoints \
    --exclude=.ruff_cache --exclude=.pytest_cache -cf - . | tar -C "$staging/repo" -xf -

"$ENGINE" create --name "$name" --platform linux/amd64 \
    -e REPO_DIR=/repo \
    -e SMOKE_MODE="$MODE" \
    -e SMOKE_BRANCH="${SMOKE_BRANCH:-master}" \
    -e SMOKE_NOTEBOOKS="${SMOKE_NOTEBOOKS:-HistoryMatch.ipynb Optimise.ipynb}" \
    -e SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-1800}" \
    --entrypoint bash "$IMAGE" /repo/tests/colab/smoke_inner.sh >/dev/null
"$ENGINE" cp "$staging/repo" "$name":/repo
"$ENGINE" start --attach "$name"
