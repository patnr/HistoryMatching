"""Execute a notebook headlessly (via `nbclient`), as a smoke test.

Usage: run_nb.py NOTEBOOK.ipynb [--timeout SECONDS] [--skip-bootstrap] [--out PATH]

- `--skip-bootstrap` removes the cell that runs `colab_bootstrap.sh` (so that
  a test of a local checkout does not fetch and install `master` on top).
- Fails (exit 1) on the first cell error, printing that cell's source and traceback.

Runs both locally (`uv run tests/colab/run_nb.py notebooks/HistoryMatch.ipynb`)
and inside the Colab runtime image (see smoke.sh), which ships `nbclient`.
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout (s)")
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--out", type=Path, help="write the executed notebook here")
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    if args.skip_bootstrap:
        n = len(nb.cells)
        nb.cells = [c for c in nb.cells if "colab_bootstrap.sh" not in c.source]
        print(f"Skipped {n - len(nb.cells)} bootstrap cell(s).")

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(args.notebook.parent)}},
    )
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        print(f"FAIL: {args.notebook} after {time.time() - t0:.0f}s\n{e}")
        return 1
    finally:
        if args.out:
            nbformat.write(nb, args.out)
    ncode = sum(c.cell_type == "code" for c in nb.cells)
    print(f"OK: {args.notebook}: {ncode} code cells in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
