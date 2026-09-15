"""Execute a notebook headlessly, as a smoke test.

Usage: run_nb.py NOTEBOOK.ipynb [--timeout SECONDS] [--skip-bootstrap] [--out PATH] [-v]

- `--skip-bootstrap` removes the cell that runs `colab_bootstrap.sh` (so that
  a test of a local checkout does not fetch and install `master` on top).
- Fails (exit 1) on the first cell error or timeout, printing that cell's source.
- `-v` prints each code cell (index, first line, duration) as it runs.

Runs both locally (`uv run tests/colab/run_nb.py notebooks/HistoryMatch.ipynb`)
and inside the Colab runtime image (see smoke.sh).

Why not `nbclient` (or `jupyter nbconvert --execute`, `papermill`, which wrap it)?
Its asyncio message loop intermittently misses the execute-reply of cells that
emit a burst of ipywidgets comm messages (our `interact`/`toggle_items` cells),
so such cells stall for the full per-cell timeout, ~50% of the time (seen 2026-09
with nbclient 0.11, jupyter_client 8.10, pyzmq 25-27, ipykernel 6 and 7).
`jupyter_client`'s blocking client polls synchronously and does not have this.
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client.manager import start_new_kernel
from nbformat.v4 import output_from_msg


class CellFailed(Exception):
    pass


def run_cell(kc, cell, timeout):
    """Execute one code cell; collect its outputs; raise `CellFailed` on error/timeout."""
    msg_id = kc.execute(cell.source)
    cell.outputs = []
    deadline = time.monotonic() + timeout
    reply = idle = False
    while not (reply and idle):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CellFailed(f"Timed out after {timeout}s")
        # Drain iopub (outputs, status), then check for the shell reply.
        try:
            msg = kc.get_iopub_msg(timeout=min(remaining, 0.1))
        except Exception:  # queue.Empty
            msg = None
        if msg and msg["parent_header"].get("msg_id") == msg_id:
            mt = msg["msg_type"]
            if mt == "status":
                idle = msg["content"]["execution_state"] == "idle"
            elif mt in ("stream", "display_data", "execute_result", "error"):
                cell.outputs.append(output_from_msg(msg))
            elif mt == "clear_output":
                cell.outputs = []
        if not reply:
            try:
                r = kc.get_shell_msg(timeout=0)
            except Exception:
                r = None
            if r and r["parent_header"].get("msg_id") == msg_id:
                reply = True
                if r["content"]["status"] == "error":
                    c = r["content"]
                    raise CellFailed(f"{c.get('ename')}: {c.get('evalue')}\n" + "\n".join(c.get("traceback", [])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout (s)")
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--out", type=Path, help="write the executed notebook here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each cell as it runs")
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    if args.skip_bootstrap:
        n = len(nb.cells)
        nb.cells = [c for c in nb.cells if "colab_bootstrap.sh" not in c.source]
        print(f"Skipped {n - len(nb.cells)} bootstrap cell(s).")

    km, kc = start_new_kernel(kernel_name="python3", cwd=str(args.notebook.parent.resolve()))
    t0 = time.time()
    status = 0
    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if args.verbose:
                head = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""
                print(f"[{i:3}] {head}", end=" ", flush=True)
            t1 = time.time()
            try:
                run_cell(kc, cell, args.timeout)
            except CellFailed as e:
                print(f"\nFAIL: {args.notebook} at cell {i} after {time.time() - t0:.0f}s: {e}")
                print("-" * 19 + f"\n{cell.source}\n" + "-" * 19)
                status = 1
                break
            if args.verbose:
                print(f"{time.time() - t1:6.1f}s", flush=True)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
        if args.out:
            nbformat.write(nb, args.out)
    if status == 0:
        ncode = sum(c.cell_type == "code" for c in nb.cells)
        print(f"OK: {args.notebook}: {ncode} code cells in {time.time() - t0:.0f}s")
    return status


if __name__ == "__main__":
    sys.exit(main())
