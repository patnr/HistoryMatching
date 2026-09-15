"""Compare two `pip freeze` outputs: fail if any pre-existing package changed.

Usage: check_freeze.py BEFORE.txt AFTER.txt

Rationale: on Colab, packages present at kernel start are already imported;
re-installing them would require a runtime restart. `colab_bootstrap.sh` must
therefore only *add* packages. This check makes that contract testable.
"""

import sys


def parse(path):
    pkgs = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("==", " @ "):
            if sep in line:
                name, ver = line.split(sep, 1)
                break
        else:
            name, ver = line, ""
        pkgs[name.lower().replace("_", "-")] = ver
    return pkgs


def main(before_path, after_path):
    before, after = parse(before_path), parse(after_path)
    changed = {k: (v, after.get(k)) for k, v in before.items() if after.get(k) != v}
    added = sorted(set(after) - set(before))
    print(f"Added packages ({len(added)}):", ", ".join(added) or "none")
    if changed:
        print("FAIL: preinstalled packages were modified (=> Colab would need a restart):")
        for k, (old, new) in sorted(changed.items()):
            print(f"  {k}: {old} -> {new if new is not None else '(removed)'}")
        return 1
    print("OK: no preinstalled package was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
