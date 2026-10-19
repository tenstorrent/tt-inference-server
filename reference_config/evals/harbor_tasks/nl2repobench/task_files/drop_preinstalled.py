#!/usr/bin/env python3
"""Delete installed copies of the project under test outside /workspace.

Some official images ship the real library in site-packages, and some test
commands install it as a dependency (more-Itertools); tests that fail to import
the agent's package from /workspace then silently pass against it. Run at image
build and again before every pytest command. Copies under /workspace (the
agent's own, e.g. `pip install -e .`) are kept.
usage: drop_preinstalled.py <module> ...
"""

import importlib
import importlib.util
import os
import shutil
import sys


def outside_workspace(name):
    sys.path[:] = [p for p in sys.path if p and not p.startswith("/workspace")]
    importlib.invalidate_caches()
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return []
    if spec is None:
        return []
    found = []
    for loc in list(spec.submodule_search_locations or []) or [spec.origin]:
        # Inside a zipped .egg the module path itself does not exist on disk.
        egg = loc or ""
        while egg != os.path.dirname(egg) and not egg.endswith(".egg"):
            egg = os.path.dirname(egg)
        target = egg if egg.endswith(".egg") else loc
        if target and os.path.exists(target) and not os.path.realpath(target).startswith("/workspace"):
            found.append(target)
    return found


def main():
    sys.path.pop(0)
    for name in sys.argv[1:]:
        for target in outside_workspace(name):
            print("drop_preinstalled: removing %s (%s)" % (target, name))
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
        if outside_workspace(name):
            raise SystemExit("drop_preinstalled: %s still importable" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
