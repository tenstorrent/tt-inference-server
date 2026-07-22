# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""sys.path bootstrap for entry-point scripts run from a base Python.

Importing this module inserts the repo root onto ``sys.path`` so that
``import workflows`` / ``import report_module`` resolve, and exposes
``REPO_ROOT`` for repo-root-relative paths.

This is the one bootstrap seed for its directory; sibling scripts do
``from _bootstrap import REPO_ROOT`` (or ``import _bootstrap``) instead of
repeating the snippet. The seed can only live here (not in an importable
package) because it is the very thing that puts the repo on ``sys.path`` --
importing a helper to do it would need the path set already.

Same-directory import ONLY (``import _bootstrap``). A module that is also
imported via its package path (e.g. ``from scripts.release.foo import ...`` in
a test, where the script's directory is not on ``sys.path``) must NOT use this
-- a top-level ``import _bootstrap`` fails there; such modules inline their own
bootstrap instead.

This file's content is identical in every directory that needs a seed: it
locates the repo root by marker, so it does not depend on its own depth.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root = nearest ancestor containing a repo marker. Marker-based (not a
# fixed parent depth) so this file is identical wherever it lives and survives
# directory moves; works without a .git dir (e.g. exported artifacts) via the
# run_workflows.py entry point.
_MARKERS = (".git", "run_workflows.py", "VERSION")
REPO_ROOT = next(
    (
        p
        for p in Path(__file__).resolve().parents
        if any((p / m).exists() for m in _MARKERS)
    ),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError(
        f"Could not locate repo root from {__file__} (markers: {_MARKERS})"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
