# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tiny subprocess + JSON-load helpers shared by drivers.

Drivers are self-contained per the design — but every one of them
shells out to a tool and reads back a JSON file. Keeping that in one
place avoids five copies of the same try/except.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def run_command(
    cmd: Sequence[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
) -> int:
    """Run ``cmd`` streaming output to logger; return exit code."""
    logger.info("Executing: %s", " ".join(str(c) for c in cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run(
        list(cmd),
        env=full_env,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    return proc.returncode


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning("Expected result file missing: %s", path)
        return None
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


def find_first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None
