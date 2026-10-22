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
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.model_naming import slugify_model_id

logger = logging.getLogger(__name__)


def run_command(
    cmd: Sequence[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    timeout_s: Optional[float] = None,
) -> int:
    """Run ``cmd`` streaming output to logger; return exit code.

    If ``timeout_s`` is set and elapses before the process exits, the
    child is killed and 124 is returned (matching ``/usr/bin/timeout``)
    so callers can treat it as a normal nonzero exit and move on.

    The tool runs in its own process group, and the whole group is killed
    when it returns or times out. aiperf runs its workers as separate
    processes: killing only the parent left them sending requests and
    holding the caller's output pipes open, so the workflow never exited.
    """
    logger.info("Executing: %s", " ".join(str(c) for c in cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.Popen(
        list(cmd),
        env=full_env,
        cwd=str(cwd) if cwd else None,
        start_new_session=True,
    )
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.error(
            "Command exceeded timeout of %.0fs and was killed: %s",
            timeout_s,
            " ".join(str(c) for c in cmd),
        )
        return 124
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


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


def safe_filename_part(text: str) -> str:
    """Sanitize ``text`` for embedding in a filename.

    Thin alias for the canonical escape in :mod:`utils.model_naming`; the
    values passed here are model ids, so they must land on the same token as
    every other name derived from a model.
    """
    return slugify_model_id(text)
