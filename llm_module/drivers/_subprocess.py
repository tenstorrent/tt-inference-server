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
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from utils.model_naming import slugify_model_id

logger = logging.getLogger(__name__)


def run_command(
    cmd: Sequence[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    timeout_s: Optional[float] = None,
    heartbeat_s: Optional[float] = None,
    heartbeat_label: str = "",
    heartbeat_status: Optional[Callable[[], str]] = None,
) -> int:
    """Run ``cmd`` streaming output to logger; return exit code.

    If ``timeout_s`` is set and elapses before the process exits, the
    child is killed and 124 is returned (matching ``/usr/bin/timeout``)
    so callers can treat it as a normal nonzero exit and move on.

    If ``heartbeat_s`` is set, an INFO line is logged every ``heartbeat_s``
    seconds while the child runs, so long benchmark runs are visible in CI
    logs. ``heartbeat_status`` may return extra progress text for that line.
    """
    logger.info("Executing: %s", " ".join(str(c) for c in cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    if not heartbeat_s:
        try:
            proc = subprocess.run(
                list(cmd),
                env=full_env,
                cwd=str(cwd) if cwd else None,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Command exceeded timeout of %.0fs and was killed: %s",
                timeout_s,
                " ".join(str(c) for c in cmd),
            )
            return 124
        return proc.returncode

    start = time.monotonic()
    proc = subprocess.Popen(list(cmd), env=full_env, cwd=str(cwd) if cwd else None)
    while True:
        elapsed = time.monotonic() - start
        wait_s = heartbeat_s
        if timeout_s is not None:
            wait_s = min(wait_s, max(timeout_s - elapsed, 0.0))
        try:
            return proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            if timeout_s is not None and elapsed >= timeout_s:
                proc.kill()
                proc.wait()
                logger.error(
                    "Command exceeded timeout of %.0fs and was killed: %s",
                    timeout_s,
                    " ".join(str(c) for c in cmd),
                )
                return 124
            status = ""
            if heartbeat_status is not None:
                try:
                    status = heartbeat_status()
                except Exception as exc:  # noqa: BLE001 - progress text is best effort
                    status = f"(status unavailable: {exc})"
            logger.info(
                "%sstill running: %.0fs elapsed%s%s",
                f"[{heartbeat_label}] " if heartbeat_label else "",
                elapsed,
                f" of {timeout_s:.0f}s timeout" if timeout_s is not None else "",
                f"; {status}" if status else "",
            )


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
