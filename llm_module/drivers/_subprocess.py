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

from utils.model_naming import slugify_model_id

logger = logging.getLogger(__name__)

# Flags whose value is a credential. The command is logged verbatim otherwise,
# which writes the token into every run log and the CI artifacts built from it.
_SECRET_FLAGS = frozenset({"--api-key"})
_REDACTED = "<redacted>"


def _redacted_command(cmd: Sequence[str]) -> str:
    """``cmd`` joined for logging, with credential flag values masked."""
    parts: List[str] = []
    mask_next = False
    for raw in cmd:
        arg = str(raw)
        if mask_next:
            parts.append(_REDACTED)
            mask_next = False
            continue
        flag, sep, _ = arg.partition("=")
        if flag in _SECRET_FLAGS:
            if sep:
                parts.append(f"{flag}={_REDACTED}")
            else:
                parts.append(arg)
                mask_next = True
            continue
        parts.append(arg)
    return " ".join(parts)


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
    """
    logger.info("Executing: %s", _redacted_command(cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
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
            _redacted_command(cmd),
        )
        return 124
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


def safe_filename_part(text: str) -> str:
    """Sanitize ``text`` for embedding in a filename.

    Thin alias for the canonical escape in :mod:`utils.model_naming`; the
    values passed here are model ids, so they must land on the same token as
    every other name derived from a model.
    """
    return slugify_model_id(text)
