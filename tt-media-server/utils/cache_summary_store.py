# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Per-job step-caching summaries, from the device worker to the job API.

A video's result reaches the API process as its frames or mp4 path only, so
the runner leaves the pipeline's ``cache_summary()`` (per expert: steps run and
steps served from the cache) in a small JSON file named by the request's task
id, which is also its job id. ``GET /v1/videos/generations/{id}`` returns it as
``cache_summary``. Both processes run in the same container.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

_CACHE_SUMMARY_DIR = Path(
    os.environ.get(
        "TT_CACHE_SUMMARY_DIR",
        os.path.join(tempfile.gettempdir(), "tt_cache_summary"),
    )
)


def _path(task_id: str) -> Optional[Path]:
    # Job ids are uuids; anything else could name a path outside the folder.
    if not task_id or not all(c.isalnum() or c in "-_" for c in task_id):
        return None
    return _CACHE_SUMMARY_DIR / f"{task_id}.json"


def _plain(value: Any) -> Any:
    """numpy scalars and arrays, as JSON numbers and lists."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_cache_summary(task_id: str, summary: list[dict]) -> None:
    """Keep one generation's cache summary for its job (an empty list: uncached)."""
    path = _path(task_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, default=_plain))
    temp.replace(path)


def read_cache_summary(task_id: str) -> Optional[list[dict]]:
    """The job's cache summary, or None when its runner left none."""
    path = _path(task_id)
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def delete_cache_summary(task_id: str) -> None:
    path = _path(task_id)
    if path is not None:
        path.unlink(missing_ok=True)
