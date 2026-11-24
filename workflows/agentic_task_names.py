# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parse an explicit agentic task-name list (``run.py --agentic-task-names``).

Kept dependency-free: run.py validates the value before any venv exists, and
``report_module.merge_agentic_runs --expected-tasks`` reads the same format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def parse_task_names(value: str) -> List[str]:
    """``a,b,c`` or ``@file`` -> the task names, in order.

    The file holds one name per line (blank lines and ``#`` comments are
    skipped) or a JSON list of names, as ``report_module.list_agentic_tasks``
    writes. Raises ValueError on
    an empty list, an empty comma-separated entry, or a repeated name (a task
    listed twice would run twice and be scored twice).
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task name list is empty")
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read task name file {path}: {exc}") from exc
        if text.lstrip().startswith("["):
            try:
                names = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"{path}: not a JSON list: {exc}") from exc
            if not all(isinstance(name, str) and name.strip() for name in names):
                raise ValueError(f"{path}: expected a JSON list of task names")
            names = [name.strip() for name in names]
        else:
            names = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
    else:
        names = [name.strip() for name in value.split(",")]
        if not all(names):
            raise ValueError(
                f"empty task name in {value!r}: expected a comma-separated list"
            )
    if not names:
        raise ValueError(f"no task names in {value!r}")
    seen = set()
    repeated = sorted({name for name in names if name in seen or seen.add(name)})
    if repeated:
        raise ValueError(f"task names listed more than once: {repeated}")
    return names
