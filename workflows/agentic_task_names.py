# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parse an explicit agentic task-name list (``run.py --agentic-task-names``).

Kept dependency-free: run.py validates the value before any venv exists, and
``report_module.merge_agentic_runs --expected-tasks`` reads the same format.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def parse_task_names(value: str) -> List[str]:
    """``a,b,c`` or ``@file`` (one name per line) -> the task names, in order.

    Blank lines and ``#`` comments in a file are skipped. Raises ValueError on
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
