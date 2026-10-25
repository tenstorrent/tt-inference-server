# SPDX-License-Identifier: Apache-2.0
"""Compact wall-time receipts; never device time or raw request content."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time


@contextmanager
def timed_attempt(path, task, phase, *, enabled=True):
    receipt = {"task_name": task, "phase": phase, "rc": None}
    if not enabled:
        yield receipt
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt.update(started_at=datetime.now(timezone.utc).isoformat(),
                   finished_at=None, elapsed_seconds=None, actual_hours=None)
    started = time.perf_counter()
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        yield receipt
    except BaseException as error:
        receipt["exception_type"] = type(error).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=elapsed, actual_hours=elapsed / 3600)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
