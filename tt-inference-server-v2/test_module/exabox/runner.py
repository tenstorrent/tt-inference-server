# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Drive the exabox shell benchmark suites against a running server.

Each suite under this directory is self-contained (``run.sh`` +
``requirements.txt`` + ``defaults.env``) and dispatched through
``run_test.sh``, which provisions a per-suite uv venv, waits for the
server, and snapshots ``/info``. Suite knobs (DURATION,
TARGET_CONCURRENCY, ...) are read from the environment — see each
suite's ``defaults.env``.

Result JSONs land under ``<ctx.output_path>/exabox/<suite>/``; one
``Block(kind="exabox")`` per suite points at them so the unified report
records the run. Forwarded to
:func:`workflow_module.accept_blocks` like the other llm_tests runners.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from report_module.schema import Block

logger = logging.getLogger(__name__)

_EXABOX_DIR = Path(__file__).resolve().parent
_DISPATCHER = _EXABOX_DIR / "run_test.sh"


@dataclass(frozen=True)
class ExaboxResult:
    test: str
    return_code: int
    elapsed_seconds: float


def available_tests() -> List[str]:
    return sorted(
        p.name
        for p in _EXABOX_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "run.sh").is_file()
    )


def run_exabox(ctx, tests: Optional[str] = None) -> List[ExaboxResult]:
    from workflow_module import accept_blocks

    known = available_tests()
    if tests:
        names = [t.strip().replace("-", "_") for t in tests.split(",") if t.strip()]
        unknown = sorted(set(names) - set(known))
        if unknown:
            raise ValueError(f"Unknown exabox test(s) {unknown}; available: {known}")
    else:
        names = known

    # ctx.base_url honors --server-url / RuntimeConfig.server_url (#4079).
    target = ctx.base_url
    results: List[ExaboxResult] = []
    blocks: List[Block] = []
    for i, name in enumerate(names, 1):
        out_dir = Path(ctx.output_path) / "exabox" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[exabox] Running %d/%d: %s -> %s", i, len(names), name, target)
        started = time.time()
        proc = subprocess.run(
            [
                str(_DISPATCHER),
                "--test",
                name,
                "--target",
                target,
                "--output-dir",
                str(out_dir),
                "--job-suffix",
                "v2",
            ]
        )
        elapsed = time.time() - started
        if proc.returncode != 0:
            logger.error(
                "[exabox] %s failed (rc=%d, %.1fs); continuing.",
                name,
                proc.returncode,
                elapsed,
            )
        results.append(ExaboxResult(name, proc.returncode, elapsed))
        blocks.append(
            Block(
                kind="exabox",
                id=name,
                title=f"Exabox {name.replace('_', ' ')}",
                task_type="exabox",
                data={
                    "test": name,
                    "return_code": proc.returncode,
                    "elapsed_seconds": round(elapsed, 1),
                    "results_dir": str(out_dir),
                    "result_files": sorted(
                        f.name for f in out_dir.iterdir() if f.is_file()
                    ),
                },
            )
        )

    accept_blocks(
        blocks,
        envelope={
            "model_name": getattr(ctx.model_spec, "model_name", ""),
            "device": ctx.device.name
            if hasattr(ctx.device, "name")
            else str(ctx.device),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return results


__all__ = ["ExaboxResult", "available_tests", "run_exabox"]
