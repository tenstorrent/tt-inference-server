# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Drive the serving-bench shell benchmark suites against a running server.

Each suite under this directory is self-contained (``run.sh`` +
``requirements.txt`` + ``defaults.env``) and dispatched through
``run_test.sh``, which provisions a per-suite uv venv, waits for the
server, and snapshots ``/info``. Suite knobs (DURATION,
TARGET_CONCURRENCY, ...) are read from the environment — see each
suite's ``defaults.env``. ``--limit-samples-mode`` selects a knob
preset (see :mod:`presets`); a value already exported by the caller
still wins.

Result JSONs land under ``<ctx.output_path>/serving_bench/<suite>/``;
one ``Block(kind="serving_bench")`` per suite points at them so the
unified report records the run. Forwarded to
:func:`workflow_module.accept_blocks` like the other llm_tests runners.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from report_module.schema import Block

from .presets import preset_env_for_mode

logger = logging.getLogger(__name__)

_SERVING_BENCH_DIR = Path(__file__).resolve().parent
_DISPATCHER = _SERVING_BENCH_DIR / "run_test.sh"


@dataclass(frozen=True)
class ServingBenchResult:
    suite: str
    return_code: int
    elapsed_seconds: float


def available_suites() -> List[str]:
    return sorted(
        p.name
        for p in _SERVING_BENCH_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "run.sh").is_file()
    )


def run_serving_bench(ctx, suites: Optional[str] = None) -> List[ServingBenchResult]:
    from workflow_module import accept_blocks

    known = available_suites()
    if suites:
        names = [s.strip().replace("-", "_") for s in suites.split(",") if s.strip()]
        unknown = sorted(set(names) - set(known))
        if unknown:
            raise ValueError(
                f"Unknown serving-bench suite(s) {unknown}; available: {known}"
            )
    else:
        names = known

    # --limit-samples-mode preset: fill knobs the caller didn't export. An
    # explicit env var wins (we only set unset keys); defaults.env's := then
    # fills anything still unset. So: caller env > preset > suite defaults.
    limit_mode = getattr(
        getattr(ctx, "runtime_config", None), "limit_samples_mode", None
    )
    env = os.environ.copy()
    applied = {}
    for key, value in preset_env_for_mode(limit_mode).items():
        if key not in os.environ:
            env[key] = value
            applied[key] = value
    if applied:
        logger.info(
            "[serving_bench] limit-samples-mode=%s preset knobs: %s",
            limit_mode,
            applied,
        )

    # ctx.base_url honors --server-url / RuntimeConfig.server_url (#4079).
    target = ctx.base_url
    results: List[ServingBenchResult] = []
    blocks: List[Block] = []
    for i, name in enumerate(names, 1):
        out_dir = Path(ctx.output_path) / "serving_bench" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[serving_bench] Running %d/%d: %s -> %s", i, len(names), name, target
        )
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
            ],
            env=env,
        )
        elapsed = time.time() - started
        if proc.returncode != 0:
            logger.error(
                "[serving_bench] %s failed (rc=%d, %.1fs); continuing.",
                name,
                proc.returncode,
                elapsed,
            )
        results.append(ServingBenchResult(name, proc.returncode, elapsed))
        blocks.append(
            Block(
                kind="serving_bench",
                id=name,
                title=f"Serving bench {name.replace('_', ' ')}",
                task_type="serving_bench",
                data={
                    "suite": name,
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
            "model_name": getattr(ctx.model_spec, "hf_model_repo", "")
            or getattr(ctx.model_spec, "model_name", ""),
            "device": ctx.device.name
            if hasattr(ctx.device, "name")
            else str(ctx.device),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return results


__all__ = ["ServingBenchResult", "available_suites", "run_serving_bench"]
