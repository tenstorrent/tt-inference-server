# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MiniMax-H3 stress run: the vendored h3-stress-kit driven against this job's deployment.

The kit walks baseline, validate, sustain, burst, recovery and edge phases with its own cases
and media (test_fixtures/h3_stress_kit/README.md). It always exits 0, so the verdict is read
from its per-request rows and the liveness it records at the end.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from report_module.schema import Block
from report_module.status import TestStatus

from .._test_common import block_id
from ..context import MediaContext

logger = logging.getLogger(__name__)

KIT_DIR = Path(__file__).resolve().parents[2] / "test_fixtures" / "h3_stress_kit"
H3_STRESS_MINUTES = 30
# The kit's per-job timeout; in-flight jobs may run this long past the phase budget.
_KIT_JOB_TIMEOUT_S = 600
_REPORT_MARGIN_S = 600
# Served model name (the MODEL env) -> the kit's server key, which selects its cases.
STRESS_SERVERS = {
    "MiniMax-H3-FL2VA-LightX2V": "fl2va",
    "MiniMax-H3-Ref2VA-LightX2V": "ref2va",
}
# The key the CI deployment accepts; the other H3 video tests send the same one.
_API_KEY = "your-secret-key"


def build_stress_command(
    model_name: str, base_url: str, out_dir: Path, minutes: int
) -> tuple[list[str], dict]:
    server = STRESS_SERVERS[model_name]
    argv = [
        sys.executable,
        str(KIT_DIR / "h3_stress.py"),
        "--servers",
        server,
        "--minutes",
        str(minutes),
        "--out",
        str(out_dir),
    ]
    env = dict(os.environ)
    # Both URLs are set so the kit's defaults (remote test deployments) can never be used.
    env.update(H3_FL2VA_URL=base_url, H3_REF2VA_URL=base_url, H3_API_KEY=_API_KEY)
    return argv, env


def judge(rows: list, run_meta: dict, server: str) -> dict:
    """Pass when every expected-ok request passed, no request got a 5xx, and the server ended
    alive, ready and without a worker restart. Accepted reject cases are reported only."""
    rows = [r for r in rows if r.get("kind") == "request" and r.get("server") == server]
    ok_rows = [r for r in rows if r.get("expect") == "ok"]
    ok_passed = sum(1 for r in ok_rows if r.get("pass") is True)
    http_5xx = sum(1 for r in rows if (r.get("http_code") or 0) >= 500)
    rejects_accepted = sorted(
        {
            r["case_id"]
            for r in rows
            if r.get("expect") == "reject" and r.get("pass") is False
        }
    )

    end = ((run_meta.get("servers_end") or {}).get(server) or {}).get("liveness") or {}
    start = ((run_meta.get("servers_start") or {}).get(server) or {}).get(
        "liveness"
    ) or {}

    reasons = []
    if not ok_rows:
        reasons.append("no expected-ok request was sent")
    elif ok_passed != len(ok_rows):
        reasons.append(f"{ok_passed}/{len(ok_rows)} expected-ok requests passed")
    if http_5xx:
        reasons.append(f"{http_5xx} requests got a 5xx")
    if end.get("http_code") != 200 or end.get("model_ready") is not True:
        reasons.append(f"server not ready at the end: {end}")
    elif end.get("worker_alive") is False:
        reasons.append("worker not alive at the end")
    if (end.get("restart_count") or 0) > (start.get("restart_count") or 0):
        reasons.append(
            f"worker restarted: {start.get('restart_count')} -> {end.get('restart_count')}"
        )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "requests": len(rows),
        "ok_sent": len(ok_rows),
        "ok_passed": ok_passed,
        "http_5xx": http_5xx,
        "rejects_accepted": rejects_accepted,
    }


def _load_run(out_root: Path) -> tuple[Path, list, dict, dict]:
    (run_dir,) = [p for p in out_root.iterdir() if p.is_dir()]
    rows = []
    results = run_dir / "results.jsonl"
    if results.is_file():
        rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    run_meta = json.loads((run_dir / "run.json").read_text())
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    return run_dir, rows, run_meta, summary


def run_minimax_h3_stress(ctx: MediaContext) -> Block:
    model_name = ctx.model_spec.model_name
    server = STRESS_SERVERS[model_name]
    out_root = Path(ctx.output_path) / "h3_stress"
    out_root.mkdir(parents=True, exist_ok=True)
    argv, env = build_stress_command(
        model_name, ctx.base_url, out_root, H3_STRESS_MINUTES
    )
    timeout = H3_STRESS_MINUTES * 60 + _KIT_JOB_TIMEOUT_S + _REPORT_MARGIN_S
    logger.info(f"H3 stress ({server}, {H3_STRESS_MINUTES} min): {' '.join(argv)}")

    block_data: dict = {"Benchmarks": {"server": server, "minutes": H3_STRESS_MINUTES}}
    try:
        subprocess.run(argv, env=env, cwd=KIT_DIR, timeout=timeout, check=False)
        run_dir, rows, run_meta, summary = _load_run(out_root)
        verdict = judge(rows, run_meta, server)
        block_data["Benchmarks"].update(
            verdict,
            results_dir=str(run_dir),
            summary_servers=summary.get("servers"),
            summary_phases=summary.get("phases"),
            summary_monitor=summary.get("monitor"),
            findings=summary.get("findings"),
        )
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        verdict = {"passed": False, "reasons": [f"{type(e).__name__}: {e}"]}
        block_data["Benchmarks"].update(verdict)

    if not verdict["passed"]:
        logger.error(f"H3 stress FAILED: {verdict['reasons']}")
        block_data["status"] = TestStatus.FAIL.value

    return Block(
        kind="benchmarks",
        task_type="video",
        title="MiniMax-H3 Stress",
        id=block_id(ctx) or None,
        targets={"minutes": H3_STRESS_MINUTES},
        data=block_data,
    )


__all__ = ["run_minimax_h3_stress"]
