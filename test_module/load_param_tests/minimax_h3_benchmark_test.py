# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The h3-benchmark generation matrix as a workflow spec test.

For the task this deployment serves: a read-only health pre-gate (the deployment
must be ready, idle, not wedged), the probe checks (routes, gate, key, pinned
assets), one smoke clip, then the planned cases -- 1 warmup + N measured runs each,
every counting clip judged against the output contract. Rows land in
``results.jsonl`` / ``results.csv`` under the run's output directory, exactly as
``quad-agent/h3-benchmark`` writes them, so its ``report`` and ``gaps`` read them.

Targets (video.json): ``task`` (t2va|fl2va|ref2va), ``plan_ci`` and ``plan_full``
(lists of ``{"cases": [...], "runs": N}``; the CI plan is used under
``--ci-mode``), ``timeout_table`` (BH1X), ``idle_wait_s``, ``target_times_s``
(per case, informational unless ``enforce_timing``), ``assets_dir``,
``verify_manifest``, ``force``, ``continue_after_timeout``, ``combo``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_bench import adapters as A
from test_module._test_common.minimax_h3_bench import host as H
from test_module._test_common.minimax_h3_bench import judge as J
from test_module._test_common.minimax_h3_bench import models as M
from test_module._test_common.minimax_h3_bench import runner as R

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_TEST_TIMEOUT_SECONDS = 14400
DEFAULT_PLAN_CI = [
    {"cases": ["T2VA-L"], "runs": 3},
    {"cases": ["T2VA-M", "T2VA-H"], "runs": 1},
]
DEFAULT_PLAN_FULL = [{"cases": ["T2VA-L", "T2VA-M", "T2VA-H"], "runs": 3}]
TIMING_FAIL_RATIO = 1.25  # median above target x this fails when timing is enforced


def _ci_mode(ctx) -> bool:
    rc = getattr(ctx, "runtime_config", None) if ctx is not None else None
    if rc is None:
        return True
    return bool(
        getattr(rc, "ci_mode", False)
        or getattr(rc, "limit_samples_mode", None) == "ci-nightly"
    )


def run_benchmark(
    *,
    base_url: str,
    task: str = "t2va",
    plan: list | None = None,
    out_dir: str,
    assets_dir: str | None = None,
    timeout_table: str = M.DEFAULT_TIMEOUT_TABLE,
    idle_wait_s: int = H.IDLE_WAIT_S,
    target_times_s: dict | None = None,
    enforce_timing: bool = False,
    verify_manifest: bool = True,
    force: bool = False,
    continue_after_timeout: bool = False,
    combo: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    plan = plan or DEFAULT_PLAN_CI
    target_times_s = target_times_s or {}
    M.configure(assets_dir=H.resolve_assets_dir(assets_dir), out_dir=out_dir)
    M.ensure_dirs()
    cases_cfg = M.load_cases()
    by_id = {c["id"]: c for c in cases_cfg["cases"]}
    adapter = A.TenstorrentH3({task: base_url}, api_key=api_key)
    tracker = H.JobTracker(adapter)
    combo = combo or f"BH1X-{H.short_name(base_url)}"
    cfg = {
        "engine": "minimax-h3 (tt-media-server)",
        "hw": "Blackhole Galaxy 1x",
        "gpus": 1,
        "node": f"{task}={base_url}",
    }
    started = time.time()
    result: dict[str, Any] = {
        "task_name": "minimax_h3_benchmark", "base_url": base_url, "task": task, "combo": combo,
        "timeout_table": timeout_table, "assets_dir": M.assets_dir(), "out_dir": out_dir,
        "plan": plan, "pregate": [], "probe": [], "smoke": None, "cases": [], "leftover_jobs": [],
        "success": False,
    }  # fmt: skip
    M.log(
        f"=== minimax_h3_benchmark {combo}: task={task} plan={json.dumps(plan)} table={timeout_table}"
    )

    # 1. selection sanity + assets
    selected = []
    for item in plan:
        for cid in item["cases"]:
            if cid not in by_id:
                raise ValueError(
                    f"unknown case {cid!r}; cases.json has {sorted(by_id)}"
                )
            if by_id[cid]["task"] != task:
                raise ValueError(
                    f"case {cid} is a {by_id[cid]['task']} case; this deployment serves {task}"
                )
            selected.append((by_id[cid], int(item["runs"])))
    needed = set(M.case_assets(H.smoke_case(task, cases_cfg)))
    for case, _ in selected:
        needed.update(M.case_assets(case))
    asset_problems = (
        M.verify_assets(needed)
        if verify_manifest
        else [
            f"missing asset {n}"
            for n in sorted(needed)
            if not os.path.exists(os.path.join(M.assets_dir(), n))
        ]
    )
    result["probe"] += asset_problems

    # 2. discovery, pre-gate, probe
    ep = H.discover(adapter, task)
    result["endpoint"] = {
        "url": ep.url, "reachable": ep.reachable, "health": ep.health, "liveness": ep.liveness,
        "gates": ep.gates, "auth_required": ep.auth_required, "key": ep.key_detail,
    }  # fmt: skip
    block = H.device_block(ep)
    if block:
        result["pregate"].append(block)
    else:
        result["pregate"] += H.pregate(adapter, ep, idle_wait_s)
        result["probe"] += H.probe_checks(ep)
    result["mesh"] = (
        ep.liveness.get("device_mesh_shape") if isinstance(ep.liveness, dict) else None
    )
    if result["pregate"] or result["probe"]:
        for p in result["pregate"] + result["probe"]:
            M.log(f"  [pregate/probe] {p}")
        result["summary"] = "deployment not fit to measure; no generation was started"
        result["elapsed_seconds"] = round(time.time() - started, 1)
        return result

    try:
        # 3. smoke
        smoke = H.smoke_case(task, cases_cfg)
        rec = R.run_once(
            adapter,
            combo,
            cfg,
            smoke,
            "smoke",
            measured=False,
            timeout_table=timeout_table,
        )
        stop = H.after_run(adapter, tracker, ep, rec, continue_after_timeout)
        problems, notes = (
            J.run_problems(rec, smoke)
            if rec.get("outcome") == "ok"
            else ([R.why(rec)], [])
        )
        result["smoke"] = {"case": smoke["id"], "outcome": rec.get("outcome"), "gen_s": rec.get("gen_s"),
                           "queue_s": rec.get("queue_s"), "problems": problems, "notes": notes, "stop": stop}  # fmt: skip
        with open(os.path.join(out_dir, f"smoke_status_{combo}_{task}.log"), "w") as fh:
            fh.write(
                ("ok" if rec.get("outcome") == "ok" and not problems else "failed")
                + "\n"
            )
            fh.write(f"outcome={rec.get('outcome')} problems={problems}\n")

        # 4. the plan
        for case, runs in selected:
            if ep.stopped:
                result["cases"].append({"case": case["id"], "task": task, "status": "skipped",
                                        "stop": ep.stopped, "runs_requested": runs, "runs_ok": 0})  # fmt: skip
                continue
            outcome = H.bench_case(
                adapter,
                tracker,
                ep,
                combo,
                cfg,
                case,
                runs,
                timeout_table,
                force,
                continue_after_timeout,
            )
            entry = H.case_result(combo, case, runs, outcome)
            target = target_times_s.get(case["id"])
            if target and entry.get("median_s") is not None:
                entry["target_s"] = target
                entry["timing_ok"] = entry["median_s"] <= target * TIMING_FAIL_RATIO
                if enforce_timing and not entry["timing_ok"]:
                    entry["status"] = "fail"
                    entry["problems"].append(
                        f"median {entry['median_s']:.1f}s > {TIMING_FAIL_RATIO}x target {target}s"
                    )
            result["cases"].append(entry)
    finally:
        result["leftover_jobs"] = tracker.cancel_leftovers()

    statuses = [c["status"] for c in result["cases"]]
    smoke_ok = (
        bool(result["smoke"])
        and result["smoke"]["outcome"] == "ok"
        and not result["smoke"]["problems"]
    )
    result["success"] = (
        smoke_ok
        and not ep.stopped
        and all(s in ("pass", "xfail") for s in statuses)
        and not result["leftover_jobs"]
    )
    medians = [c["median_s"] for c in result["cases"] if c.get("median_s") is not None]
    result["summary"] = (
        f"smoke {'ok' if smoke_ok else 'FAILED'}; "
        f"{statuses.count('pass')} pass / {statuses.count('xfail')} xfail / {statuses.count('fail')} fail / "
        f"{statuses.count('skipped')} skipped of {len(statuses)} cases"
        + (f"; median gen {statistics.median(medians):.1f}s" if medians else "")
        + (f"; host stopped: {ep.stopped}" if ep.stopped else "")
    )
    result["stopped"] = ep.stopped
    result["elapsed_seconds"] = round(time.time() - started, 1)
    result["artifacts"] = {
        "results_jsonl": M.results_path(), "results_csv": M.results_csv_path(),
        "run_log": os.path.join(out_dir, "run.log"), "clips_dir": M.clips_dir(),
    }  # fmt: skip
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    M.log(f"=== {combo}: {result['summary']}")
    return result


class MiniMaxH3BenchmarkTest(BaseTest):
    """h3-benchmark cases against the deployment this workflow started."""

    KIND = "minimax_h3_benchmark"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    def _plan(self) -> list:
        if _ci_mode(self.ctx):
            return list(self.targets.get("plan_ci") or DEFAULT_PLAN_CI)
        return list(self.targets.get("plan_full") or DEFAULT_PLAN_FULL)

    def _out_dir(self) -> str:
        if self.targets.get("out_dir"):
            return str(self.targets["out_dir"])
        if self.ctx is not None:
            return str(Path(self.ctx.output_path) / "minimax_h3_bench")
        return "/tmp/minimax_h3_bench"

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await asyncio.to_thread(
            run_benchmark,
            base_url=self.base_url,
            task=str(self.targets.get("task", "t2va")),
            plan=self._plan(),
            out_dir=self._out_dir(),
            assets_dir=self.targets.get("assets_dir"),
            timeout_table=str(
                self.targets.get("timeout_table", M.DEFAULT_TIMEOUT_TABLE)
            ),
            idle_wait_s=int(self.targets.get("idle_wait_s", H.IDLE_WAIT_S)),
            target_times_s=dict(self.targets.get("target_times_s") or {}),
            enforce_timing=bool(self.targets.get("enforce_timing", False)),
            verify_manifest=bool(self.targets.get("verify_manifest", True)),
            force=bool(self.targets.get("force", False)),
            continue_after_timeout=bool(
                self.targets.get("continue_after_timeout", False)
            ),
            combo=self.targets.get("combo"),
        )


def run_minimax_h3_benchmark(
    ctx: MediaContext, targets: dict[str, Any] | None = None
) -> Block:
    return MiniMaxH3BenchmarkTest(
        TestConfig(
            {
                "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        targets or {},
        ctx=ctx,
    ).run_tests()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run the MiniMax-H3 benchmark cases against a deployment."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task", default="t2va", choices=list(A.TASKS))
    parser.add_argument("--cases", default="T2VA-L", help="comma-separated case ids")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="/tmp/minimax_h3_bench")
    parser.add_argument("--assets")
    parser.add_argument(
        "--timeout-table",
        default=M.DEFAULT_TIMEOUT_TABLE,
        choices=sorted(M.TIMEOUT_TABLE_S),
    )
    parser.add_argument(
        "--no-manifest", action="store_true", help="do not verify asset hashes"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-after-timeout", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = run_benchmark(
        base_url=args.base_url, task=args.task,
        plan=[{"cases": [c.strip() for c in args.cases.split(",") if c.strip()], "runs": args.runs}],
        out_dir=args.out, assets_dir=args.assets, timeout_table=args.timeout_table,
        verify_manifest=not args.no_manifest, force=args.force, continue_after_timeout=args.continue_after_timeout,
    )  # fmt: skip
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "plan"}, indent=2, default=str
        )
    )
    return 0 if result.get("success") else 1


__all__ = ["MiniMaxH3BenchmarkTest", "run_benchmark", "run_minimax_h3_benchmark"]

if __name__ == "__main__":
    sys.exit(main())
