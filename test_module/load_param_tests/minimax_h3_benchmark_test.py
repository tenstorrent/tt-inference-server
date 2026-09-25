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

Targets (video.json): ``task`` (t2va|fl2va|ref2va; optional -- the task comes from the
model spec's ``MODEL_RUNNER`` and a different ``task`` fails the test before anything is
sent), ``plan_ci`` and ``plan_full`` (lists of ``{"cases": [...], "runs": N}``; the CI plan
is used under ``--ci-mode``; the defaults cover every case of the task), ``timeout_table``
(BH1X), ``idle_wait_s``, ``target_times_s`` (per case, informational unless
``enforce_timing``), ``assets_dir``, ``verify_manifest``, ``allow_resume`` (default false:
CI never resumes from a previous results.jsonl), ``api_key``, ``continue_after_timeout``,
``combo``, ``out_dir``, ``out_subdir`` (this entry's directory under it; see ``_out_dir``)
and ``skip_smoke`` (default false: a later entry against the same deployment may skip it).

An entry's plan must fit its ``test_config.timeout`` (``plan_estimate_s``; cases not started
by the deadline are skipped and fail the test). At the BH1X pace the ref2va defaults need
~5.7 h (CI) and ~11 h (full), beyond the template's 4 h: raise the entry's timeout or split
the plan over entries (``skip_smoke`` + ``out_subdir`` on the later ones).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
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
from test_module._test_common.minimax_h3_client import resolve_server_api_key

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_TEST_TIMEOUT_SECONDS = 14400
# One keyframe at 5/10/15 s, then first + last keyframes at 5/10/15 s.
FL2VA_CASES = ["FL2VA-L", "FL2VA-M", "FL2VA-H1", "FL2VA-L2", "FL2VA-M2", "FL2VA-H"]
# Rising risk: the 5 s row, then 10 s and 15 s, and the max-inputs REF2VA-H family last: an
# out-of-memory there can leave the mesh degraded until the ranks restart, and must not cost
# the cheaper cases their verdict.
REF2VA_CASES = [
    "REF2VA-L", "REF2VA-M5", "SIZE-V",
    "REF2VA-L10", "REF2VA-M", "SIZE-V10",
    "REF2VA-L15", "REF2VA-M15", "SIZE-V15",
    "REF2VA-H5", "REF2VA-H10", "REF2VA-H",
]  # fmt: skip
# Per task, every case of that task in cases.json.
DEFAULT_PLAN_CI = {
    "t2va": [
        {"cases": ["T2VA-L"], "runs": 3},
        {"cases": ["T2VA-M", "T2VA-H"], "runs": 1},
    ],
    "fl2va": [
        {"cases": FL2VA_CASES[:1], "runs": 3},
        {"cases": FL2VA_CASES[1:], "runs": 1},
    ],
    "ref2va": [{"cases": REF2VA_CASES, "runs": 1}],
}
DEFAULT_PLAN_FULL = {
    "t2va": [{"cases": ["T2VA-L", "T2VA-M", "T2VA-H"], "runs": 3}],
    "fl2va": [{"cases": FL2VA_CASES, "runs": 3}],
    "ref2va": [{"cases": REF2VA_CASES, "runs": 3}],
}
RUNNER_TASKS = {f"tt-minimax-h3-{task}": task for task in A.TASKS}
TIMING_FAIL_RATIO = 1.25  # median above target x this fails when timing is enforced
# Each TIMEOUT_TABLE_S budget is this many times the slowest observed run of its shape.
BUDGET_FACTOR = 3
# Output directory -> the entry (targets + plan) that took it, for this process: another
# suite entry without out_subdir writes below it instead of over it.
_OUT_DIR_OWNERS: dict[str, str] = {}


def _ci_mode(ctx) -> bool:
    rc = getattr(ctx, "runtime_config", None) if ctx is not None else None
    if rc is None:
        return True
    return bool(
        getattr(rc, "ci_mode", False)
        or getattr(rc, "limit_samples_mode", None) == "ci-nightly"
    )


def deployment_task(ctx, requested: str | None = None) -> str:
    """The task to benchmark: the one the model spec's ``MODEL_RUNNER`` serves
    (tt-minimax-h3-t2va|fl2va|ref2va), else ``requested`` (``targets.task``), else t2va.
    A ``requested`` task the spec contradicts is a suite error, raised before anything runs."""
    if requested is not None and requested not in A.TASKS:
        raise ValueError(f"targets.task {requested!r} is not one of {list(A.TASKS)}")
    spec = getattr(ctx, "model_spec", None) if ctx is not None else None
    runner = (getattr(spec, "env_vars", None) or {}).get("MODEL_RUNNER")
    derived = RUNNER_TASKS.get(runner)
    if runner and derived is None and requested is None:
        raise ValueError(
            f"MODEL_RUNNER={runner!r} is not a MiniMax-H3 runner ({sorted(RUNNER_TASKS)}); "
            "set targets.task to say which task this deployment serves"
        )
    if derived and requested and requested != derived:
        raise ValueError(
            f"targets.task is {requested!r} but the model spec's MODEL_RUNNER={runner!r} "
            f"serves {derived}: this entry would benchmark the wrong task; drop targets.task "
            "or point the suite at the right deployment"
        )
    return derived or requested or "t2va"


def plan_estimate_s(
    task: str,
    plan: list,
    timeout_table: str = M.DEFAULT_TIMEOUT_TABLE,
    skip_smoke: bool = False,
) -> int:
    """Expected wall time of ``plan`` at the pace its budgets assume (budget / BUDGET_FACTOR per
    generation): the smoke clip unless skipped, then 1 warmup + N runs of every case."""
    cases_cfg = M.load_cases()
    by_id = {c["id"]: c for c in cases_cfg["cases"]}
    smoke = [] if skip_smoke else [(H.smoke_case(task, cases_cfg), 1)]
    gens = smoke + [
        (by_id[cid], 1 + int(item["runs"])) for item in plan for cid in item["cases"]
    ]
    total = sum(M.case_budget_s(case, timeout_table)[0] * n for case, n in gens)
    return round(total / BUDGET_FACTOR)


def _cases_slug(plan: list) -> str:
    ids = list(dict.fromkeys(cid for item in plan for cid in item["cases"]))
    slug = "_".join(ids)
    if len(slug) > 64:
        digest = hashlib.sha256(slug.encode()).hexdigest()[:8]
        slug = f"{ids[0]}_to_{ids[-1]}_{len(ids)}cases_{digest}"
    return slug


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
    deadline_s: float | None = None,
    skip_smoke: bool = False,
) -> dict[str, Any]:
    if task not in DEFAULT_PLAN_CI:
        raise ValueError(f"unknown task {task!r}; pick from {list(DEFAULT_PLAN_CI)}")
    plan = plan or DEFAULT_PLAN_CI[task]
    target_times_s = target_times_s or {}
    resolved = H.resolve_assets_dir(assets_dir)
    M.configure(
        assets_dir=resolved,
        out_dir=out_dir,
        explicit=resolved == assets_dir,
        prefer_pinned=verify_manifest,
    )
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
        "success": False, "deadline_hit": False,
    }  # fmt: skip
    M.log(
        f"=== minimax_h3_benchmark {combo}: task={task} plan={json.dumps(plan)} table={timeout_table}"
        + (" smoke=skipped" if skip_smoke else "")
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
    result["estimate_s"] = plan_estimate_s(task, plan, timeout_table, skip_smoke)
    if deadline_s is not None and result["estimate_s"] > deadline_s:
        M.log(
            f"  [plan] WARNING: ~{result['estimate_s']}s at the {timeout_table} pace, over the "
            f"{deadline_s:.0f}s deadline: the last cases will be skipped (raise the timeout or split the plan)"
        )
    smoke = H.smoke_case(task, cases_cfg)
    needed = set() if skip_smoke else set(M.case_assets(smoke))
    for case, _ in selected:
        needed.update(M.case_assets(case))
    asset_problems = (
        M.verify_assets(needed)
        if verify_manifest
        else [f"missing asset {n}" for n in sorted(needed) if M.asset_path(n) is None]
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
        # 3. smoke (a later entry against the same deployment may skip it)
        if skip_smoke:
            rec, problems, notes, stop = {"outcome": "skipped"}, [], [], ""
            status = "skipped"
            M.log(f"  [smoke] {smoke['id']} skipped (skip_smoke)")
        else:
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
            status = "ok" if rec.get("outcome") == "ok" and not problems else "failed"
        result["smoke"] = {"case": smoke["id"], "outcome": rec.get("outcome"), "gen_s": rec.get("gen_s"),
                           "queue_s": rec.get("queue_s"), "problems": problems, "notes": notes, "stop": stop}  # fmt: skip
        with open(os.path.join(out_dir, f"smoke_status_{combo}_{task}.log"), "w") as fh:
            fh.write(status + "\n")
            fh.write(f"outcome={rec.get('outcome')} problems={problems}\n")

        # 4. the plan
        for case, runs in selected:
            over = deadline_s is not None and time.time() - started > deadline_s
            if over and not ep.stopped:
                result["deadline_hit"] = True
            if ep.stopped or over:
                stop = (
                    ep.stopped
                    or f"deadline: {deadline_s:.0f}s elapsed before this case started"
                )
                result["cases"].append({"case": case["id"], "task": task, "status": "skipped",
                                        "stop": stop, "runs_requested": runs, "runs_ok": 0})  # fmt: skip
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
        and result["smoke"]["outcome"] in ("ok", "skipped")
        and not result["smoke"]["problems"]
    )
    result["success"] = (
        smoke_ok
        and not ep.stopped
        and all(s in ("pass", "xfail") for s in statuses)
        and not result["leftover_jobs"]
    )
    medians = [c["median_s"] for c in result["cases"] if c.get("median_s") is not None]
    smoke_word = "skipped" if skip_smoke else "ok" if smoke_ok else "FAILED"
    result["summary"] = (
        f"smoke {smoke_word}; "
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

    def _task(self) -> str:
        requested = self.targets.get("task")
        return deployment_task(self.ctx, None if requested is None else str(requested))

    def _plan(self, task: str = "t2va") -> list:
        if _ci_mode(self.ctx):
            return list(self.targets.get("plan_ci") or DEFAULT_PLAN_CI[task])
        return list(self.targets.get("plan_full") or DEFAULT_PLAN_FULL[task])

    def _out_dir(self, plan: list) -> str:
        """``out_dir`` (default ``<output>/minimax_h3_bench``), plus ``out_subdir`` when set.
        Without one, the first entry of the run keeps that directory -- the single-entry
        layout -- and any other entry writes to a subdirectory named after its cases (with a
        counter when that is taken), so no entry overwrites another's summary.json or rows,
        even when two entries run the same plan. The same entry again keeps its directory."""
        if self.targets.get("out_dir"):
            base = str(self.targets["out_dir"])
        elif self.ctx is not None:
            base = str(Path(self.ctx.output_path) / "minimax_h3_bench")
        else:
            base = "/tmp/minimax_h3_bench"
        entry = json.dumps([self.targets, plan], sort_keys=True, default=str)
        key = hashlib.sha256(entry.encode()).hexdigest()
        sub = self.targets.get("out_subdir")
        if sub:
            rel = Path(str(sub))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(
                    f"targets.out_subdir {sub!r} must be a relative path below {base}"
                )
            path = os.path.join(base, str(rel))
            if _OUT_DIR_OWNERS.setdefault(path, key) != key:
                raise ValueError(
                    f"targets.out_subdir {sub!r}: {path} is already another entry's directory"
                )
            return path
        slug = _cases_slug(plan)
        candidates = itertools.chain(
            [base, os.path.join(base, slug)],
            (os.path.join(base, f"{slug}-{n}") for n in itertools.count(2)),
        )
        return next(p for p in candidates if _OUT_DIR_OWNERS.setdefault(p, key) == key)

    async def _run_specific_test_async(self) -> dict[str, Any]:
        task = self._task()
        plan = self._plan(task)
        return await asyncio.to_thread(
            run_benchmark,
            base_url=self.base_url,
            task=task,
            plan=plan,
            out_dir=self._out_dir(plan),
            assets_dir=self.targets.get("assets_dir"),
            timeout_table=str(
                self.targets.get("timeout_table", M.DEFAULT_TIMEOUT_TABLE)
            ),
            idle_wait_s=int(self.targets.get("idle_wait_s", H.IDLE_WAIT_S)),
            target_times_s=dict(self.targets.get("target_times_s") or {}),
            enforce_timing=bool(self.targets.get("enforce_timing", False)),
            verify_manifest=bool(self.targets.get("verify_manifest", True)),
            # CI never resumes: a previous run's rows must not stand in for generations that
            # did not happen. Resume stays a CLI feature (main() below, --force off by default).
            force=not bool(self.targets.get("allow_resume", False)),
            continue_after_timeout=bool(
                self.targets.get("continue_after_timeout", False)
            ),
            combo=self.targets.get("combo"),
            api_key=self.targets.get("api_key") or resolve_server_api_key(),
            deadline_s=self._deadline_s(),
            skip_smoke=bool(self.targets.get("skip_smoke", False)),
        )

    def _deadline_s(self) -> float | None:
        """Cooperative bound: BaseTest's timeout cannot interrupt the worker thread, so the
        benchmark stops starting cases this long after it began (600 s before the timeout,
        for cancel and teardown) and reports the cases it never started as skipped."""
        if not self.timeout:
            return None
        return max(60.0, float(self.timeout) - 600.0)


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
    parser.add_argument(
        "--cases",
        help="comma-separated case ids (default: the task's first case, e.g. T2VA-L)",
    )
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
    cases = args.cases or DEFAULT_PLAN_CI[args.task][0]["cases"][0]
    result = run_benchmark(
        base_url=args.base_url, task=args.task,
        plan=[{"cases": [c.strip() for c in cases.split(",") if c.strip()], "runs": args.runs}],
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
