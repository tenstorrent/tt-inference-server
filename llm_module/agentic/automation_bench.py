# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""zapier/AutomationBench runner.

AutomationBench is not a Harbor benchmark: its 47 SaaS tools are simulated
in-process and each task is graded by assertions on the final world state, so
the whole run is one ``auto-bench`` process talking to the model endpoint.
This module builds that command, runs it from the AUTOMATION_BENCH venv, and
rewrites the CLI's ``--export-json`` file as a Harbor-shaped ``result.json``
so the agentic parser and reports read it like any other agentic eval.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Returned when ``timeout_sec`` kills the run, matching the agentic watchdog.
TIMEOUT_EXIT_CODE = 124
EXPORT_FILE_NAME = "automation_bench_results.json"
RESULT_FILE_NAME = "result.json"
# Not part of the official pass rate (see AUTOMATION_BENCH_DOMAINS).
UNSCORED_DOMAINS = ("simple",)


@dataclass(frozen=True)
class AutomationBenchRunConfig:
    task_name: str
    model_name: str
    api_base: str
    job_dir: Path
    venv_python: Path
    checkout_dir: Path
    domains: list[str]
    toolset: str = "api"
    max_steps: int = 50
    max_concurrent: int = 32
    n_tasks: int | None = None
    task_names: list[str] = field(default_factory=list)
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_sec: float | None = None
    api_key_var: str = "OPENAI_API_KEY"

    @property
    def export_path(self) -> Path:
        return self.job_dir / EXPORT_FILE_NAME

    @property
    def result_path(self) -> Path:
        return self.job_dir / RESULT_FILE_NAME


def build_command(config: AutomationBenchRunConfig) -> list[str]:
    """``auto-bench`` invocation for *config*.

    Run as ``python -m automationbench.scripts.eval`` from the checkout:
    AutomationBench is exported, not installed, into its venv. ``--api
    chat_completions`` is pinned so the client choice cannot depend on the
    model name (``auto`` routes ``claude-*``/``gemini-*`` names elsewhere).
    """
    cmd = [
        str(config.venv_python),
        "-m",
        "automationbench.scripts.eval",
        "--model",
        config.model_name,
        "--base-url",
        config.api_base,
        "--api-key-var",
        config.api_key_var,
        "--api",
        "chat_completions",
        "--domains",
        ",".join(config.domains),
        "--toolset",
        config.toolset,
        "--max-steps",
        str(config.max_steps),
        "--max-concurrent",
        str(config.max_concurrent),
        "--export-json",
        str(config.export_path),
    ]
    if config.n_tasks is not None:
        cmd.extend(["--num-examples", str(config.n_tasks)])
    if config.task_names:
        cmd.extend(["--tasks", ",".join(config.task_names)])
    if config.reasoning_effort:
        cmd.extend(["--reasoning-effort", config.reasoning_effort])
    if config.extra_body:
        cmd.extend(["--extra-body", json.dumps(config.extra_body, sort_keys=True)])
    return cmd


def run(config: AutomationBenchRunConfig) -> int:
    """Run auto-bench, then write ``result.json``. Returns the exit code."""
    config.job_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(config)
    # Logged from the config rather than ``cmd``, which carries the API key's
    # environment variable name.
    logger.info(
        "Running auto-bench for %s: model=%s base_url=%s domains=%s toolset=%s "
        "max_steps=%d max_concurrent=%d n_tasks=%s",
        config.task_name,
        config.model_name,
        config.api_base,
        ",".join(config.domains),
        config.toolset,
        config.max_steps,
        config.max_concurrent,
        config.n_tasks,
    )
    started_at = datetime.now(timezone.utc)
    try:
        rc = subprocess.run(
            cmd, cwd=config.checkout_dir, timeout=config.timeout_sec, check=False
        ).returncode
    except subprocess.TimeoutExpired:
        logger.error(
            "auto-bench exceeded timeout_sec=%s for %s",
            config.timeout_sec,
            config.task_name,
        )
        return TIMEOUT_EXIT_CODE
    finished_at = datetime.now(timezone.utc)
    if rc != 0:
        return rc
    if not config.export_path.exists():
        logger.error("auto-bench exited 0 but wrote no %s", config.export_path)
        return 1
    export = json.loads(config.export_path.read_text(encoding="utf-8"))
    result = to_harbor_result(
        export,
        eval_name=config.task_name,
        started_at=started_at,
        finished_at=finished_at,
    )
    config.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = result["automation_bench"]
    logger.info(
        "AutomationBench %s: pass_rate=%.4f partial_credit=%.4f over %d scored task(s); "
        "per domain: %s",
        config.task_name,
        summary["pass_rate"],
        summary["partial_credit"],
        summary["n_scored_tasks"],
        {d: round(v["pass_rate"], 4) for d, v in summary["per_domain"].items()},
    )
    if summary["aborted_tasks"]:
        logger.warning(
            "AutomationBench %s: %d task(s) aborted by infrastructure: %s",
            config.task_name,
            len(summary["aborted_tasks"]),
            summary["aborted_tasks"],
        )
    return 0


def task_domain(task_name: str) -> str:
    """AutomationBench task names are ``<domain>.<task>``."""
    return task_name.split(".", 1)[0]


def to_harbor_result(
    export: Mapping[str, Any],
    *,
    eval_name: str,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    """Rewrite an auto-bench export as a Harbor-shaped ``result.json``.

    ``task_completed_correctly`` (every assertion passed, exported as
    ``passed``) averaged over the scored domains is the official pass rate,
    so it is the eval's metric; ``partial_credit`` (exported as ``score``,
    the fraction of assertions passed) is kept per domain alongside it.
    Unscored ``simple`` tasks are reported per domain but left out of both.
    """
    tasks = list(export.get("tasks") or [])
    scored = [
        t for t in tasks if task_domain(t.get("name", "")) not in UNSCORED_DOMAINS
    ]
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for t in tasks:
        by_domain[task_domain(t.get("name", ""))].append(t)

    rewards: dict[str, list[str]] = defaultdict(list)
    for t in scored:
        rewards["1.0" if t.get("passed") else "0.0"].append(t.get("name", ""))

    pass_rate = _mean([1.0 if t.get("passed") else 0.0 for t in scored])
    partial_credit = _mean([float(t.get("score", 0.0)) for t in scored])
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "n_total_trials": len(scored),
        "stats": {
            "evals": {
                eval_name: {
                    "n_trials": len(scored),
                    "n_errors": sum(1 for t in scored if t.get("errors")),
                    "metrics": [{"mean": pass_rate}],
                    "reward_stats": {"reward": dict(rewards)},
                }
            }
        },
        "automation_bench": {
            "pass_rate": pass_rate,
            "partial_credit": partial_credit,
            "n_scored_tasks": len(scored),
            # Tasks cut off mid-work by infrastructure (scored 0): serving
            # trouble, not model failures. auto-bench re-runs them up to
            # --ensure-rounds times before exporting.
            "aborted_tasks": list(
                (export.get("summary") or {}).get("aborted_tasks") or []
            ),
            "per_domain": {
                domain: {
                    "n_tasks": len(domain_tasks),
                    "pass_rate": _mean(
                        [1.0 if t.get("passed") else 0.0 for t in domain_tasks]
                    ),
                    "partial_credit": _mean(
                        [float(t.get("score", 0.0)) for t in domain_tasks]
                    ),
                }
                for domain, domain_tasks in sorted(by_domain.items())
            },
            "meta": export.get("meta", {}),
        },
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
