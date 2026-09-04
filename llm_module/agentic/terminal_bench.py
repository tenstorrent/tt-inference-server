# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from llm_module.agentic.progress import (
    TIMEOUT_EXIT_CODE,
    make_terminal_bench_probe,
    run_with_progress,
)

logger = logging.getLogger(__name__)

# Conservative fallback per-trial agent budget when the config leaves
# ``agent_timeout_sec`` unset (Harbor then uses each task's own default, which
# we cannot see from here). Only feeds the deadline math; the stall watchdog is
# the real protection, so err generous to avoid false kills.
_DEFAULT_AGENT_TIMEOUT_SEC = 60 * 60


@dataclass(frozen=True)
class TerminalBenchRunConfig:
    task_name: str
    dataset: str
    agent: str
    model_name: str
    jobs_dir: Path
    api_base: str
    n_concurrent_trials: int
    n_attempts: int
    environment_type: str
    agent_kwargs: dict[str, Any]
    n_tasks: Optional[int]
    override_cpus: Optional[int]
    override_memory_mb: Optional[int]
    timeout_multiplier: Optional[float]
    agent_timeout_sec: Optional[float]
    task_names: list[str] = field(default_factory=list)
    exclude_task_names: list[str] = field(default_factory=list)
    quiet: bool = True
    yes: bool = True
    agent_import_path: Optional[str] = None
    environment_env: dict[str, str] = field(default_factory=dict)
    verifier_env: dict[str, str] = field(default_factory=dict)
    # Wave-aware deadline model (see progress.py). Reserved allowance for
    # Harbor's additive non-agent phases (env build, agent setup, verifier);
    # currently NOT folded into the per-task budget -- B is just
    # ``agent_timeout_sec``, and the stall watchdog is the real protection.
    per_task_overhead_sec: int = 20 * 60
    startup_grace_sec: int = 10 * 60
    stall_grace_sec: int = 5 * 60
    progress_log_interval_sec: int = 5 * 60
    # When False the progress watchdog logs deadlines but never kills the harbor
    # subprocess, letting it run to completion.
    enforce_agent_deadline: bool = False
    # Interpreter whose bin/ holds the ``harbor`` CLI. When ``None`` the current
    # interpreter is used (standalone ``run_agentic.py`` already re-execs into
    # the EVALS_AGENTIC venv). Set on the release path, where the harness runs
    # as a child of the WORKFLOW_RUN_SCRIPT engine and must reach harbor explicitly.
    venv_python: Optional[Path] = None


def _get_agent_kwargs(config: TerminalBenchRunConfig) -> dict[str, Any]:
    agent_kwargs = dict(config.agent_kwargs)
    agent_kwargs.setdefault("api_base", config.api_base)
    return agent_kwargs


def _format_kwarg(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_harbor_config(config: TerminalBenchRunConfig) -> Path:
    config_path = config.jobs_dir / f"{config.task_name}_harbor_config.json"
    config.jobs_dir.mkdir(parents=True, exist_ok=True)

    dataset_config: dict[str, Any] = {"name": config.dataset}
    if config.n_tasks is not None:
        dataset_config["n_tasks"] = config.n_tasks
    if config.task_names:
        dataset_config["task_names"] = config.task_names
    if config.exclude_task_names:
        dataset_config["exclude_task_names"] = config.exclude_task_names

    environment_config: dict[str, Any] = {"type": config.environment_type}
    if config.override_cpus is not None:
        environment_config["override_cpus"] = config.override_cpus
    if config.override_memory_mb is not None:
        environment_config["override_memory_mb"] = config.override_memory_mb

    agent_config: dict[str, Any] = {
        "model_name": config.model_name,
        "override_timeout_sec": config.agent_timeout_sec,
        "kwargs": _get_agent_kwargs(config),
    }
    if config.agent_import_path:
        agent_config["import_path"] = config.agent_import_path
    else:
        agent_config["name"] = config.agent

    if config.environment_env:
        environment_config["env"] = config.environment_env

    verifier_config: dict[str, Any] = {}
    if config.verifier_env:
        verifier_config["env"] = config.verifier_env

    harbor_config: dict[str, Any] = {
        "job_name": config.task_name,
        "jobs_dir": str(config.jobs_dir),
        "n_attempts": config.n_attempts,
        "n_concurrent_trials": config.n_concurrent_trials,
        "quiet": config.quiet,
        "environment": environment_config,
        "agents": [agent_config],
        "datasets": [dataset_config],
    }
    if verifier_config:
        harbor_config["verifier"] = verifier_config
    if config.timeout_multiplier is not None:
        harbor_config["timeout_multiplier"] = config.timeout_multiplier
    if config.agent_timeout_sec is not None:
        # Keep the agent override exact even if timeout_multiplier is used for
        # verifier or environment timeouts.
        harbor_config["agent_timeout_multiplier"] = 1.0

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(harbor_config, f, indent=2)

    return config_path


def _needs_config_file(config: TerminalBenchRunConfig) -> bool:
    return (
        config.agent_timeout_sec is not None
        or config.agent_import_path is not None
        or bool(config.environment_env)
        or bool(config.verifier_env)
    )


def _annotate_result_file(result_file: Path) -> None:
    try:
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "_result_format" not in data:
            data["_result_format"] = "harbor"
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except (json.JSONDecodeError, IOError) as e:
        msg = (
            f"Could not annotate result file {result_file} with '_result_format' field "
            f"required for report processing: {e}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e


def run(config: TerminalBenchRunConfig) -> int:
    interpreter = config.venv_python or Path(sys.executable)
    harbor_exec = Path(interpreter).parent / "harbor"

    if _needs_config_file(config):
        harbor_config_path = _write_harbor_config(config)
        cmd = [str(harbor_exec), "run", "--config", str(harbor_config_path)]
        if config.yes:
            cmd.append("--yes")
    else:
        config.jobs_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(harbor_exec),
            "run",
            "--dataset",
            config.dataset,
            "--agent",
            config.agent,
            "--model",
            config.model_name,
            "--jobs-dir",
            str(config.jobs_dir),
            "--job-name",
            config.task_name,
            "--n-concurrent",
            str(config.n_concurrent_trials),
            "--n-attempts",
            str(config.n_attempts),
            "--env",
            config.environment_type,
        ]
        if config.quiet:
            cmd.append("--quiet")
        if config.yes:
            cmd.append("--yes")
        if config.n_tasks is not None:
            cmd.extend(["--n-tasks", str(config.n_tasks)])
        if config.override_cpus is not None:
            cmd.extend(["--override-cpus", str(config.override_cpus)])
        if config.override_memory_mb is not None:
            cmd.extend(["--override-memory-mb", str(config.override_memory_mb)])
        if config.timeout_multiplier is not None:
            cmd.extend(["--timeout-multiplier", str(config.timeout_multiplier)])
        for task_name in config.task_names:
            cmd.extend(["--include-task-name", task_name])
        for task_name in config.exclude_task_names:
            cmd.extend(["--exclude-task-name", task_name])

        agent_kwargs = _get_agent_kwargs(config)
        for key, value in agent_kwargs.items():
            cmd.extend(["--agent-kwarg", f"{key}={_format_kwarg(value)}"])

    job_dir = config.jobs_dir / config.task_name
    agent_timeout = (
        config.agent_timeout_sec
        if config.agent_timeout_sec is not None
        else _DEFAULT_AGENT_TIMEOUT_SEC
    )
    per_task_budget = agent_timeout
    rc = run_with_progress(
        cmd,
        cwd=None,
        env=os.environ.copy(),
        probe=make_terminal_bench_probe(job_dir),
        label=config.task_name,
        per_task_budget_s=per_task_budget,
        concurrency=config.n_concurrent_trials,
        startup_grace_s=config.startup_grace_sec,
        stall_grace_s=config.stall_grace_sec,
        log_interval_s=config.progress_log_interval_sec,
        enforce_deadlines=config.enforce_agent_deadline,
        log=logger,
    )
    # A watchdog timeout (124) still leaves harbor's per-trial results (each
    # already graded inline) in result.json worth annotating; only a genuine
    # harness error aborts before annotation.
    if rc != 0 and rc != TIMEOUT_EXIT_CODE:
        return rc
    result_path = job_dir / "result.json"
    if rc == TIMEOUT_EXIT_CODE and not result_path.exists():
        logger.error(
            "Harbor timed out before writing any results; nothing to annotate."
        )
        return rc
    if rc == TIMEOUT_EXIT_CODE:
        logger.warning(
            "Harbor hit the deadline; annotating the partial results in %s.",
            result_path,
        )
    _annotate_result_file(result_path)
    # Partial results were graded and annotated; report success so downstream
    # scoring reads the (lower) partial score instead of treating the run as a
    # hard failure.
    return 0
