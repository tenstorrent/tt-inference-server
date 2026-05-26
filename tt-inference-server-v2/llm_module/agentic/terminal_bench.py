# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


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

    harbor_config: dict[str, Any] = {
        "job_name": config.task_name,
        "jobs_dir": str(config.jobs_dir),
        "n_attempts": config.n_attempts,
        "n_concurrent_trials": config.n_concurrent_trials,
        "quiet": config.quiet,
        "environment": environment_config,
        "agents": [
            {
                "name": config.agent,
                "model_name": config.model_name,
                "override_timeout_sec": config.agent_timeout_sec,
                "kwargs": _get_agent_kwargs(config),
            }
        ],
        "datasets": [dataset_config],
    }
    if config.timeout_multiplier is not None:
        harbor_config["timeout_multiplier"] = config.timeout_multiplier
    if config.agent_timeout_sec is not None:
        # Keep the agent override exact even if timeout_multiplier is used for
        # verifier or environment timeouts.
        harbor_config["agent_timeout_multiplier"] = 1.0

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(harbor_config, f, indent=2)

    return config_path


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
    harbor_exec = Path(sys.executable).parent / "harbor"

    if config.agent_timeout_sec is not None:
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

    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    _annotate_result_file(config.jobs_dir / config.task_name / "result.json")
    return result.returncode
