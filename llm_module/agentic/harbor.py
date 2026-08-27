# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Runner for agentic evals executed through the Harbor CLI.

Drives every agentic benchmark we run -- terminal-bench, tau3-bench, and
SWE-bench -- by shelling out to ``harbor run``. Harbor owns task acquisition,
sandboxing (docker / kubernetes / ...), the agent, and scoring, and writes a
``result.json`` the agentic parser reads directly, so this module only has to
translate an eval config into a Harbor job config.
"""

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

HARBOR_SIGTERM_GRACE_SEC = 300


def _run_with_timeout(cmd: list[str], timeout_sec: Optional[float]) -> int:
    """Run harbor, terminating it *gracefully* if it overruns."""
    with subprocess.Popen(cmd) as proc:
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            logger.warning(
                "harbor exceeded %ss; sending SIGTERM and allowing %ss for "
                "sandbox teardown before SIGKILL.",
                timeout_sec,
                HARBOR_SIGTERM_GRACE_SEC,
            )
            proc.terminate()
            try:
                proc.wait(timeout=HARBOR_SIGTERM_GRACE_SEC)
                logger.info("harbor shut down cleanly after SIGTERM.")
            except subprocess.TimeoutExpired:
                logger.error(
                    "harbor did not exit %ss after SIGTERM; sending SIGKILL. "
                    "Sandboxes may be left behind.",
                    HARBOR_SIGTERM_GRACE_SEC,
                )
                proc.kill()
                proc.wait()
            raise


@dataclass(frozen=True)
class HarborRunConfig:
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
    agent_setup_timeout_multiplier: Optional[float] = None
    task_names: list[str] = field(default_factory=list)
    exclude_task_names: list[str] = field(default_factory=list)
    quiet: bool = True
    yes: bool = True
    # Stream harbor's per-trial DEBUG logs to stdout. On by default so CI -- a
    # non-TTY where harbor's Rich Live progress does not render -- still shows
    # which evals and rollouts are running. Orthogonal to (and safe alongside)
    # ``quiet``, which only governs the progress display.
    debug: bool = False
    agent_import_path: Optional[str] = None
    # Injected into the agent's container for the agent phase. Harbor's
    # installed agents also fall back to the harbor host's ``os.environ``, so
    # this is only needed to override the ambient value (or to be explicit
    # about it) -- e.g. pointing one eval at a different endpoint.
    agent_env: dict[str, str] = field(default_factory=dict)
    environment_env: dict[str, str] = field(default_factory=dict)
    verifier_env: dict[str, str] = field(default_factory=dict)
    # Provider-specific environment knobs (namespace, image_mode, node_selector,
    # ... for the ``kubernetes`` environment). Only expressible through the
    # config file, so a non-empty value forces that path.
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    # Interpreter whose bin/ holds the ``harbor`` CLI. When ``None`` the current
    # interpreter is used (standalone ``run_agentic.py`` already re-execs into
    # the EVALS_AGENTIC venv). Set on the release path, where the harness runs
    # as a child of the WORKFLOW_RUN_SCRIPT engine and must reach harbor explicitly.
    venv_python: Optional[Path] = None
    harbor_timeout_sec: Optional[float] = None


def _get_agent_kwargs(config: HarborRunConfig) -> dict[str, Any]:
    """Agent kwargs with the resolved endpoint added as ``api_base``.

    Note that ``api_base`` is not how every agent learns the endpoint. Harbor's
    in-container "installed" agents (mini-swe-agent among them) accept the kwarg
    but ignore it, reading ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` from the
    agent env -- which falls back to the harbor host's environment, where
    ``agentic_eval_tests._configure_openai_env`` has already exported them. Set
    ``agent_env`` to override that per eval. Agents implemented in Harbor itself
    (e.g. terminus-2) do read this kwarg, hence the unconditional default.
    """
    agent_kwargs = dict(config.agent_kwargs)
    agent_kwargs.setdefault("api_base", config.api_base)
    return agent_kwargs


def _format_kwarg(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_harbor_config(config: HarborRunConfig) -> Path:
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
    if config.environment_kwargs:
        environment_config["kwargs"] = config.environment_kwargs

    agent_config: dict[str, Any] = {
        "model_name": config.model_name,
        "override_timeout_sec": config.agent_timeout_sec,
        "kwargs": _get_agent_kwargs(config),
    }
    if config.agent_import_path:
        agent_config["import_path"] = config.agent_import_path
    else:
        agent_config["name"] = config.agent
    if config.agent_env:
        agent_config["env"] = config.agent_env

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
    if config.agent_setup_timeout_multiplier is not None:
        harbor_config["agent_setup_timeout_multiplier"] = (
            config.agent_setup_timeout_multiplier
        )
    if config.timeout_multiplier is not None:
        harbor_config["timeout_multiplier"] = config.timeout_multiplier
    if config.agent_timeout_sec is not None:
        # Keep the agent override exact even if timeout_multiplier is used for
        # verifier or environment timeouts.
        harbor_config["agent_timeout_multiplier"] = 1.0

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(harbor_config, f, indent=2)

    return config_path


def _needs_config_file(config: HarborRunConfig) -> bool:
    return (
        config.agent_timeout_sec is not None
        or config.agent_setup_timeout_multiplier is not None
        or config.agent_import_path is not None
        or bool(config.agent_env)
        or bool(config.environment_env)
        or bool(config.verifier_env)
        or bool(config.environment_kwargs)
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


def run(config: HarborRunConfig) -> int:
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
        if config.agent_setup_timeout_multiplier is not None:
            cmd.extend(
                [
                    "--agent-setup-timeout-multiplier",
                    str(config.agent_setup_timeout_multiplier),
                ]
            )
        for task_name in config.task_names:
            cmd.extend(["--include-task-name", task_name])
        for task_name in config.exclude_task_names:
            cmd.extend(["--exclude-task-name", task_name])

        agent_kwargs = _get_agent_kwargs(config)
        for key, value in agent_kwargs.items():
            cmd.extend(["--agent-kwarg", f"{key}={_format_kwarg(value)}"])

    if config.debug:
        cmd.append("--debug")

    logger.info("Running command: %s", " ".join(cmd))
    try:
        returncode = _run_with_timeout(cmd, config.harbor_timeout_sec)
    except subprocess.TimeoutExpired:
        # A stuck harbor/trial otherwise hangs to the outer job cap.
        logger.error(
            "harbor run exceeded harbor_timeout_sec=%s; terminated.",
            config.harbor_timeout_sec,
        )
        return 124
    if returncode != 0:
        return returncode
    _annotate_result_file(config.jobs_dir / config.task_name / "result.json")
    return returncode
