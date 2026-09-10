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

import copy
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

# Harbor's built-in agent name for mini-swe-agent. When this agent is used we
# bring its generated model config to parity with the standalone SWE-bench
# harness (see ``_apply_mini_swe_agent_defaults``).
_MINI_SWE_AGENT = "mini-swe-agent"


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
    # Rich Live progress: quiet=True shows only the loading bar; quiet=False
    # adds per-trial stage spinners (env start, agent start, verification).
    # In CI (non-TTY) Rich degrades Live to static line-by-line output, which
    # is exactly the visibility we want — each trial stage prints a line.
    quiet: bool = False
    yes: bool = True
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
    # Per-request LLM read timeout, injected into a mini-swe-agent run as
    # ``model.model_kwargs.timeout`` (see ``_apply_mini_swe_agent_defaults``).
    # litellm's OpenAI-compatible path otherwise defaults to an infinite read
    # timeout, so a never-answered request hangs the trial. ``None`` opts out.
    # Ignored for every non-mini agent (terminus-2, tau3, ... carry their own
    # timeout knob in ``agent_kwargs``).
    llm_timeout_sec: Optional[int] = 10 * 60
    # Allowance for Harbor's additive non-agent phases (environment build,
    # agent setup, verifier). Added to the agent timeout for each trial wave.
    per_task_overhead_sec: int = 20 * 60
    startup_grace_sec: int = 10 * 60
    stall_grace_sec: int = 5 * 60
    progress_log_interval_sec: int = 60
    # When False, heuristic wave/stall deadlines are log-only. Explicit hard
    # timeouts remain enforced.
    enforce_agent_deadline: bool = False
    # Interpreter whose bin/ holds the ``harbor`` CLI. When ``None`` the current
    # interpreter is used (standalone ``run_agentic.py`` already re-execs into
    # the EVALS_AGENTIC venv). Set on the release path, where the harness runs
    # as a child of the WORKFLOW_RUN_SCRIPT engine and must reach harbor explicitly.
    venv_python: Optional[Path] = None
    harbor_timeout_sec: Optional[float] = None


def _apply_mini_swe_agent_defaults(
    agent_kwargs: dict[str, Any], config: HarborRunConfig
) -> None:
    """Bring a mini-swe-agent run to parity with the standalone SWE-bench harness.

    Harbor's ``mini-swe-agent`` agent takes an inline ``config`` mapping that it
    dumps to a mini-swe-agent YAML and injects with ``-c``. The standalone
    harness (``llm_module/agentic/swebench.py`` before the unification) always
    wrote three model defaults that harbor does not add on its own; without them
    a mini run inside harbor diverges from the standalone eval:

    * ``model.model_kwargs.drop_params=True`` -- litellm drops params the server
      rejects instead of erroring the whole request (needed for e.g. top_k in
      ``extra_body``).
    * ``model.model_kwargs.timeout`` -- litellm's OpenAI-compatible path defaults
      to an infinite read timeout, so a never-answered request otherwise hangs
      the trial until the whole-agent budget. ``llm_timeout_sec=None`` opts out.
    * ``model.cost_tracking='ignore_errors'`` -- a cost-lookup miss must not
      abort the run.

    Each value is a ``setdefault``, so an explicit per-eval value in
    ``agent_kwargs['config']`` always wins. ``agent_kwargs`` is mutated in place;
    the caller passes a deep copy so the shared eval config is never touched.
    """
    cfg = agent_kwargs.setdefault("config", {})
    if not isinstance(cfg, dict):
        return
    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        return
    model.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model.setdefault("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        return
    model_kwargs.setdefault("drop_params", True)
    if config.llm_timeout_sec is not None:
        model_kwargs.setdefault("timeout", config.llm_timeout_sec)


def _get_agent_kwargs(config: HarborRunConfig) -> dict[str, Any]:
    """Agent kwargs with the resolved endpoint added as ``api_base``.

    Note that ``api_base`` is not how every agent learns the endpoint. Harbor's
    in-container "installed" agents (mini-swe-agent among them) accept the kwarg
    but ignore it, reading ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` from the
    agent env -- which falls back to the harbor host's environment, where
    ``agentic_eval_tests._configure_openai_env`` has already exported them. Set
    ``agent_env`` to override that per eval. Agents implemented in Harbor itself
    (e.g. terminus-2) do read this kwarg, hence the unconditional default.

    A deep copy is taken so the mini-swe-agent parity defaults never mutate the
    shared, module-level eval config.
    """
    agent_kwargs = copy.deepcopy(dict(config.agent_kwargs))
    agent_kwargs.setdefault("api_base", config.api_base)
    if config.agent == _MINI_SWE_AGENT and config.agent_import_path is None:
        _apply_mini_swe_agent_defaults(agent_kwargs, config)
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

    job_dir = config.jobs_dir / config.task_name
    agent_timeout = (
        config.agent_timeout_sec
        if config.agent_timeout_sec is not None
        else _DEFAULT_AGENT_TIMEOUT_SEC
    )
    per_task_budget = agent_timeout + config.per_task_overhead_sec
    # ``harbor_timeout_sec`` is an optional flat backstop kept from the unified
    # harness; the wave-aware stall/ceiling watchdog is the primary protection.
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
        hard_timeout_s=config.harbor_timeout_sec,
        enforce_deadlines=config.enforce_agent_deadline,
        log=logger,
    )
    # A watchdog timeout can still leave useful per-trial diagnostics in
    # result.json. Annotate that file, but preserve rc=124: Harbor computes
    # accuracy over completed trials, so treating a partial file as success can
    # inflate its score by excluding unfinished trials from the denominator.
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
    try:
        _annotate_result_file(result_path)
    except RuntimeError:
        if rc != TIMEOUT_EXIT_CODE:
            raise
        logger.warning(
            "Harbor timed out while result.json was incomplete; preserving rc=%d.",
            rc,
            exc_info=True,
        )
    return rc
