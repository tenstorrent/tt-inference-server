# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Driver adapters for v2 agentic eval harnesses."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from workflow_module.engine_types import EvalLimitMode

from ..agentic.harbor import HarborRunConfig, run as run_harbor
from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..parsers.agentic import AgenticEvalParser
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


def _openai_model_name(model: str) -> str:
    """Add LiteLLM's provider prefix to the server's model ID.

    The server model ID may itself begin with ``openai/``. In that case,
    ``openai/openai/...`` is intentional: the first segment selects LiteLLM's
    OpenAI provider and the remainder is sent unchanged to the server.
    """
    return f"openai/{model}"


def _runtime_environment_env(configured_env: dict[str, str]) -> dict[str, str]:
    """Add runtime credentials only for compose tasks that need interpolation."""
    environment_env = dict(configured_env)
    if "TAU2_USER_MODEL" not in environment_env:
        return environment_env
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        value = os.getenv(key)
        if value:
            environment_env.setdefault(key, value)
    return environment_env


class AgenticEvalDriver(LLMDriver):
    """Base adapter for one configured agentic eval task."""

    name = "agentic"

    def __init__(self, task: Any, *, runtime_config: Any = None) -> None:
        self.task = task
        self.runtime_config = runtime_config
        self.venv_python = _agentic_venv_python()
        # Set at the start of run(); stamped onto the harness job/output folder
        # so a re-run into the same logs dir does not collide with the previous
        # run's folder (harbor/sweagent refuse to start in an existing one).
        self._run_stamp: Optional[str] = None
        self._parser = AgenticEvalParser(
            task_name=task.task_name,
            score=task.score,
            limit_mode=_get_limit_mode(runtime_config),
        )

    def result_path(self, server: ServerConnection, context: DriverContext) -> Path:
        output_dir = _agentic_output_dir(
            context.output_dir,
            server.model,
            self.task,
            release_layout=context.agentic_release_layout,
            run_stamp=self._run_stamp,
        )
        return output_dir / "result.json"

    def failure_block(self, *, return_code: int, device: str = ""):
        return self._parser.failure_block(return_code=return_code, device=device)

    def _load_result(
        self,
        rc: int,
        result_path: Path,
    ) -> DriverResult:
        if rc != 0:
            return DriverResult(return_code=rc, raw=None, raw_path=None)
        if not result_path.exists():
            raise RuntimeError(
                f"Result JSON not found at {result_path} after rc=0 for "
                f"task {self.task.task_name!r}."
            )
        with result_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._parser = AgenticEvalParser(
            task_name=self.task.task_name,
            score=self.task.score,
            result_path=result_path,
            limit_mode=_get_limit_mode(self.runtime_config),
        )
        return DriverResult(return_code=rc, raw=raw, raw_path=result_path)


class HarborAgenticDriver(AgenticEvalDriver):
    """Runs one agentic eval task through Harbor.

    One driver for every agentic benchmark: terminal-bench, tau3-bench, and
    SWE-bench differ only in the dataset/agent/environment Harbor is pointed at.
    """

    name = "harbor"

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        n_tasks = resolve_n_tasks(self.task, self.runtime_config)
        if n_tasks == 0:
            logger.info(
                "Skipping agentic task %s: n_tasks=0 for this limit mode",
                self.task.task_name,
            )
            return DriverResult(return_code=0, raw=None, raw_path=None)
        self._run_stamp = _run_timestamp()
        _cfg = self.task.agentic_eval_config
        logger.info(
            "[agentic] starting %r: n_tasks=%s n_concurrent=%s agent=%s dataset=%s model=%s",
            self.task.task_name,
            n_tasks,
            getattr(_cfg, "n_concurrent_trials", "?"),
            getattr(_cfg, "agent", "?"),
            getattr(_cfg, "dataset", "?"),
            getattr(_cfg, "model", None),
        )
        run_config = build_harbor_config(
            self.task,
            server,
            context,
            runtime_config=self.runtime_config,
            n_tasks=n_tasks,
            venv_python=self.venv_python,
            run_stamp=self._run_stamp,
        )
        rc = run_harbor(run_config)
        return self._load_result(rc, self.result_path(server, context))


def _agentic_venv_python() -> Optional[Path]:
    """Interpreter of the EVALS_AGENTIC venv whose bin/ holds harbor/sweagent.

    Returned to the harness so it can locate its CLI even when the agentic
    driver runs as a child of the WORKFLOW_RUN_SCRIPT engine (release path) rather
    than after ``run_agentic.py`` re-execs into the agentic venv. Resolution
    failures fall back to ``None`` (current interpreter), preserving standalone
    behavior.
    """
    try:
        from workflow_module.engine_types import WorkflowVenvType
        from workflow_module.venv_provisioner import get_venv_provisioner

        return Path(get_venv_provisioner().venv_python(WorkflowVenvType.EVALS_AGENTIC))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not resolve EVALS_AGENTIC venv python (%s).", e)
        return None


def make_agentic_driver(task: Any, *, runtime_config: Any = None) -> AgenticEvalDriver:
    if task.agentic_eval_config is None:
        raise RuntimeError(
            f"EVALS_AGENTIC task {task.task_name!r} has no agentic_eval_config set."
        )
    return HarborAgenticDriver(task, runtime_config=runtime_config)


def build_harbor_config(
    task: Any,
    server: ServerConnection,
    context: DriverContext,
    *,
    runtime_config: Any = None,
    n_tasks: Optional[int] = None,
    venv_python: Optional[Path] = None,
    run_stamp: Optional[str] = None,
) -> HarborRunConfig:
    cfg = task.agentic_eval_config
    task_output_dir = _agentic_output_dir(
        context.output_dir,
        server.model,
        task,
        release_layout=context.agentic_release_layout,
        run_stamp=run_stamp,
    )
    return HarborRunConfig(
        task_name=task_output_dir.name,
        dataset=cfg.dataset,
        agent=cfg.agent,
        model_name=_openai_model_name(cfg.model or server.model),
        jobs_dir=task_output_dir.parent,
        api_base=f"{server.url_with_port}/v1",
        n_concurrent_trials=cfg.n_concurrent_trials,
        n_attempts=cfg.n_attempts,
        environment_type=cfg.environment_type,
        agent_kwargs=cfg.agent_kwargs,
        n_tasks=n_tasks if n_tasks is not None else cfg.n_tasks,
        override_cpus=cfg.override_cpus,
        override_memory_mb=cfg.override_memory_mb,
        timeout_multiplier=cfg.timeout_multiplier,
        agent_timeout_sec=cfg.agent_timeout_sec,
        agent_setup_timeout_multiplier=cfg.agent_setup_timeout_multiplier,
        task_names=resolve_task_names(task, runtime_config),
        exclude_task_names=cfg.exclude_task_names,
        quiet=cfg.quiet,
        yes=cfg.yes,
        agent_import_path=cfg.agent_import_path,
        agent_env=cfg.agent_env,
        environment_env=_runtime_environment_env(cfg.environment_env),
        verifier_env=cfg.verifier_env,
        environment_kwargs=(
            cfg.environment_kwargs if cfg.environment_type == "kubernetes" else {}
        ),
        harbor_timeout_sec=cfg.harbor_timeout_sec,
        llm_timeout_sec=cfg.llm_timeout_sec,
        per_task_overhead_sec=cfg.per_task_overhead_sec,
        startup_grace_sec=cfg.startup_grace_sec,
        stall_grace_sec=cfg.stall_grace_sec,
        progress_log_interval_sec=cfg.progress_log_interval_sec,
        enforce_agent_deadline=cfg.enforce_agent_deadline,
        venv_python=venv_python,
    )


def resolve_task_names(task: Any, runtime_config: Any = None) -> List[str]:
    agentic_config = task.agentic_eval_config
    if agentic_config is None:
        return []
    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode is not None and limit_mode in agentic_config.task_names_map:
        return agentic_config.task_names_map[limit_mode]
    return agentic_config.task_names


def resolve_n_tasks(task: Any, runtime_config: Any = None) -> Optional[int]:
    agentic_config = task.agentic_eval_config
    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode is None:
        return agentic_config.n_tasks if agentic_config else None

    limit_arg = task.limit_samples_map.get(limit_mode)
    if limit_arg is None:
        return agentic_config.n_tasks if agentic_config else None
    if isinstance(limit_arg, float) and limit_arg < 1:
        logger.warning(
            "Agentic eval limits are task counts, not fractions; using one task for %s",
            task.task_name,
        )
        return 1
    return int(limit_arg)


def _get_limit_mode(runtime_config: Any = None) -> Optional[EvalLimitMode]:
    if runtime_config is None or not getattr(
        runtime_config, "limit_samples_mode", None
    ):
        return None
    return EvalLimitMode.from_string(runtime_config.limit_samples_mode)


def _run_timestamp() -> str:
    """Filesystem-safe per-run stamp, e.g. ``20260813T120000``."""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _agentic_output_dir(
    output_root: Path,
    model_id: str,
    task: Any,
    *,
    release_layout: bool = False,
    run_stamp: Optional[str] = None,
) -> Path:
    safe_model_id = model_id.replace("/", "__")
    # Stamp the per-task folder so a re-run into the same logs dir gets a fresh
    # folder; the harnesses (harbor job, sweagent output) refuse to start in an
    # existing one.
    task_dir = task.task_name if not run_stamp else f"{task.task_name}_{run_stamp}"
    if release_layout:
        # release run: group all agentic results under a single top-level
        # ``agentic/`` dir (sibling of ``llm/`` / ``prefix_cache/``) so the
        # tree mirrors the LLM layout: agentic/eval_<hf>/<task>.
        return Path(output_root) / "agentic" / f"eval_{safe_model_id}" / task_dir
    # standalone agentic run: eval_<hf>/agentic/<task>.
    return Path(output_root) / f"eval_{safe_model_id}" / "agentic" / task_dir


__all__ = [
    "AgenticEvalDriver",
    "HarborAgenticDriver",
    "build_harbor_config",
    "make_agentic_driver",
    "resolve_n_tasks",
    "resolve_task_names",
]
