# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Driver adapters for v2 agentic eval harnesses."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from workflows.workflow_types import EvalLimitMode

from ..agentic.swebench import SWEbenchRunConfig, run as run_swebench
from ..agentic.terminal_bench import TerminalBenchRunConfig, run as run_terminal_bench
from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..parsers.agentic import AgenticEvalParser
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class AgenticEvalDriver(LLMDriver):
    """Base adapter for one configured agentic eval task."""

    name = "agentic"

    def __init__(self, task: Any, *, runtime_config: Any = None) -> None:
        self.task = task
        self.runtime_config = runtime_config
        self.venv_python = _agentic_venv_python()
        self._parser = AgenticEvalParser(task_name=task.task_name, score=task.score)

    def result_path(self, server: ServerConnection, context: DriverContext) -> Path:
        output_dir = _agentic_output_dir(
            context.output_dir,
            server.model,
            self.task,
            release_layout=context.agentic_release_layout,
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
        )
        return DriverResult(return_code=rc, raw=raw, raw_path=result_path)


class SWEbenchAgenticDriver(AgenticEvalDriver):
    name = "swebench"

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        n_tasks = resolve_n_tasks(self.task, self.runtime_config)
        if n_tasks == 0:
            logger.info(
                "Skipping SWE-bench task %s: n_tasks=0 for this limit mode",
                self.task.task_name,
            )
            return DriverResult(return_code=0, raw=None, raw_path=None)
        run_config = build_swebench_config(
            self.task,
            server,
            context,
            runtime_config=self.runtime_config,
            n_tasks=n_tasks,
            venv_python=self.venv_python,
        )
        rc = run_swebench(run_config)
        return self._load_result(rc, self.result_path(server, context))


class TerminalBenchAgenticDriver(AgenticEvalDriver):
    name = "terminal_bench"

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        n_tasks = resolve_n_tasks(self.task, self.runtime_config)
        if n_tasks == 0:
            logger.info(
                "Skipping Terminal-Bench task %s: n_tasks=0 for this limit mode",
                self.task.task_name,
            )
            return DriverResult(return_code=0, raw=None, raw_path=None)
        run_config = build_terminal_bench_config(
            self.task,
            server,
            context,
            runtime_config=self.runtime_config,
            n_tasks=n_tasks,
            venv_python=self.venv_python,
        )
        rc = run_terminal_bench(run_config)
        return self._load_result(rc, self.result_path(server, context))


def _agentic_venv_python() -> Optional[Path]:
    """Interpreter of the EVALS_AGENTIC venv whose bin/ holds harbor/sweagent.

    Returned to the harness so it can locate its CLI even when the agentic
    driver runs as a child of the V2_RUN_SCRIPT engine (release path) rather
    than after ``run_agentic.py`` re-execs into the agentic venv. Resolution
    failures fall back to ``None`` (current interpreter), preserving standalone
    behavior.
    """
    try:
        from workflows.workflow_types import WorkflowVenvType
        from workflows.workflow_venvs import VENV_CONFIGS

        return VENV_CONFIGS[WorkflowVenvType.EVALS_AGENTIC].venv_python
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not resolve EVALS_AGENTIC venv python (%s).", e)
        return None


def make_agentic_driver(task: Any, *, runtime_config: Any = None) -> AgenticEvalDriver:
    if task.swebench_eval_config is not None:
        return SWEbenchAgenticDriver(task, runtime_config=runtime_config)
    if task.agentic_eval_config is not None:
        return TerminalBenchAgenticDriver(task, runtime_config=runtime_config)
    raise RuntimeError(
        f"EVALS_AGENTIC task {task.task_name!r} has neither "
        "swebench_eval_config nor agentic_eval_config set."
    )


def build_swebench_config(
    task: Any,
    server: ServerConnection,
    context: DriverContext,
    *,
    runtime_config: Any = None,
    n_tasks: Optional[int] = None,
    venv_python: Optional[Path] = None,
) -> SWEbenchRunConfig:
    cfg = task.swebench_eval_config
    return SWEbenchRunConfig(
        task_name=task.task_name,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        sweagent_subset=cfg.sweagent_subset,
        agent_backend=cfg.agent_backend,
        model_name=cfg.model or f"openai/{server.model}",
        api_base=f"{server.url_with_port}/v1",
        output_dir=_agentic_output_dir(
            context.output_dir,
            server.model,
            task,
            release_layout=context.agentic_release_layout,
        ),
        sweagent_config=cfg.sweagent_config,
        mini_config=cfg.mini_config,
        mini_model_class=cfg.mini_model_class,
        mini_environment_class=cfg.mini_environment_class,
        n_concurrent_trials=cfg.n_concurrent_trials,
        max_workers=cfg.max_workers,
        n_tasks=n_tasks if n_tasks is not None else cfg.n_tasks,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=cfg.max_output_tokens,
        completion_kwargs=cfg.completion_kwargs,
        swebench_timeout_sec=cfg.swebench_timeout_sec,
        shuffle=cfg.shuffle,
        random_delay_multiplier=cfg.random_delay_multiplier,
        score_existing_predictions=False,
        instance_ids=resolve_instance_ids(task, runtime_config),
        venv_python=venv_python,
    )


def build_terminal_bench_config(
    task: Any,
    server: ServerConnection,
    context: DriverContext,
    *,
    runtime_config: Any = None,
    n_tasks: Optional[int] = None,
    venv_python: Optional[Path] = None,
) -> TerminalBenchRunConfig:
    cfg = task.agentic_eval_config
    jobs_dir = _agentic_output_dir(
        context.output_dir,
        server.model,
        task,
        release_layout=context.agentic_release_layout,
    ).parent
    return TerminalBenchRunConfig(
        task_name=task.task_name,
        dataset=cfg.dataset,
        agent=cfg.agent,
        model_name=cfg.model or f"openai/{server.model}",
        jobs_dir=jobs_dir,
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
        task_names=resolve_task_names(task, runtime_config),
        exclude_task_names=cfg.exclude_task_names,
        quiet=cfg.quiet,
        yes=cfg.yes,
        agent_import_path=cfg.agent_import_path,
        environment_env=cfg.environment_env,
        verifier_env=cfg.verifier_env,
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


def resolve_instance_ids(task: Any, runtime_config: Any = None) -> List[str]:
    swebench_config = task.swebench_eval_config
    if swebench_config is None:
        return []
    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode is not None and limit_mode in swebench_config.instance_ids_map:
        return swebench_config.instance_ids_map[limit_mode]
    return []


def resolve_n_tasks(task: Any, runtime_config: Any = None) -> Optional[int]:
    agentic_config = task.agentic_eval_config or task.swebench_eval_config
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


def _agentic_output_dir(
    output_root: Path,
    model_id: str,
    task: Any,
    *,
    release_layout: bool = False,
) -> Path:
    safe_model_id = model_id.replace("/", "__")
    if release_layout:
        # release run: group all agentic results under a single top-level
        # ``agentic/`` dir (sibling of ``llm/`` / ``prefix_cache/``) so the
        # tree mirrors the LLM layout: agentic/eval_<hf>/<task>.
        return Path(output_root) / "agentic" / f"eval_{safe_model_id}" / task.task_name
    # standalone agentic run: eval_<hf>/agentic/<task>.
    return Path(output_root) / f"eval_{safe_model_id}" / "agentic" / task.task_name


__all__ = [
    "AgenticEvalDriver",
    "SWEbenchAgenticDriver",
    "TerminalBenchAgenticDriver",
    "build_swebench_config",
    "build_terminal_bench_config",
    "make_agentic_driver",
    "resolve_instance_ids",
    "resolve_n_tasks",
    "resolve_task_names",
]
