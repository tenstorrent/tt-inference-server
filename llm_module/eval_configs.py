# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Select the standard (lm-eval / lmms-eval) eval tasks for an LLM model."""

from __future__ import annotations

import logging
from typing import Iterable, List

from workflows.workflow_types import EvalLimitMode, WorkflowVenvType

from .eval_command import _get_limit_mode, _parse_eval_samples_mapping

logger = logging.getLogger(__name__)

# Standard LLM/VLM eval backends driven by build_eval_command. EVALS_AUDIO /
# EVALS_EMBEDDING are media model types with their own v2 eval runners.
_STANDARD_EVAL_VENVS = frozenset(
    {
        WorkflowVenvType.EVALS_COMMON,
        WorkflowVenvType.EVALS_META,
        WorkflowVenvType.EVALS_VISION,
    }
)

# Aliases accepted by --agentic-benchmark.  Keep the selector in this
# configuration module so preflight admission and the runtime agentic driver
# cannot disagree about which tasks a command selected.
_AGENTIC_BENCHMARK_PREFIX_ALIASES = {
    "tau3": "tau3_bench_",
    "swebench": "swe_bench_",
}
_AGENTIC_BENCHMARK_EXACT_ALIASES = {
    "tb2.0": "terminal_bench_2",
    "tb2.1": "terminal_bench_2_1",
}


def filter_tasks_by_min_context(tasks: Iterable, model_spec) -> list:
    """Return release tasks explicitly admitted by the selected device row.

    ``EvalTask.min_context_required`` is an opt-in catalogue statement, not a
    derived clamp.  Tasks without it remain selected and must pass their own
    validators.  This distinction keeps malformed or oversized agentic
    contracts fail closed unless the catalogue author deliberately marks the
    task as unavailable below a specific context.
    """
    available = getattr(
        getattr(model_spec, "device_model_spec", None), "max_context", None
    )
    selected = []
    for task in tasks:
        required = getattr(task, "min_context_required", None)
        if required is None:
            selected.append(task)
            continue
        if not isinstance(required, int) or isinstance(required, bool) or required <= 0:
            raise ValueError(
                f"Eval task {task.task_name!r} min_context_required must be a "
                f"positive integer, got {required!r}"
            )
        if (
            not isinstance(available, int)
            or isinstance(available, bool)
            or available <= 0
        ):
            raise ValueError(
                "Cannot apply an eval task's min_context_required without a "
                f"positive DeviceModelSpec.max_context, got {available!r}"
            )
        if available < required:
            logger.info(
                "Skipping release eval task %s: device max_context=%d is below "
                "its explicit min_context_required=%d",
                task.task_name,
                available,
                required,
            )
            continue
        selected.append(task)
    return selected


def parse_agentic_benchmark(value: str) -> tuple[list[str], set[str]]:
    """Parse --agentic-benchmark into task-name prefix and exact matchers."""
    prefixes: list[str] = []
    exacts: set[str] = set()
    for token in (part.strip().lower() for part in value.split(",")):
        if not token or token == "all":
            continue
        if token in _AGENTIC_BENCHMARK_PREFIX_ALIASES:
            prefixes.append(_AGENTIC_BENCHMARK_PREFIX_ALIASES[token])
        elif token in _AGENTIC_BENCHMARK_EXACT_ALIASES:
            exacts.add(_AGENTIC_BENCHMARK_EXACT_ALIASES[token])
        else:
            exacts.add(token)
    return prefixes, exacts


def filter_agentic_tasks_by_benchmark(tasks: list, selection: str) -> list:
    """Return exactly the configured agentic tasks selected by the CLI."""
    prefixes, exacts = parse_agentic_benchmark(selection)
    if not prefixes and not exacts:
        return tasks
    selected = [
        task
        for task in tasks
        if task.task_name in exacts
        or any(task.task_name.startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        available = [task.task_name for task in tasks]
        raise RuntimeError(
            f"--agentic-benchmark {selection!r} matched no EVALS_AGENTIC tasks. "
            f"Available for this model: {available}. Aliases: "
            f"{sorted(_AGENTIC_BENCHMARK_PREFIX_ALIASES)} + "
            f"{sorted(_AGENTIC_BENCHMARK_EXACT_ALIASES)}."
        )
    logger.info(
        "--agentic-benchmark %r selected %d of %d agentic task(s): %s",
        selection,
        len(selected),
        len(tasks),
        [task.task_name for task in selected],
    )
    return selected


def _select_tasks(tasks: list, runtime_config) -> list:
    """Apply --eval-samples / smoke-test task selection (real copy of
    ``run_evals._select_eval_config``, minus the EvalConfig wrapper)."""
    eval_samples = getattr(runtime_config, "eval_samples", None)
    if eval_samples and tasks:
        mapping = _parse_eval_samples_mapping(eval_samples)
        if mapping:
            requested = set(mapping.keys())
            filtered = [t for t in tasks if t.task_name in requested]
            if not filtered:
                available = sorted({t.task_name for t in tasks})
                raise ValueError(
                    "--eval-samples specified task(s) "
                    f"{sorted(requested)} but none match this model's eval "
                    f"tasks {available}."
                )
            unknown = requested - {t.task_name for t in filtered}
            if unknown:
                logger.warning(
                    "--eval-samples references task(s) not configured for this "
                    "model: %s",
                    sorted(unknown),
                )
            logger.info(
                "--eval-samples filtering eval tasks down to: %s",
                [t.task_name for t in filtered],
            )
            return filtered

    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode != EvalLimitMode.SMOKE_TEST or not tasks:
        return tasks

    selected_task = tasks[0]
    logger.info(
        "Smoke-test mode enabled; running only first eval task: %s",
        selected_task.task_name,
    )
    return [selected_task]


def get_llm_eval_tasks(model_spec, runtime_config=None) -> List:
    """Return the standard eval tasks for ``model_spec`` (empty if none).

    Looks the model up in ``EVAL_CONFIGS`` by ``model_name``, drops non-standard
    (agentic/media) task venvs, then applies --eval-samples / smoke-test
    selection. Returns ``[]`` when the model has no standard eval tasks so the
    caller can no-op cleanly (e.g. a model with only agentic evals).
    """
    from reference_config.evals.eval_config import EVAL_CONFIGS

    eval_config = EVAL_CONFIGS.get(model_spec.model_name)
    if eval_config is None or not eval_config.tasks:
        logger.info("No EVAL_CONFIGS entry / tasks for model=%s", model_spec.model_name)
        return []

    standard = [
        t for t in eval_config.tasks if t.workflow_venv_type in _STANDARD_EVAL_VENVS
    ]
    if not standard:
        logger.info(
            "Model %s has eval tasks but none use a standard (lm-eval/lmms-eval) "
            "venv; nothing for the standard eval path to run.",
            model_spec.model_name,
        )
        return []

    return _select_tasks(standard, runtime_config)


__all__ = [
    "filter_tasks_by_min_context",
    "filter_agentic_tasks_by_benchmark",
    "get_llm_eval_tasks",
    "parse_agentic_benchmark",
]
