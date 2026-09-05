# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Select standard and agentic eval tasks for an LLM model."""

from __future__ import annotations

import logging
from typing import List

from workflow_module.engine_types import EvalLimitMode, WorkflowVenvType

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

# Aliases accepted by --agentic-benchmark. Keep selection beside context
# reachability so release planning and the runtime agentic runner cannot choose
# different task sets.
_AGENTIC_BENCHMARK_PREFIX_ALIASES = {
    "tau3": "tau3_bench_",
    "swebench": "swe_bench_",
}
_AGENTIC_BENCHMARK_EXACT_ALIASES = {
    "tb2.0": "terminal_bench_2",
    "tb2.1": "terminal_bench_2_1",
}


def parse_agentic_benchmark(value: str) -> tuple:
    """Parse --agentic-benchmark into task-name prefix and exact matchers."""
    prefixes: list[str] = []
    exacts = set()
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


def _positive_token_budget(value):
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def agentic_task_required_context(task):
    """Return a task's declared per-request context envelope, if known.

    SWE-bench declares its limits directly. TerminalBench places the same
    fields in ``agent_kwargs.model_info``. ``min_context_required`` remains an
    optional explicit floor and wins when it reserves more headroom than the
    harness input/output envelope.

    An incomplete legacy task has no derived envelope and remains reachable
    unless it declares ``min_context_required``. This preserves existing tasks
    while making every fully declared agentic contract context-aware.
    """
    explicit = _positive_token_budget(getattr(task, "min_context_required", None))
    max_input = max_output = None

    swebench = getattr(task, "swebench_eval_config", None)
    terminal = getattr(task, "agentic_eval_config", None)
    if swebench is not None:
        max_input = _positive_token_budget(getattr(swebench, "max_input_tokens", None))
        max_output = _positive_token_budget(
            getattr(swebench, "max_output_tokens", None)
        )
    elif terminal is not None:
        agent_kwargs = getattr(terminal, "agent_kwargs", None)
        model_info = (
            agent_kwargs.get("model_info") if isinstance(agent_kwargs, dict) else None
        )
        if isinstance(model_info, dict):
            max_input = _positive_token_budget(model_info.get("max_input_tokens"))
            max_output = _positive_token_budget(model_info.get("max_output_tokens"))

    envelope = (
        max_input + max_output
        if max_input is not None and max_output is not None
        else None
    )
    if explicit is None:
        return envelope
    if envelope is None:
        return explicit
    return max(explicit, envelope)


def filter_reachable_agentic_tasks(tasks: list, model_spec) -> list:
    """Drop agentic tasks whose request envelope exceeds device max context."""
    available = getattr(
        getattr(model_spec, "device_model_spec", None), "max_context", None
    )
    if not isinstance(available, int) or isinstance(available, bool):
        return tasks

    selected = []
    for task in tasks:
        required = agentic_task_required_context(task)
        if required is not None and available < required:
            logger.info(
                "Skipping agentic eval task %s: device max_context=%d is below "
                "its required request context=%d",
                task.task_name,
                available,
                required,
            )
            continue
        selected.append(task)
    return selected


def select_agentic_eval_tasks(tasks: list, model_spec, runtime_config=None) -> list:
    """Apply agentic type, explicit benchmark, and context selection."""
    agentic = [
        task
        for task in tasks
        if task.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    ]
    selection = getattr(runtime_config, "agentic_benchmark", None)
    if isinstance(selection, str) and selection.strip():
        agentic = filter_agentic_tasks_by_benchmark(agentic, selection)
    return filter_reachable_agentic_tasks(agentic, model_spec)


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


def get_llm_eval_tasks(model_spec, runtime_config=None, device=None) -> List:
    """Return the standard eval tasks for ``model_spec`` (empty if none).

    Looks the model up in the registered target pack by ``model_name``, drops
    non-standard (agentic/media) task venvs, applies per-device tier-2/3
    overrides when ``device`` is given, then applies --eval-samples /
    smoke-test selection. Returns ``[]`` when the model has no standard eval
    tasks so the caller can no-op cleanly (e.g. a model with only agentic
    evals).
    """
    from workflow_module.target_pack import get_target_pack

    pack = get_target_pack()
    eval_config = pack.eval_config(model_spec.model_name)
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

    if device is not None:
        standard = [pack.resolve_eval_task_for_device(t, device) for t in standard]

    return _select_tasks(standard, runtime_config)


__all__ = [
    "agentic_task_required_context",
    "filter_agentic_tasks_by_benchmark",
    "filter_reachable_agentic_tasks",
    "get_llm_eval_tasks",
    "parse_agentic_benchmark",
    "select_agentic_eval_tasks",
]
