# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Select the standard (lm-eval / lmms-eval) eval tasks for an LLM model."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import List

from workflow_module.engine_types import EvalLimitMode
from workflow_module.engine_types import WorkflowVenvType

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


def _apply_profile_request_body_overrides(tasks: list, model_spec) -> list:
    """Apply request bodies owned by the resolved model profile."""
    overrides = model_spec.metadata.get("eval_task_request_body_overrides", {})
    return [
        replace(task, request_body=overrides[task.task_name])
        if task.task_name in overrides
        else task
        for task in tasks
    ]


def eval_task_names(runtime_config):
    """Task names selected by --eval-tasks, or None (all tasks)."""
    raw = getattr(runtime_config, "eval_tasks", None)
    if not raw:
        return None
    names = [name.strip() for name in str(raw).split(",") if name.strip()]
    return names or None


def _select_tasks(tasks: list, runtime_config) -> list:
    """Apply --eval-tasks / --eval-samples / smoke-test task selection (real
    copy of ``run_evals._select_eval_config``, minus the EvalConfig wrapper).

    --eval-tasks keeps every sample of each named task: it splits one model's
    eval set across runs, it never caps a task. A name the model does not
    configure is an error, so a typo cannot silently drop a task from a split.
    """
    selected = eval_task_names(runtime_config)
    if selected and tasks:
        configured = {t.task_name for t in tasks}
        unknown = sorted(set(selected) - configured)
        if unknown:
            raise ValueError(
                f"--eval-tasks names task(s) {unknown} that this model does not "
                f"configure; its eval tasks are {sorted(configured)}."
            )
        filtered = [t for t in tasks if t.task_name in set(selected)]
        logger.info(
            "--eval-tasks selecting eval tasks: %s", [t.task_name for t in filtered]
        )
        return without_preset_skip(filtered, _get_limit_mode(runtime_config))
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
            return without_preset_skip(filtered, _get_limit_mode(runtime_config))

    limit_mode = _get_limit_mode(runtime_config)
    tasks = tasks_for_limit_mode(tasks, limit_mode)
    if limit_mode != EvalLimitMode.SMOKE_TEST or not tasks:
        return tasks

    selected_task = tasks[0]
    logger.info(
        "Smoke-test mode enabled; running only first eval task: %s",
        selected_task.task_name,
    )
    return [selected_task]


def _preset_skips(task, limit_mode) -> bool:
    """True when ``task.limit_samples_map`` carries 0 for ``limit_mode``."""
    limit = (getattr(task, "limit_samples_map", None) or {}).get(limit_mode)
    return limit is not None and not isinstance(limit, bool) and limit == 0


def without_preset_skip(tasks, limit_mode) -> List:
    """Explicitly selected tasks (--eval-tasks / --eval-samples) run in full: a
    preset's 0 for the active mode is dropped from the copy handed on, so the
    eval command never renders ``--limit 0`` for a task the operator asked for."""
    if limit_mode is None:
        return tasks
    kept = []
    for t in tasks:
        if _preset_skips(t, limit_mode):
            logger.info(
                "Eval task %s is selected explicitly; the %s preset's skip (limit 0) "
                "does not apply, running the full set",
                t.task_name,
                limit_mode.name.lower().replace("_", "-"),
            )
            limits = {m: v for m, v in t.limit_samples_map.items() if m != limit_mode}
            t = replace(t, limit_samples_map=limits)
        kept.append(t)
    return kept


def tasks_for_limit_mode(tasks, limit_mode) -> List:
    """Drop tasks a limit preset excludes: a ``limit_samples_map`` value of 0
    for the active mode means "do not run this task in that preset" (the
    agentic driver already reads 0 as n_tasks=0 and skips). A task with no
    entry for the mode runs in full. Smoke-test and a run without a preset
    keep every task."""
    if limit_mode in (None, EvalLimitMode.SMOKE_TEST) or not tasks:
        return tasks
    kept = []
    for t in tasks:
        if _preset_skips(t, limit_mode):
            logger.info(
                "Skipping eval task %s: limit 0 for %s",
                t.task_name,
                limit_mode.name.lower().replace("_", "-"),
            )
        else:
            kept.append(t)
    return kept


def tasks_for_modalities(tasks, supported_modalities) -> List:
    """Drop tasks whose ``modalities`` the serving entry does not support.

    A task with ``modalities=None`` is text-only and always kept. A spec that
    declares no modalities is treated as text-only, so a vision task is
    skipped on it rather than sent to a server that rejects images.
    """
    supported = set(supported_modalities or ["text"])
    kept = []
    for t in tasks:
        needed = set(getattr(t, "modalities", None) or ())
        if needed <= supported:
            kept.append(t)
        else:
            logger.info(
                "Skipping eval task %s: needs modalities %s, entry supports %s",
                t.task_name,
                sorted(needed),
                sorted(supported),
            )
    return kept


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
    eval_config = pack.eval_config(model_spec.hf_model_repo)
    if eval_config is None or not eval_config.tasks:
        logger.info("No EVAL_CONFIGS entry / tasks for model=%s", model_spec.model_name)
        return []

    standard = [
        t for t in eval_config.tasks if t.workflow_venv_type in _STANDARD_EVAL_VENVS
    ]
    standard = tasks_for_modalities(
        standard, getattr(model_spec, "supported_modalities", None)
    )
    if not standard:
        logger.info(
            "Model %s has eval tasks but none use a standard (lm-eval/lmms-eval) "
            "venv; nothing for the standard eval path to run.",
            model_spec.model_name,
        )
        return []

    if device is not None:
        standard = [pack.resolve_eval_task_for_device(t, device) for t in standard]

    standard = _apply_profile_request_body_overrides(standard, model_spec)

    return _select_tasks(standard, runtime_config)


__all__ = [
    "get_llm_eval_tasks",
    "tasks_for_limit_mode",
    "tasks_for_modalities",
    "without_preset_skip",
]
