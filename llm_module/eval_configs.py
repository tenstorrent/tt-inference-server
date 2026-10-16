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

_PROFILE_OVERRIDE_FIELDS = frozenset(
    {
        "apply_chat_template",
        "batch_size",
        "capture_reasoning",
        "custom_dataset_kwargs",
        "doc_ids",
        "fewshot_as_multiturn",
        "gen_kwargs",
        "log_samples",
        "max_attempts",
        "max_concurrent",
        "min_context_required",
        "model_kwargs",
        "propagate_seed_to_gen_kwargs",
        "request_body",
        "thinking",
        "thinking_kwarg",
        "use_chat_api",
        "wall_clock_timeout_seconds",
    }
)


def _apply_profile_task_overrides(tasks: list, model_spec, runtime_config) -> list:
    """Apply limit-mode eval transport overrides owned by one model profile."""
    limit_mode = _get_limit_mode(runtime_config)
    metadata = getattr(model_spec, "metadata", None) or {}
    configured = metadata.get("eval_task_mode_overrides") or {}
    if limit_mode is None or not configured:
        return tasks

    mode_overrides = None
    for mode_name, overrides in configured.items():
        if EvalLimitMode.from_string(mode_name) == limit_mode:
            mode_overrides = overrides
            break
    if not mode_overrides:
        return tasks

    task_names = {task.task_name for task in tasks}
    unknown_tasks = set(mode_overrides) - task_names
    if unknown_tasks:
        raise ValueError(
            f"Model profile {model_spec.model_id!r} defines eval overrides for "
            f"unknown tasks: {sorted(unknown_tasks)}"
        )

    resolved = []
    for task in tasks:
        overrides = mode_overrides.get(task.task_name)
        if not overrides:
            resolved.append(task)
            continue
        unsupported = set(overrides) - _PROFILE_OVERRIDE_FIELDS
        if unsupported:
            raise ValueError(
                f"Model profile {model_spec.model_id!r} cannot override "
                f"EvalTask fields {sorted(unsupported)} for {task.task_name!r}"
            )
        resolved.append(replace(task, **overrides))
        logger.info(
            "Applied model-profile eval overrides for model_id=%s task=%s "
            "limit_mode=%s: %s",
            model_spec.model_id,
            task.task_name,
            limit_mode.name,
            sorted(overrides),
        )
    return resolved


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
    eval_config = pack.eval_config(model_spec.hf_model_repo)
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

    standard = _apply_profile_task_overrides(standard, model_spec, runtime_config)

    return _select_tasks(standard, runtime_config)


__all__ = ["get_llm_eval_tasks"]
