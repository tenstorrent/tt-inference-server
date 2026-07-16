# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Select the standard (lm-eval / lmms-eval) eval tasks for an LLM model."""

from __future__ import annotations

import logging
from typing import List

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
    from evals.eval_config import EVAL_CONFIGS

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


__all__ = ["get_llm_eval_tasks"]
