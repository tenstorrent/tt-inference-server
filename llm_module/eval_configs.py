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


# The media an lmms-eval task sends when it declares no ``modalities`` of its
# own: EVALS_VISION tasks send images.
_VENV_MODALITIES = {WorkflowVenvType.EVALS_VISION: ("image",)}


def _task_modalities(task) -> set:
    declared = getattr(task, "modalities", None)
    if declared is None:
        declared = _VENV_MODALITIES.get(task.workflow_venv_type, ())
    return {str(m).lower() for m in declared}


_SCORE_OVERRIDE_KEYS = frozenset(
    {"gpu_reference_score", "gpu_reference_score_ref", "mode_reference_scores"}
)
_MODE_REFERENCE_KEYS = frozenset({"score", "ref", "tolerance"})


def _limit_mode_key(name, where):
    try:
        return EvalLimitMode.from_string(str(name))
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _apply_profile_score_overrides(tasks: list, model_spec) -> list:
    """Apply the eval gates owned by the resolved model profile.

    Two implementations of one checkpoint share the model's eval tasks, so a
    gate that is right for one profile lives in that profile's catalog
    metadata rather than on the task:

    - ``eval_task_score_overrides`` (per task name): ``gpu_reference_score``
      and ``gpu_reference_score_ref`` replace the task's full-set reference;
      ``mode_reference_scores`` ({mode name: {score, ref, tolerance}}) is
      merged per mode into the task's subset references (a null entry removes
      that mode's reference).
    - ``eval_task_limit_samples_overrides`` (per task name): {mode name: limit}
      merged per mode into the task's ``limit_samples_map`` (a null entry
      removes that mode's cap).

    Mode names resolve through ``EvalLimitMode.from_string`` (``ci_nightly``,
    ``CI-NIGHTLY``). An unknown key or mode raises ValueError naming it, so a
    typo cannot silently leave a profile on the shared gate. Task names the
    model does not configure, or that this run filtered out (an agentic task,
    an image task on a text-only profile), are left alone, as for the
    request-body overrides.
    """
    metadata = getattr(model_spec, "metadata", None) or {}
    score_overrides = metadata.get("eval_task_score_overrides") or {}
    limit_overrides = metadata.get("eval_task_limit_samples_overrides") or {}
    if not score_overrides and not limit_overrides:
        return tasks
    from reference_config.evals.eval_config import ModeReferenceScore

    profile = getattr(model_spec, "model_name", "profile")
    out = []
    for task in tasks:
        fields = {}
        override = score_overrides.get(task.task_name)
        if override is not None:
            where = f"eval_task_score_overrides[{task.task_name}] of {profile}"
            unknown = sorted(set(override) - _SCORE_OVERRIDE_KEYS)
            if unknown:
                raise ValueError(
                    f"{where}: unknown key(s) {unknown}; allowed keys are "
                    f"{sorted(_SCORE_OVERRIDE_KEYS)}"
                )
            if task.score is None:
                raise ValueError(f"{where}: the task carries no score to override")
            score_fields = {
                key: override[key]
                for key in ("gpu_reference_score", "gpu_reference_score_ref")
                if key in override
            }
            if "mode_reference_scores" in override:
                modes = dict(task.score.mode_reference_scores or {})
                for name, entry in (override["mode_reference_scores"] or {}).items():
                    mode = _limit_mode_key(name, where)
                    if entry is None:
                        modes.pop(mode, None)
                        continue
                    bad = sorted(set(entry) - _MODE_REFERENCE_KEYS)
                    if bad or "score" not in entry:
                        raise ValueError(
                            f"{where}: mode_reference_scores[{name}] needs 'score' "
                            f"and allows {sorted(_MODE_REFERENCE_KEYS)}; got "
                            f"{sorted(entry)}"
                        )
                    tolerance = entry.get("tolerance")
                    modes[mode] = ModeReferenceScore(
                        score=float(entry["score"]),
                        ref=str(entry.get("ref") or ""),
                        tolerance=None if tolerance is None else float(tolerance),
                    )
                score_fields["mode_reference_scores"] = modes
            fields["score"] = replace(task.score, **score_fields)
        limits = limit_overrides.get(task.task_name)
        if limits is not None:
            where = f"eval_task_limit_samples_overrides[{task.task_name}] of {profile}"
            limit_map = dict(task.limit_samples_map or {})
            for name, limit in limits.items():
                mode = _limit_mode_key(name, where)
                if limit is None:
                    limit_map.pop(mode, None)
                    continue
                if (
                    isinstance(limit, bool)
                    or not isinstance(limit, (int, float))
                    or limit <= 0
                ):
                    raise ValueError(
                        f"{where}: the limit for {name} must be a positive number, got {limit!r}"
                    )
                limit_map[mode] = limit
            fields["limit_samples_map"] = limit_map
        out.append(replace(task, **fields) if fields else task)
    return out


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
        return filtered
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
    overrides when ``device`` is given and the profile's own request-body and
    eval-gate overrides, then applies --eval-samples / smoke-test selection.
    Returns ``[]`` when the model has no standard eval tasks so the caller can
    no-op cleanly (e.g. a model with only agentic evals).
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
    # lmms-eval tasks send media (images unless the task declares its own
    # ``modalities``, e.g. audio). A serving profile that does not declare that
    # modality (a text-only device block of a multimodal entry) skips them, as
    # the benchmarks workflow skips its image sweep.
    served = {
        str(m).lower()
        for m in (getattr(model_spec, "supported_modalities", None) or [])
    }
    kept = []
    for t in standard:
        missing = sorted(_task_modalities(t) - served)
        if missing:
            logger.info(
                "Skipping eval task %s: %s serves no %s modality",
                t.task_name,
                model_spec.model_name,
                "/".join(missing),
            )
        else:
            kept.append(t)
    standard = kept
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
    standard = _apply_profile_score_overrides(standard, model_spec)

    from reference_config.evals.qwen38_gpqa_arm import applies, configure
    if applies(model_spec):
        standard = configure(standard, runtime_config)

    return _select_tasks(standard, runtime_config)


__all__ = ["get_llm_eval_tasks"]
