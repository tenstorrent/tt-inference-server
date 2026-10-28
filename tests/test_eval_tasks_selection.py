#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""--eval-tasks splits a model's eval set across runs without capping any task."""

from types import SimpleNamespace

import pytest

from llm_module.eval_configs import _select_tasks, eval_task_names, get_llm_eval_tasks
from llm_module.eval_command import _resolve_eval_samples
from workflow_module.engine_types import WorkflowVenvType


def _task(name):
    return SimpleNamespace(task_name=name, workflow_venv_type=None)


TASKS = [_task("aime25"), _task("gpqa_diamond_cot_zeroshot"), _task("mmlu_generative")]


def _rc(eval_tasks):
    return SimpleNamespace(
        eval_tasks=eval_tasks, eval_samples=None, limit_samples_mode=None
    )


def test_unset_keeps_every_task():
    assert _select_tasks(TASKS, _rc(None)) == TASKS
    assert eval_task_names(_rc("")) is None


def test_selects_named_tasks_in_configured_order():
    sel = _select_tasks(TASKS, _rc("mmlu_generative, aime25"))
    assert [t.task_name for t in sel] == ["aime25", "mmlu_generative"]


def test_an_unknown_task_name_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="does not configure"):
        _select_tasks(TASKS, _rc("aime25,aime26"))


def test_selected_tasks_keep_every_sample():
    """No --samples filter is produced: the split never caps a task."""
    task = SimpleNamespace(task_name="aime25", workflow_venv_type=None)
    assert _resolve_eval_samples(task, _rc("aime25")) is None


def _spec(modalities):
    return SimpleNamespace(
        hf_model_repo="org/model",
        model_name="model",
        supported_modalities=modalities,
        metadata={},
    )


def _venv_task(name, venv):
    return SimpleNamespace(task_name=name, workflow_venv_type=venv)


def test_image_eval_tasks_run_only_where_the_profile_serves_images(monkeypatch):
    """An lmms-eval (EVALS_VISION) task sends images: a profile narrowed to text
    skips it, a profile that declares the image modality keeps it."""
    tasks = [
        _venv_task("gpqa", WorkflowVenvType.EVALS_COMMON),
        _venv_task("mmmu_pro", WorkflowVenvType.EVALS_VISION),
    ]
    pack = SimpleNamespace(
        eval_config=lambda repo: SimpleNamespace(tasks=tasks),
        resolve_eval_task_for_device=lambda t, device: t,
    )
    import workflow_module.target_pack as target_pack

    monkeypatch.setattr(target_pack, "get_target_pack", lambda: pack)
    text_only = get_llm_eval_tasks(_spec(["text"]), _rc(None))
    assert [t.task_name for t in text_only] == ["gpqa"]
    with_images = get_llm_eval_tasks(_spec(["text", "image"]), _rc(None))
    assert [t.task_name for t in with_images] == ["gpqa", "mmmu_pro"]


def test_gemma4_12b_gpu_profile_runs_the_image_eval(monkeypatch):
    """The 12B GPU reference profile serves every modality of the checkpoint, so
    MMMU-Pro is in its eval set and --eval-tasks can select it (a text-only profile
    would make that name an error)."""
    from reference_config.evals.eval_config import _eval_config_map
    from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
    from workflows.utils import get_repo_root_path
    import workflow_module.target_pack as target_pack

    # The catalog's eval configs, independent of which MODEL_SPECS catalog is active.
    monkeypatch.setattr(
        target_pack,
        "get_target_pack",
        lambda: SimpleNamespace(
            eval_config=lambda repo: _eval_config_map.get(repo),
            resolve_eval_task_for_device=lambda t, device: t,
        ),
    )
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    specs = [
        spec
        for template in templates
        if template.weights == ["google/gemma-4-12B-it"]
        for spec in template.expand_to_specs()
    ]
    gpu = resolve_model_spec(
        specs, model="google/gemma-4-12B-it", device="gpu", catalog_name="Shield"
    )
    assert gpu.supported_modalities == ["text", "image", "video", "audio"]
    names = [t.task_name for t in get_llm_eval_tasks(gpu, _rc(None))]
    assert "mmmu_pro_standard_cot" in names
    selected = get_llm_eval_tasks(gpu, _rc("mmmu_pro_standard_cot"))
    assert [t.task_name for t in selected] == ["mmmu_pro_standard_cot"]
