# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from reference_config.benchmarking.benchmark_config import get_benchmark_config
from reference_config.evals.eval_config import (
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import EvalLimitMode, ModelStatusTypes, WorkflowVenvType


def _qwen38_spec():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    template = next(t for t in templates if t.weights == ["Qwen/Qwen3.8-27B"])
    return template.expand_to_specs()[0]


def _qwen36_on_qwen38_spec():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    template = next(
        t
        for t in templates
        if t.weights == ["Qwen/Qwen3.6-27B"]
        and t.impl.impl_name == "qwen38-autoport"
    )
    return template.expand_to_specs()[0]


def test_qwen38_release_is_complete_and_runs_only_128_128_target():
    template_spec = _qwen38_spec()
    spec = resolve_model_spec(
        [template_spec],
        model="Qwen/Qwen3.8-27B",
        device="p300x2",
        impl="qwen38-autoport",
        catalog_name="Shield",
    )
    config = get_benchmark_config(spec)

    assert spec.status == ModelStatusTypes.COMPLETE
    assert spec.impl.impl_name == "qwen38-autoport"
    assert len(config.tasks) == 1
    params = config.tasks[0].param_map[spec.device_type]
    assert [(p.isl, p.osl, p.max_concurrency, p.num_prompts) for p in params] == [
        (128, 128, 1, 8)
    ]


def test_qwen36_checkpoint_comparison_uses_qwen38_serving_profile():
    qwen38 = _qwen38_spec()
    qwen36 = _qwen36_on_qwen38_spec()

    assert qwen36.impl == qwen38.impl
    assert qwen36.status == qwen38.status == ModelStatusTypes.COMPLETE
    assert qwen36.has_builtin_warmup == qwen38.has_builtin_warmup

    qwen38_device = qwen38.device_model_spec
    qwen36_device = qwen36.device_model_spec
    assert qwen36_device.max_concurrency == qwen38_device.max_concurrency == 1
    assert qwen36_device.max_context == qwen38_device.max_context
    qwen38_vllm_args = dict(qwen38_device.vllm_args)
    qwen36_vllm_args = dict(qwen36_device.vllm_args)
    assert qwen38_vllm_args.pop("model") == "Qwen/Qwen3.8-27B"
    assert qwen36_vllm_args.pop("model") == "Qwen/Qwen3.6-27B"
    assert qwen36_vllm_args == qwen38_vllm_args
    assert qwen36_device.override_tt_config == qwen38_device.override_tt_config

    qwen38_env = dict(qwen38_device.env_vars)
    qwen36_env = dict(qwen36_device.env_vars)
    assert qwen38_env.pop("HF_MODEL") == "Qwen/Qwen3.8-27B"
    assert qwen36_env.pop("HF_MODEL") == "Qwen/Qwen3.6-27B"
    assert qwen36_env == qwen38_env

    qwen38_config = get_benchmark_config(qwen38)
    qwen36_config = get_benchmark_config(qwen36)
    assert qwen36_config.tasks == qwen38_config.tasks


def test_qwen36_checkpoint_comparison_uses_qwen38_eval_configuration():
    qwen38_tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    qwen36_tasks = _eval_config_map["Qwen/Qwen3.6-27B"].tasks

    assert qwen36_tasks == qwen38_tasks


def test_qwen38_release_has_one_result_per_requested_eval_suite():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks

    assert [task.task_name for task in tasks] == [
        "r1_gpqa_diamond",
        "terminal_bench_2_1",
        "swe_bench_verified",
    ]
    assert [task.workflow_venv_type for task in tasks] == [
        WorkflowVenvType.EVALS_COMMON,
        WorkflowVenvType.EVALS_AGENTIC,
        WorkflowVenvType.EVALS_AGENTIC,
    ]
    assert tasks[0].limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 10

    terminal = tasks[1].agentic_eval_config
    swe = tasks[2].swebench_eval_config
    assert len(terminal.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 5
    assert len(swe.instance_ids_map[EvalLimitMode.CI_NIGHTLY]) == 5
    assert terminal.n_concurrent_trials == swe.n_concurrent_trials == 1


def test_qwen38_ci_eval_thresholds_are_exact_integer_counts():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    cases = [
        (tasks[0], 10, 90.0, 80.0),
        (tasks[1], 5, 80.0, 60.0),
        (tasks[2], 5, 60.0, 40.0),
    ]

    for task, total, passing_score, failing_score in cases:
        reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
        assert reference["tolerance"] == 0.0
        assert accept_eval_score(reference, passing_score, n_total=total) is True
        assert accept_eval_score(reference, failing_score, n_total=total) is False
