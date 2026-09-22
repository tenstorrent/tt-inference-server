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
