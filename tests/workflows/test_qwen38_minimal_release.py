# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from llm_module.benchmark_configs import get_llm_configs
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


def test_qwen38_release_runs_full_benchmarks_but_grades_only_128_128():
    template_spec = _qwen38_spec()
    spec = resolve_model_spec(
        [template_spec],
        model="Qwen/Qwen3.8-27B",
        device="p300x2",
        impl="qwen38-autoport",
        catalog_name="Shield",
    )
    benchmark_config = get_benchmark_config(spec)
    run_configs = get_llm_configs(spec, spec.device_type)

    assert spec.status == ModelStatusTypes.COMPLETE
    assert spec.impl.impl_name == "qwen38-autoport"
    assert spec.device_model_spec.max_concurrency == 16
    assert spec.device_model_spec.max_tokens_all_users == 1_050_592
    assert spec.device_model_spec.vllm_args["max_num_seqs"] == "16"
    assert spec.device_model_spec.env_vars["QWEN_DECODE_BUCKETS"] == "1"
    assert len(spec.device_model_spec.tt_metal_source_ref) == 40
    assert spec.device_model_spec.tt_metal_source_paths == [
        "models/autoports/qwen_qwen3_8_27b",
    ]
    assert (
        spec.device_model_spec.vllm_plugin_source_repo
        == "https://github.com/mvasiljevicTT/vllm-tt-plugin.git"
    )
    assert spec.device_model_spec.vllm_plugin_source_ref == (
        "ba14de9216be50d21b42e4557c5ad1fbf2abd58c"
    )
    assert len(benchmark_config.tasks) == 3
    assert [
        (cfg.isl, cfg.osl, cfg.max_concurrency, cfg.num_prompts) for cfg in run_configs
    ] == [
        (128, 128, 1, 8),
        (128, 252, 1, 4),
        (1024, 252, 1, 4),
        (4096, 252, 1, 4),
        (16384, 252, 1, 2),
        (32768, 252, 1, 1),
        (65536, 252, 1, 1),
        (131072, 252, 1, 1),
        (261892, 252, 1, 1),
        (4096, 252, 8, 32),
        (32768, 252, 8, 8),
        (131072, 252, 8, 8),
        (4096, 252, 16, 64),
        (32768, 252, 16, 16),
    ]
    graded = [cfg for cfg in run_configs if cfg.targets]
    assert [(cfg.isl, cfg.osl, cfg.max_concurrency) for cfg in graded] == [
        (128, 128, 1)
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
    assert tasks[0].max_concurrent == 5
    assert terminal.n_concurrent_trials == swe.n_concurrent_trials == 5
    assert terminal.agent_timeout_sec == 6 * 60 * 60
    assert swe.llm_timeout_sec == 60 * 60
    assert swe.mini_container_timeout_sec == 8 * 60 * 60


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
