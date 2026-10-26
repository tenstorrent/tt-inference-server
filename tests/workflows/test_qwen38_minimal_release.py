# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from llm_module.benchmark_configs import get_llm_configs
from reference_config.benchmarking.benchmark_config import get_benchmark_config
from reference_config.evals.eval_config import _eval_config_map
from reference_config.evals.qwen38_goal12 import controls
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

    assert spec.status == ModelStatusTypes.EXPERIMENTAL
    assert spec.impl.impl_name == "qwen38-autoport"
    assert spec.device_model_spec.max_concurrency == 32
    assert spec.device_model_spec.max_tokens_all_users == 262144
    expected_resident_batch = template_spec.device_model_spec.vllm_args["max_num_seqs"]
    assert expected_resident_batch in (16, 32)
    assert spec.device_model_spec.vllm_args["max_num_seqs"] == expected_resident_batch
    assert spec.impl.code_path == "models/autoports/qwen_qwen3_8_27b"
    assert spec.device_model_spec.vllm_args["enable_prefix_caching"] is True
    assert len(benchmark_config.tasks) == 3
    assert len(run_configs) == 23
    assert max(cfg.max_concurrency for cfg in run_configs) == 32
    assert max(cfg.isl for cfg in run_configs) == 131072
    graded = [cfg for cfg in run_configs if cfg.targets]
    assert [(cfg.isl, cfg.osl, cfg.max_concurrency) for cfg in graded] == [
        (128, 128, 1)
    ]


def test_qwen38_release_has_one_result_per_requested_eval_suite():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks

    assert [task.task_name for task in tasks] == [
        "r1_gpqa_diamond",
        "terminal_bench_2",
        "swe_bench_verified",
    ]
    assert [task.workflow_venv_type for task in tasks] == [
        WorkflowVenvType.EVALS_COMMON,
        WorkflowVenvType.EVALS_AGENTIC,
        WorkflowVenvType.EVALS_AGENTIC,
    ]
    c = controls()
    assert tasks[0].limit_samples_map[EvalLimitMode.CI_NIGHTLY] == c["sample_count"]
    assert tasks[0].model_kwargs["timeout"] == c["gpqa_llm_timeout_sec"]
    assert tasks[0].wall_clock_timeout_seconds == c["gpqa_suite_timeout_sec"]

    terminal = tasks[1].agentic_eval_config
    swe = tasks[2].agentic_eval_config
    assert terminal.dataset == "terminal-bench/terminal-bench-2"
    assert terminal.dataset_ref == c["terminal_dataset_ref"]
    for task, agent in ((tasks[1], terminal), (tasks[2], swe)):
        assert agent.task_names_map[EvalLimitMode.CI_NIGHTLY] == c["task_names"][task.task_name]
        assert agent.n_concurrent_trials == c["max_concurrent"]
        assert agent.agent_timeout_sec == c["agent_timeout_sec"]
        assert agent.llm_timeout_sec == c["llm_timeout_sec"]
        assert agent.harbor_timeout_sec == c["suite_timeout_sec"]
    assert all(t.score.published_score is None and t.score.gpu_reference_score is None for t in tasks)
