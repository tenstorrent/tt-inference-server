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
        impl="qwen38-27b-qb2",
        catalog_name="Shield",
    )
    benchmark_config = get_benchmark_config(spec)
    run_configs = get_llm_configs(spec, spec.device_type)

    assert spec.status == ModelStatusTypes.COMPLETE
    assert spec.impl.impl_name == "qwen38-27b-qb2"
    assert spec.device_model_spec.max_concurrency == 16
    assert spec.device_model_spec.max_tokens_all_users == 1_050_592
    assert spec.device_model_spec.vllm_args["max_num_seqs"] == "16"
    assert spec.device_model_spec.env_vars["EXTRA_MODELS_DIR"] == (
        "../../tt-metal/models/demos"
    )
    assert len(benchmark_config.tasks) == 3
    assert len(run_configs) == 24
    assert max(cfg.max_concurrency for cfg in run_configs) == 16
    assert max(cfg.isl for cfg in run_configs) == 131072
    graded = [cfg for cfg in run_configs if cfg.targets]
    assert [(cfg.isl, cfg.osl, cfg.max_concurrency) for cfg in graded] == [
        (128, 128, 1)
    ]


def test_qwen38_release_has_one_result_per_requested_eval_suite():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks

    assert [task.task_name for task in tasks] == ["terminal_bench_2_1"]
    assert [task.workflow_venv_type for task in tasks] == [
        WorkflowVenvType.EVALS_AGENTIC
    ]

    terminal = tasks[0].agentic_eval_config
    assert terminal.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/compile-compcert",
        "terminal-bench/cobol-modernization",
        "terminal-bench/password-recovery",
        "terminal-bench/portfolio-optimization",
        "terminal-bench/qemu-startup",
    ]
    assert terminal.n_concurrent_trials == 5
    assert terminal.agent_timeout_sec == 6 * 60 * 60
    assert terminal.collect_server_metrics is True
    assert terminal.agent_kwargs["temperature"] == 1.0
    assert terminal.agent_kwargs["max_turns"] == 76
    assert terminal.agent_kwargs["model_info"]["max_output_tokens"] == 16 * 1024
    assert terminal.agent_kwargs["llm_kwargs"]["max_tokens"] == 16 * 1024
    assert terminal.agent_kwargs["llm_kwargs"]["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"reasoning_effort": "medium"},
    }
    assert terminal.task_overrides == {
        "terminal-bench/qemu-startup": {
            "path": "tasks/qemu-startup",
            "git_url": "https://github.com/mvasiljevicTT/terminal-bench-2-1.git",
            "git_commit_id": "a355fc6aaeaf62ba94b6cab023e179c7e440c651",
        }
    }


def test_qwen38_ci_eval_thresholds_are_exact_integer_counts():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    cases = [(tasks[0], 5, 80.0, 60.0)]

    for task, total, passing_score, failing_score in cases:
        reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
        assert reference["tolerance"] == 0.0
        assert accept_eval_score(reference, passing_score, n_total=total) is True
        assert accept_eval_score(reference, failing_score, n_total=total) is False
