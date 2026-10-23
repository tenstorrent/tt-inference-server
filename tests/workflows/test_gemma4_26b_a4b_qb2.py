# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from llm_module.benchmark_configs import get_llm_configs
from llm_module.eval_configs import _apply_profile_request_body_overrides
from reference_config.evals.eval_config import (
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import EvalLimitMode, ModelStatusTypes, WorkflowVenvType


def _gemma4_26b_specs():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    return [
        spec
        for template in templates
        if template.weights == ["google/gemma-4-26B-A4B-it"]
        for spec in template.expand_to_specs()
    ]


def test_dedicated_gemma4_profile_is_opt_in_and_uses_the_demo_bundle():
    specs = _gemma4_26b_specs()
    default = resolve_model_spec(
        specs,
        model="google/gemma-4-26B-A4B-it",
        device="p300x2",
        catalog_name="Shield",
    )
    dedicated = resolve_model_spec(
        specs,
        model="google/gemma-4-26B-A4B-it",
        device="p300x2",
        impl="tt-vllm-plugin",
        catalog_name="Shield",
    )

    assert default.impl.impl_name == "tt-transformers"
    assert dedicated.impl.impl_name == "tt-vllm-plugin"
    assert dedicated.status == ModelStatusTypes.FUNCTIONAL
    assert dedicated.device_model_spec.default_impl is False
    assert dedicated.device_model_spec.max_concurrency == 1
    assert (
        dedicated.device_model_spec.env_vars["EXTRA_MODELS_DIR"]
        == "../../tt-metal/models/demos"
    )
    assert "TT_MODEL_CLASS_OVERRIDES" not in dedicated.device_model_spec.env_vars
    assert dedicated.device_model_spec.vllm_args["hf-overrides"] == (
        '{"architectures":["TTGemma4A4BForCausalLM"]}'
    )
    assert dedicated.device_model_spec.vllm_args["reasoning-parser"] == "gemma4"
    assert dedicated.device_model_spec.vllm_args["seed"] == 9472
    assert dedicated.device_model_spec.vllm_args["max_num_seqs"] == 1
    assert dedicated.device_model_spec.vllm_args["enable-prefix-caching"] is True
    assert dedicated.device_model_spec.env_vars["GEMMA4_PREFIX_CACHING"] == "1"
    assert dedicated.device_model_spec.env_vars["GEMMA4_PREFIX_TRACE_SUFFIX"] == "1"


def test_gemma4_profile_leaves_cache_paths_to_the_launch_environment():
    dedicated = resolve_model_spec(
        _gemma4_26b_specs(),
        model="google/gemma-4-26B-A4B-it",
        device="p300x2",
        impl="tt-vllm-plugin",
        catalog_name="Shield",
    )
    # Runtime spec env vars overwrite existing settings, including local-server
    # cache paths. Deployment-specific defaults do not belong in this profile.
    cache_vars = {"TT_METAL_CACHE", "HF_HOME", "HF_HUB_CACHE"}
    assert cache_vars.isdisjoint(dedicated.env_vars)
    assert cache_vars.isdisjoint(dedicated.device_model_spec.env_vars)


def test_gemma4_26b_release_eval_cohort():
    """The shared EvalConfig runs the model-card presets (full GPQA Diamond, the
    24-task Terminal-Bench wave, LiveCodeBench v6, MMMU-Pro) on every 26B profile;
    SWE-bench keeps its 5-instance list, 1-of-5 gate and testbed agent
    environment for runs without a preset."""
    tasks = _eval_config_map["google/gemma-4-26B-A4B-it"].tasks

    assert [task.task_name for task in tasks] == [
        "r1_gpqa_diamond",
        "terminal_bench_2",
        "livecodebench",
        "mmmu_pro_standard_cot",
        "swe_bench_verified",
    ]
    assert [task.workflow_venv_type for task in tasks] == [
        WorkflowVenvType.EVALS_COMMON,
        WorkflowVenvType.EVALS_AGENTIC,
        WorkflowVenvType.EVALS_COMMON,
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AGENTIC,
    ]

    gpqa, terminal_task, _lcb, _mmmu, swe_task = tasks
    # GPQA runs in full in both presets and is graded against the full-set GPU
    # reference, not a subset reference.
    assert EvalLimitMode.CI_NIGHTLY not in gpqa.limit_samples_map
    assert gpqa.seed == 42
    assert gpqa.score.gpu_reference_score == 80.81
    reference = resolve_eval_reference(gpqa.score, EvalLimitMode.CI_NIGHTLY)
    assert reference["is_subset_reference"] is False

    terminal = terminal_task.agentic_eval_config
    swe = swe_task.agentic_eval_config
    assert len(terminal.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 24
    assert terminal.n_concurrent_trials == 24
    assert len(swe.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 5
    assert swe.n_concurrent_trials == 1
    assert terminal.n_attempts == swe.n_attempts == 1
    assert terminal.override_cpus == 16
    assert swe.llm_timeout_sec == 30 * 60
    assert (
        swe.agent_kwargs["config"]["environment"]["env"]["CONDA_DEFAULT_ENV"]
        == "testbed"
    )
    assert terminal_task.request_body["seed"] == swe_task.request_body["seed"] == 9472
    # The presets skip SWE-bench (limit 0); its 1-of-5 gate applies to a run
    # that selects it without a preset.
    assert swe_task.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0
    reference = resolve_eval_reference(swe_task.score, EvalLimitMode.CI_NIGHTLY)
    assert reference["tolerance"] == 0.0
    assert accept_eval_score(reference, 20.0, n_total=5) is True
    assert accept_eval_score(reference, 0.0, n_total=5) is False


def test_gemma4_26b_tt_vllm_plugin_uses_validated_gpqa_transport():
    dedicated = resolve_model_spec(
        _gemma4_26b_specs(),
        model="google/gemma-4-26B-A4B-it",
        device="p300x2",
        impl="tt-vllm-plugin",
        catalog_name="Shield",
    )
    gpqa = _apply_profile_request_body_overrides(
        [_eval_config_map["google/gemma-4-26B-A4B-it"].tasks[0]], dedicated
    )[0]

    assert gpqa.request_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_gemma4_26b_release_grades_only_128_128_at_concurrency_one():
    dedicated = resolve_model_spec(
        _gemma4_26b_specs(),
        model="google/gemma-4-26B-A4B-it",
        device="p300x2",
        impl="tt-vllm-plugin",
        catalog_name="Shield",
    )
    run_configs = get_llm_configs(dedicated, dedicated.device_type)
    graded = [config for config in run_configs if config.targets]
    benchmark_keys = [
        (config.isl, config.osl, config.max_concurrency) for config in run_configs
    ]

    assert len(benchmark_keys) == len(set(benchmark_keys))
    assert all(config.max_concurrency == 1 for config in run_configs)

    assert [(config.isl, config.osl, config.max_concurrency) for config in graded] == [
        (128, 128, 1)
    ]
    functional = graded[0].targets["functional"]
    target = graded[0].targets["target"]
    assert functional.ttft_ms == 240.0
    assert round(functional.tput_user, 1) == 22.4
    assert round(functional.tput, 1) == 22.4
    assert (target.ttft_ms, target.tput_user, target.tput) == (24.0, 224.0, 224.0)
