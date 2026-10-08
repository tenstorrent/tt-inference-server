# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from llm_module.benchmark_configs import get_llm_configs
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
        impl="gemma4-26b-a4b-qb2",
        catalog_name="Shield",
    )

    assert default.impl.impl_name == "tt-transformers"
    assert dedicated.impl.code_path == "models/demos/gemma4_26b_a4b_qb2"
    assert dedicated.status == ModelStatusTypes.FUNCTIONAL
    assert dedicated.device_model_spec.default_impl is False
    assert (
        dedicated.device_model_spec.env_vars["EXTRA_MODELS_DIR"]
        == "../../tt-metal/models/demos"
    )
    assert "TT_MODEL_CLASS_OVERRIDES" not in dedicated.device_model_spec.env_vars
    assert dedicated.device_model_spec.vllm_args["hf-overrides"] == (
        '{"architectures":["TTGemma4A4BForCausalLM"]}'
    )
    assert dedicated.device_model_spec.vllm_args["default-chat-template-kwargs"] == (
        '{"enable_thinking": true}'
    )
    assert dedicated.device_model_spec.vllm_args["reasoning-parser"] == "gemma4"


def test_gemma4_26b_release_uses_the_validated_serial_10_5_5_eval_cohort():
    tasks = _eval_config_map["google/gemma-4-26B-A4B-it"].tasks

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

    gpqa, terminal_task, swe_task = tasks
    assert gpqa.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 10
    assert gpqa.max_concurrent == 1

    terminal = terminal_task.agentic_eval_config
    swe = swe_task.agentic_eval_config
    assert len(terminal.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 5
    assert len(swe.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 5
    assert terminal.n_concurrent_trials == swe.n_concurrent_trials == 1
    assert terminal.n_attempts == swe.n_attempts == 1
    assert terminal.override_cpus == 16

    for task, total, passing_score, failing_score in [
        (gpqa, 10, 80.0, 70.0),
        (terminal_task, 5, 20.0, 0.0),
        (swe_task, 5, 20.0, 0.0),
    ]:
        reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
        assert reference["tolerance"] == 0.0
        assert accept_eval_score(reference, passing_score, n_total=total) is True
        assert accept_eval_score(reference, failing_score, n_total=total) is False


def test_gemma4_26b_release_includes_near_max_context_and_grades_only_128_128():
    dedicated = resolve_model_spec(
        _gemma4_26b_specs(),
        model="google/gemma-4-26B-A4B-it",
        device="p300x2",
        impl="gemma4-26b-a4b-qb2",
        catalog_name="Shield",
    )
    run_configs = get_llm_configs(dedicated, dedicated.device_type)
    graded = [config for config in run_configs if config.targets]

    assert any(
        (config.isl, config.osl, config.max_concurrency, config.num_prompts)
        == (261888, 128, 1, 1)
        and not config.targets
        for config in run_configs
    )

    assert [(config.isl, config.osl, config.max_concurrency) for config in graded] == [
        (128, 128, 1)
    ]
    functional = graded[0].targets["functional"]
    target = graded[0].targets["target"]
    assert functional.ttft_ms == 240.0
    assert round(functional.tput_user, 1) == 22.4
    assert round(functional.tput, 1) == 22.4
    assert (target.ttft_ms, target.tput_user, target.tput) == (24.0, 224.0, 224.0)
