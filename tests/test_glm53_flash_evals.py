# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The GLM-5.3-Flash (tt_lab_glm) eval set and the lm-eval commands it builds per device."""

from types import SimpleNamespace

import pytest

from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    GLM53_FLASH_CONTEXT,
    EvalLimitMode,
)
from workflows.model_spec import (
    _MODEL_SPECS_DIR,
    get_model_spec_map,
    load_templates_from_yaml,
)
from workflows.workflow_types import WorkflowVenvType

MODEL_SPECS = get_model_spec_map(
    load_templates_from_yaml(_MODEL_SPECS_DIR / "dev" / "llm.yaml", env="dev")
)
DEVICES = {
    "id_tt-lab-glm_GLM-5.3-Flash_blackhole_galaxy": 8,
}
TASKS = {
    task.task_name: task for task in ALL_EVAL_CONFIGS["zai-org/GLM-5.3-Flash"].tasks
}
LM_EVAL_TASKS = [
    name
    for name, task in TASKS.items()
    if task.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC
]


def _model_args(command):
    return dict(
        part.split("=", 1)
        for part in command[command.index("--model_args") + 1].split(",")
    )


def test_task_set():
    assert list(TASKS) == [
        "gpqa_diamond_cot_zeroshot",
        "terminal_bench_2_1",
        "tau3_bench_banking",
    ]
    for name in ("terminal_bench_2_1", "tau3_bench_banking"):
        assert TASKS[name].workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC


def test_every_task_has_a_published_baseline():
    published = {name: task.score.published_score for name, task in TASKS.items()}
    assert published == {
        "gpqa_diamond_cot_zeroshot": 91.2,
        "terminal_bench_2_1": 84.3,
        "tau3_bench_banking": 47.2,
    }
    for task in TASKS.values():
        assert task.score.published_score_ref


@pytest.mark.parametrize("name", LM_EVAL_TASKS)
def test_generation_fits_the_serve(name):
    task = TASKS[name]
    assert task.model_kwargs["max_length"] == GLM53_FLASH_CONTEXT
    gen = task.gen_kwargs
    assert int(gen["max_gen_toks"]) < GLM53_FLASH_CONTEXT
    # Chat with the model card's sampling and a reasoning budget.
    assert task.use_chat_api
    assert gen["until"] == ["<|endoftext|>"]
    assert (gen["temperature"], gen["top_p"]) == (1.0, 0.95)
    assert int(gen["max_gen_toks"]) >= 16384


@pytest.mark.parametrize("spec_id, concurrency", DEVICES.items())
@pytest.mark.parametrize("name", LM_EVAL_TASKS)
def test_commands_run_as_many_requests_as_the_device_serves(
    tmp_path, spec_id, concurrency, name
):
    spec = MODEL_SPECS[spec_id]
    runtime = SimpleNamespace(
        limit_samples_mode="ci-nightly", eval_samples=None, eval_shard=None
    )
    command = build_eval_command(
        TASKS[name], spec, spec.device_type, tmp_path, 8000, runtime_config=runtime
    )
    assert _model_args(command)["num_concurrent"] == str(concurrency)
    api = "chat/completions" if TASKS[name].use_chat_api else "completions"
    assert _model_args(command)["base_url"].endswith(f"/v1/{api}")


def test_ci_sample_counts():
    # All of GPQA Diamond: at 40 questions the noise was wider than the tolerance.
    gpqa = TASKS["gpqa_diamond_cot_zeroshot"]
    assert gpqa.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0.999
    for name, count in (("terminal_bench_2_1", 5), ("tau3_bench_banking", 4)):
        config = TASKS[name].agentic_eval_config
        assert len(config.task_names_map[EvalLimitMode.CI_NIGHTLY]) == count


def test_agentic_trials_fit_the_galaxy():
    for name in ("terminal_bench_2_1", "tau3_bench_banking"):
        assert TASKS[name].agentic_eval_config.n_concurrent_trials <= 4
    model_info = TASKS["terminal_bench_2_1"].agentic_eval_config.agent_kwargs[
        "model_info"
    ]
    assert model_info["max_input_tokens"] <= GLM53_FLASH_CONTEXT


def test_tau3_calls_the_served_model():
    config = TASKS["tau3_bench_banking"].agentic_eval_config
    assert config.environment_env["TAU2_USER_MODEL"] == "openai/zai-org/GLM-5.3-Flash"
    assert config.verifier_env["TAU2_NL_ASSERTIONS_MODEL"] == (
        "openai/zai-org/GLM-5.3-Flash"
    )
