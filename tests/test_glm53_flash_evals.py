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
    "id_tt-lab-glm_GLM-5.3-Flash_blackhole_galaxy": 4,
    "id_tt-lab-glm_GLM-5.3-Flash_p150x8": 1,
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
        "aime25",
        "r1_math500",
        "mmlu_pro",
        "livecodebench",
        "humaneval_instruct",
        "mbpp_instruct",
        "leaderboard_ifeval",
        "ruler",
        "longbench2_generate",
        "tau3_bench_banking",
    ]
    assert TASKS["tau3_bench_banking"].workflow_venv_type == (
        WorkflowVenvType.EVALS_AGENTIC
    )


@pytest.mark.parametrize("name", LM_EVAL_TASKS)
def test_generation_fits_the_serve(name):
    task = TASKS[name]
    assert task.model_kwargs["max_length"] == GLM53_FLASH_CONTEXT
    gen = task.gen_kwargs
    assert int(gen["max_gen_toks"]) < GLM53_FLASH_CONTEXT
    if name != "ruler":
        # Chat with the model card's sampling and a reasoning budget.
        assert task.use_chat_api
        assert gen["until"] == ["<|endoftext|>"]
        assert (gen["temperature"], gen["top_p"]) == (1.0, 0.95)
        assert int(gen["max_gen_toks"]) >= 16384
    if name == "longbench2_generate":
        longest = task.custom_dataset_kwargs["maximum_isl"]
        assert longest + int(gen["max_gen_toks"]) <= GLM53_FLASH_CONTEXT


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
    ci = {
        name: TASKS[name].limit_samples_map.get(EvalLimitMode.CI_NIGHTLY)
        for name in LM_EVAL_TASKS
    }
    assert ci == {
        "gpqa_diamond_cot_zeroshot": 0.2,  # 40 of 198
        "aime25": 0.5,  # 15 of 30
        "r1_math500": 0.1,  # 50 of 500
        "mmlu_pro": 0.02,  # about 240 of 12,032, per subject
        "livecodebench": 30,  # of the 131-problem v6 window
        "humaneval_instruct": 0.2,  # 33 of 164
        "mbpp_instruct": 0.1,  # 50 of 500
        "leaderboard_ifeval": 0.2,  # 108 of 541
        "ruler": None,  # per length, by limit_factor
        "longbench2_generate": 2,
    }
    ruler = TASKS["ruler"].custom_dataset_kwargs
    assert ruler["max_seq_lengths"] == [4096, 32768, 131072]
    assert ruler["num_samples_per_length"] * ruler["limit_factor"] == pytest.approx(1)


def test_code_tasks_may_execute_code():
    for name in ("livecodebench", "humaneval_instruct", "mbpp_instruct"):
        assert TASKS[name].allow_code_execution
    for name in ("gpqa_diamond_cot_zeroshot", "leaderboard_ifeval", "ruler"):
        assert not TASKS[name].allow_code_execution


def test_published_scores_are_cited():
    published = {
        name: task.score.published_score
        for name, task in TASKS.items()
        if task.score.published_score is not None
    }
    assert published == {"gpqa_diamond_cot_zeroshot": 91.2, "tau3_bench_banking": 47.2}
    for name in published:
        assert "artificialanalysis.ai" in TASKS[name].score.published_score_ref


def test_tau3_calls_the_served_model():
    config = TASKS["tau3_bench_banking"].agentic_eval_config
    assert config.environment_env["TAU2_USER_MODEL"] == "openai/zai-org/GLM-5.3-Flash"
    assert config.verifier_env["TAU2_NL_ASSERTIONS_MODEL"] == (
        "openai/zai-org/GLM-5.3-Flash"
    )
    assert config.n_concurrent_trials <= 4
