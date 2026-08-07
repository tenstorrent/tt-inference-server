# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llm_module.eval_command import build_eval_command
from llm_module.lm_eval_no_server_seed import _drop_server_seed
from reference_config.evals.eval_config import (
    EvalTask,
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.workflow_types import EvalLimitMode


def _eval_tasks():
    return _eval_config_map["google/diffusiongemma-26B-A4B-it"].tasks


def _task(task_name):
    return next(task for task in _eval_tasks() if task.task_name == task_name)


def _gpqa_task():
    return _task("gpqa_diamond_cot_zeroshot")


def _build_command(task):
    model_spec = SimpleNamespace(
        model_id="diffusiongemma-26B-A4B-it",
        model_name="diffusiongemma-26B-A4B-it",
        hf_model_repo="google/diffusiongemma-26B-A4B-it",
        device_model_spec=SimpleNamespace(
            max_context=262144,
            max_concurrency=1,
            eval_max_retries=0,
        ),
    )
    return build_eval_command(
        task,
        model_spec,
        "P300x2",
        "/tmp/evals",
        8000,
    )


def _command_gen_kwargs(command):
    raw = command[command.index("--gen_kwargs") + 1]
    return dict(item.split("=", 1) for item in raw.split(","))


def test_diffusiongemma_release_evals_are_gpqa_and_terminal_bench_only():
    assert [task.task_name for task in _eval_tasks()] == [
        "gpqa_diamond_cot_zeroshot",
        "terminal_bench_2_1",
    ]

    task = _gpqa_task()

    assert task.use_chat_api is True
    assert task.model_kwargs["max_length"] == 16384
    assert task.gen_kwargs["stream"] == "false"
    assert task.score.score_func_kwargs["result_keys"] == [
        "exact_match,flexible-extract"
    ]


def test_diffusiongemma_gpqa_gpu_reference_requires_more_than_67_percent():
    task = _gpqa_task()
    score = task.score

    assert score.gpu_reference_score == 70.0
    assert score.tolerance == pytest.approx(3 / 70)

    reference = resolve_eval_reference(score, None)
    assert accept_eval_score(reference, 132 / 198 * 100, n_total=198) is False
    assert accept_eval_score(reference, 133 / 198 * 100, n_total=198) is True


def test_diffusiongemma_nightly_gpqa_uses_the_same_quality_gate():
    task = _gpqa_task()
    reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)

    # A 5% GPQA subset has approximately ten questions, so its attainable
    # scores jump by ten points: 60% fails and 70% passes the >67% boundary.
    assert reference["is_subset_reference"] is False
    assert accept_eval_score(reference, 60.0, n_total=10) is False
    assert accept_eval_score(reference, 70.0, n_total=10) is True


def test_diffusiongemma_gpqa_reserves_whole_canvas_output_budget():
    task = _gpqa_task()

    assert task.gen_kwargs["max_gen_toks"] == 13824
    assert (
        task.gen_kwargs["max_gen_toks"]
        == (task.model_kwargs["max_length"] - 2432) // 256 * 256
    )


def test_diffusiongemma_gpqa_uses_neutral_model_owned_sampling_params():
    task = _gpqa_task()

    assert task.gen_kwargs["do_sample"] == "true"
    assert task.gen_kwargs["temperature"] == 1.0
    assert task.propagate_seed_to_gen_kwargs is False
    for unsupported_key in (
        "top_k",
        "top_p",
        "logprobs",
        "response_format",
        "bad_words",
    ):
        assert unsupported_key not in task.gen_kwargs


def test_diffusiongemma_gpqa_keeps_harness_seed_out_of_server_requests():
    command = _build_command(_gpqa_task())
    gen_kwargs = _command_gen_kwargs(command)

    assert gen_kwargs["do_sample"] == "true"
    assert gen_kwargs["temperature"] == "1.0"
    assert "seed" not in gen_kwargs
    assert command[command.index("--seed") + 1] == "42"
    assert command[0].endswith("/bin/python")
    assert command[1].endswith("llm_module/lm_eval_no_server_seed.py")


def test_diffusiongemma_gpqa_seed_wrapper_removes_adapter_seed():
    payload = {"model": "diffusiongemma", "seed": 42, "temperature": 1.0}

    assert _drop_server_seed(payload) == {
        "model": "diffusiongemma",
        "temperature": 1.0,
    }


def test_eval_task_propagates_seed_to_server_by_default():
    task = EvalTask(
        task_name="seeded_sampling",
        gen_kwargs={"do_sample": "true", "temperature": 1.0},
    )
    command = _build_command(task)

    assert _command_gen_kwargs(command)["seed"] == "42"
    assert command[command.index("--seed") + 1] == "42"
    assert command[0].endswith("/bin/lm_eval")


def test_diffusiongemma_terminal_bench_ci_is_small_and_single_request():
    task = _task("terminal_bench_2_1")
    config = task.agentic_eval_config

    assert config.n_concurrent_trials == 1
    assert config.n_attempts == 1
    assert config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/fix-git",
        "terminal-bench/openssl-selfsigned-cert",
        "terminal-bench/prove-plus-comm",
    ]
    assert config.agent_timeout_sec == 45 * 60
    assert config.agent_kwargs["temperature"] == 1.0
    assert config.agent_kwargs["model_info"] == {
        "max_input_tokens": 16 * 1024,
        "max_output_tokens": 4 * 1024,
    }
    assert config.agent_kwargs["llm_kwargs"] == {
        "top_p": 1.0,
        "max_tokens": 4 * 1024,
        "timeout": 15 * 60,
    }


def test_diffusiongemma_dev_catalog_maps_gpqa_into_release_evals():
    env = {**os.environ, "MODEL_SPECS_ENV": "dev"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from reference_config.evals.eval_config import EVAL_CONFIGS; "
                "assert 'diffusiongemma-26B-A4B-it' in EVAL_CONFIGS"
            ),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
