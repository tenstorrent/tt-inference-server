# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

import json
from types import SimpleNamespace

import pytest

from llm_module.agentic.harbor import _write_harbor_config
from llm_module.config import DriverContext, ServerConnection
from llm_module.drivers.agentic import build_harbor_config
from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import (
    MINI_SWE_AGENT_VERSION,
    _eval_config_map,
)

MODEL = "google/gemma-4-31B-it"
TASKS = _eval_config_map[MODEL].tasks


def test_only_requested_text_suites():
    assert [task.task_name for task in TASKS] == [
        "r1_gpqa_diamond",
        "swe_bench_verified",
        "terminal_bench_2_1",
    ]
    assert all(task.priority == "should" for task in TASKS)
    assert TASKS[0].score.published_score == 84.3
    assert TASKS[0].score.gpu_reference_score == 83.33
    for task in TASKS[1:]:
        assert task.score.published_score is None
        assert task.score.gpu_reference_score is None


@pytest.mark.parametrize("mode", ["smoke-test", "ci-nightly"])
def test_stock_common_route_exact_gpqa_cohort(tmp_path, monkeypatch, mode):
    monkeypatch.delenv("TT_EVAL_CLIENT_CONCURRENCY", raising=False)
    spec = SimpleNamespace(
        model_id="gemma-4-31B-it",
        model_name="gemma-4-31B-it",
        hf_model_repo=MODEL,
        device_model_spec=SimpleNamespace(
            max_context=262144, max_concurrency=8, eval_max_retries=0
        ),
    )
    command = build_eval_command(
        TASKS[0],
        spec,
        "P300x2",
        tmp_path,
        8000,
        runtime_config=SimpleNamespace(limit_samples_mode=mode),
    )
    assert json.loads(command[command.index("--samples") + 1]) == {
        "r1_gpqa_diamond": [
            152,
            189,
            46,
            87,
            52,
            181,
            31,
            185,
            50,
            187,
            153,
            135,
            81,
            93,
            7,
            42,
            24,
            168,
            20,
            28,
        ]
        if mode == "ci-nightly"
        else [152, 189, 46]
    }
    assert "--limit" not in command
    args = command[command.index("--model_args") + 1]
    for expected in (
        "num_concurrent=4",
        "timeout=14400",
        "max_length=262080",
        "revision=842da3794eaa0b77d5f08bae87a17459d91ff475",
    ):
        assert expected in args
    generation = command[command.index("--gen_kwargs") + 1]
    for expected in (
        "max_gen_toks=126976",
        "temperature=1.0",
        "top_p=0.95",
        "top_k=20",
    ):
        assert expected in generation
    assert json.loads(command[command.index("--request-body") + 1])[
        "chat_template_kwargs"
    ]["enable_thinking"]


@pytest.mark.parametrize("mode", ["smoke-test", "ci-nightly"])
@pytest.mark.parametrize(
    "index,names",
    [
        (1, ["pylint-dev__pylint-7080", "django__django-11265"]),
        (
            2,
            [
                "terminal-bench/llm-inference-batching-scheduler",
                "terminal-bench/mailman",
            ],
        ),
    ],
)
def test_stock_harbor_route_preserves_smoke_limits_and_task_resources(
    tmp_path, mode, index, names
):
    task = TASKS[index]
    if mode == "ci-nightly" and task.task_name == "swe_bench_verified":
        names = [
            "pylint-dev__pylint-7080",
            "django__django-11265",
            "psf__requests-1766",
            "pydata__xarray-4966",
            "astropy__astropy-14598",
        ]
    if mode == "ci-nightly" and task.task_name == "terminal_bench_2_1":
        names = [
            "terminal-bench/llm-inference-batching-scheduler",
            "terminal-bench/mailman",
            "terminal-bench/db-wal-recovery",
            "terminal-bench/winning-avg-corewars",
            "terminal-bench/feal-linear-cryptanalysis",
            "terminal-bench/fix-code-vulnerability",
            "terminal-bench/constraints-scheduling",
            "terminal-bench/sparql-university",
            "terminal-bench/dna-insert",
            "terminal-bench/video-processing",
        ]
    config = build_harbor_config(
        task,
        ServerConnection(base_url="http://localhost", service_port=8000, model=MODEL),
        DriverContext(output_dir=tmp_path),
        runtime_config=SimpleNamespace(limit_samples_mode=mode),
        n_tasks=len(names),
    )
    generated = json.loads(_write_harbor_config(config).read_text())
    assert generated["datasets"][0]["task_names"] == names
    assert generated["datasets"][0]["n_tasks"] == len(names)
    assert generated["n_concurrent_trials"] == 2
    assert "override_cpus" not in generated["environment"]
    assert "override_memory_mb" not in generated["environment"]
    agent = generated["agents"][0]
    assert agent["override_timeout_sec"] == 21600
    kwargs = agent["kwargs"]
    if index == 1:
        assert agent["env"]["BASH_ENV"] == "/root/.bashrc"
        assert kwargs["version"] == MINI_SWE_AGENT_VERSION == "2.2.8"
        assert kwargs["max_tokens"] == 32768
        model = kwargs["config"]["model"]["model_kwargs"]
        assert model["temperature"] == 1.0
    else:
        assert kwargs["store_all_messages"] is True
        assert kwargs["temperature"] == 1.0
        model = kwargs["llm_kwargs"]
        assert model["max_tokens"] == 32768
    assert model["timeout"] == 3600
    assert model["top_p"] == 0.95
    assert model["extra_body"] == {
        **({"return_token_ids": True} if index == 1 else {}),
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_only_prepared_candidate_enabled():
    from reference_config.evals.eval_config import EvalLimitMode

    assert [task.limit_samples_map[EvalLimitMode.CI_NIGHTLY] for task in TASKS] == [
        20,
        0,
        0,
    ]
    assert TASKS[1].agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "pylint-dev__pylint-7080",
        "django__django-11265",
        "psf__requests-1766",
        "pydata__xarray-4966",
        "astropy__astropy-14598",
    ]
