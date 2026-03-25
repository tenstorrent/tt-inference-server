#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import types
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.eval_config import (
    TerminalBenchEvalConfig,
    EvalConfig,
    EvalTask,
    SWEbenchEvalConfig,
)
from workflows.workflow_types import WorkflowVenvType


def _import_run_evals(monkeypatch):
    base_strategy_module = types.ModuleType(
        "utils.media_clients.base_strategy_interface"
    )

    class _BaseMediaStrategy:
        pass

    base_strategy_module.BaseMediaStrategy = _BaseMediaStrategy
    monkeypatch.setitem(
        sys.modules,
        "utils.media_clients.base_strategy_interface",
        base_strategy_module,
    )

    media_factory_module = types.ModuleType("utils.media_clients.media_client_factory")

    class _MediaTaskType:
        EVALUATION = "evaluation"
        BENCHMARK = "benchmark"

    class _MediaClientFactory:
        @staticmethod
        def run_media_task(*args, **kwargs):
            return 0

    media_factory_module.MediaClientFactory = _MediaClientFactory
    media_factory_module.MediaTaskType = _MediaTaskType
    media_factory_module.STRATEGY_MAP = {}
    monkeypatch.setitem(
        sys.modules,
        "utils.media_clients.media_client_factory",
        media_factory_module,
    )
    monkeypatch.delitem(sys.modules, "evals.run_evals", raising=False)
    return importlib.import_module("evals.run_evals")


def test_select_eval_config_smoke_test_keeps_only_first_task(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    eval_config = EvalConfig(
        hf_model_repo="test/repo",
        tasks=[EvalTask(task_name="first"), EvalTask(task_name="second")],
    )
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    selected_config = run_evals._select_eval_config(eval_config, runtime_config)

    assert [task.task_name for task in selected_config.tasks] == ["first"]


def test_build_eval_command_smoke_test_uses_limit_three(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(task_name="first")
    model_spec = SimpleNamespace(hf_model_repo="test/repo", model_id="test-model")
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="n150",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    limit_index = cmd.index("--limit")
    assert cmd[limit_index + 1] == str(run_evals.SMOKE_TEST_EVAL_LIMIT)


def test_build_eval_command_uses_base_url_without_appending_service_port(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv(
        "BASE_URL",
        "https://cpp-server-mock-b0b73cbb.workload.tenstorrent.com",
    )
    monkeypatch.delenv("DEPLOY_URL", raising=False)

    task = EvalTask(task_name="first")
    model_spec = SimpleNamespace(hf_model_repo="test/repo", model_id="test-model")
    runtime_config = SimpleNamespace(limit_samples_mode=None)

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="n150",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    model_args = cmd[cmd.index("--model_args") + 1]
    assert (
        "base_url=https://cpp-server-mock-b0b73cbb.workload.tenstorrent.com/v1/completions"
        in model_args
    )


def test_build_eval_command_accepts_full_completions_base_url(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv(
        "BASE_URL",
        "https://cpp-server-mock-b0b73cbb.workload.tenstorrent.com/v1/completions",
    )
    monkeypatch.delenv("DEPLOY_URL", raising=False)

    task = EvalTask(task_name="first")
    model_spec = SimpleNamespace(hf_model_repo="test/repo", model_id="test-model")
    runtime_config = SimpleNamespace(limit_samples_mode=None)

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="n150",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    model_args = cmd[cmd.index("--model_args") + 1]
    assert (
        "base_url=https://cpp-server-mock-b0b73cbb.workload.tenstorrent.com/v1/completions"
        in model_args
    )


def test_configure_openai_api_key_preserves_existing_api_key(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv("API_KEY", "real-api-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args = SimpleNamespace(jwt_secret="")

    resolved = run_evals._configure_openai_api_key(
        args=args,
        model_type=run_evals.ModelType.LLM,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert resolved == "real-api-key"
    assert run_evals.os.environ["OPENAI_API_KEY"] == "real-api-key"


def test_configure_openai_api_key_uses_api_key_for_llm_workflows(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv("API_KEY", "llm-api-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args = SimpleNamespace(jwt_secret="")

    resolved = run_evals._configure_openai_api_key(
        args=args,
        model_type=run_evals.ModelType.LLM,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert resolved == "llm-api-key"
    assert run_evals.os.environ["OPENAI_API_KEY"] == "llm-api-key"


def test_configure_openai_api_key_uses_default_only_when_unset(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args = SimpleNamespace(jwt_secret="")

    resolved = run_evals._configure_openai_api_key(
        args=args,
        model_type=run_evals.ModelType.LLM,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert resolved == "your-secret-key"


def test_validate_generation_response_shape_rejects_empty_choices(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)

    try:
        run_evals._validate_generation_response_shape(
            response_json={"choices": []},
            endpoint="https://example.test/v1/chat/completions",
            use_chat_api=True,
        )
    except RuntimeError as exc:
        assert "non-empty 'choices' array" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty choices")


def test_validate_generation_endpoint_rejects_mock_like_empty_choices(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)

    class _FakeResponse:
        text = '{"choices":[]}'

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"choices": []}

    monkeypatch.setattr(
        run_evals.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(),
    )

    prompt_client = SimpleNamespace(
        headers={"Authorization": "Bearer your-secret-key"},
        completions_url="https://example.test/v1/completions",
        _get_api_base_url=lambda: "https://example.test/v1",
    )

    try:
        run_evals._validate_generation_endpoint(
            prompt_client=prompt_client,
            model_name="test/repo",
            use_chat_api=True,
        )
    except RuntimeError as exc:
        assert "non-empty 'choices' array" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty choices")


def test_build_agentic_eval_command_uses_harbor_and_vllm_base(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        agentic_eval_config=TerminalBenchEvalConfig(
            dataset="terminal-bench/terminal-bench-2",
            agent="terminus-2",
            n_tasks=5,
            override_cpus=32,
            override_memory_mb=49152,
            agent_kwargs={
                "temperature": 0.0,
                "llm_kwargs": {"top_p": 0.95, "extra_body": {"top_k": 20}},
            },
        ),
        limit_samples_map={},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode=None)

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[1].endswith("evals/agentic/run_agentic_eval.py")
    assert cmd[2] == "terminal-bench"
    assert cmd[cmd.index("--dataset") + 1] == "terminal-bench/terminal-bench-2"
    assert cmd[cmd.index("--agent") + 1] == "terminus-2"
    assert cmd[cmd.index("--model-name") + 1] == "openai/Qwen/Qwen3.6-27B"
    assert cmd[cmd.index("--n-tasks") + 1] == "5"
    assert cmd[cmd.index("--override-cpus") + 1] == "32"
    assert cmd[cmd.index("--override-memory-mb") + 1] == "49152"
    agent_kwargs_json = cmd[cmd.index("--agent-kwargs-json") + 1]
    agent_kwargs = json.loads(agent_kwargs_json)
    assert agent_kwargs["temperature"] == 0.0
    assert agent_kwargs["llm_kwargs"] == {"top_p": 0.95, "extra_body": {"top_k": 20}}
    assert cmd[cmd.index("--api-base") + 1] == "http://127.0.0.1:8000/v1"


def test_build_agentic_eval_command_uses_task_smoke_limit(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        agentic_eval_config=TerminalBenchEvalConfig(
            dataset="terminal-bench/terminal-bench-2",
            agent="terminus-2",
            n_tasks=5,
        ),
        limit_samples_map={run_evals.EvalLimitMode.SMOKE_TEST: 1},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[cmd.index("--n-tasks") + 1] == "1"


def test_build_agentic_eval_command_writes_harbor_config_for_agent_timeout(
    monkeypatch, tmp_path
):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        agentic_eval_config=TerminalBenchEvalConfig(
            dataset="terminal-bench/terminal-bench-2",
            agent="terminus-2",
            n_attempts=5,
            n_tasks=10,
            override_cpus=32,
            override_memory_mb=49152,
            timeout_multiplier=2.0,
            agent_timeout_sec=3 * 60 * 60,
            agent_kwargs={
                "temperature": 1.0,
            },
        ),
        limit_samples_map={},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode=None)

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path=tmp_path,
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[1].endswith("evals/agentic/run_agentic_eval.py")
    assert cmd[2] == "terminal-bench"
    assert cmd[cmd.index("--agent-timeout-sec") + 1] == str(3 * 60 * 60)
    assert cmd[cmd.index("--n-attempts") + 1] == "5"
    assert cmd[cmd.index("--timeout-multiplier") + 1] == "2.0"
    assert cmd[cmd.index("--n-tasks") + 1] == "10"
    assert cmd[cmd.index("--override-cpus") + 1] == "32"
    assert cmd[cmd.index("--override-memory-mb") + 1] == "49152"
    agent_kwargs = json.loads(cmd[cmd.index("--agent-kwargs-json") + 1])
    assert agent_kwargs["temperature"] == 1.0


def test_build_swebench_eval_command_uses_wrapper_and_task_limit(monkeypatch, tmp_path):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="swe_bench_verified",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        swebench_eval_config=SWEbenchEvalConfig(
            dataset_name="SWE-bench/SWE-bench_Verified",
            sweagent_subset="verified",
            dataset_split="test",
            n_concurrent_trials=2,
            max_workers=3,
            n_tasks=10,
            temperature=1.0,
            top_p=0.95,
            max_input_tokens=200 * 1024,
            max_output_tokens=32 * 1024,
            completion_kwargs={"extra_body": {"top_k": 20}},
        ),
        limit_samples_map={run_evals.EvalLimitMode.SMOKE_TEST: 2},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path=tmp_path,
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[0].endswith(".venv_evals_agentic/bin/python")
    assert cmd[1].endswith("evals/agentic/run_agentic_eval.py")
    assert cmd[2] == "swebench"
    assert cmd[cmd.index("--task-name") + 1] == "swe_bench_verified"
    assert cmd[cmd.index("--dataset-name") + 1] == "SWE-bench/SWE-bench_Verified"
    assert cmd[cmd.index("--sweagent-subset") + 1] == "verified"
    assert cmd[cmd.index("--agent-backend") + 1] == "mini-swe-agent"
    assert cmd[cmd.index("--model-name") + 1] == "openai/Qwen/Qwen3.6-27B"
    assert cmd[cmd.index("--api-base") + 1] == "http://127.0.0.1:8000/v1"
    assert cmd[cmd.index("--mini-config") + 1] == "swebench.yaml"
    assert cmd[cmd.index("--mini-model-class") + 1] == "litellm"
    assert cmd[cmd.index("--mini-environment-class") + 1] == "docker"
    assert cmd[cmd.index("--n-concurrent-trials") + 1] == "2"
    assert cmd[cmd.index("--max-workers") + 1] == "3"
    assert cmd[cmd.index("--n-tasks") + 1] == "2"
    assert cmd[cmd.index("--max-input-tokens") + 1] == str(200 * 1024)
    assert cmd[cmd.index("--max-output-tokens") + 1] == str(32 * 1024)
    assert (
        cmd[cmd.index("--completion-kwargs-json") + 1]
        == '{"extra_body": {"top_k": 20}}'
    )


class TestClampMaxGenToks:
    """#3533 Problem 6: clamp eval-client max_gen_toks to fit within the
    server's max_context. Tasks tuned for a model's full context (e.g. Qwen3
    with max_gen_toks=32768 assuming 65K) otherwise over-subscribe a forge
    entry with smaller max_context and trigger 100% server-side rejection."""

    def test_clamps_when_max_gen_toks_exceeds_ceiling(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        # max_context=4096 -> ceiling = max(256, 4096 - 1024) = 3072.
        out = run_evals._clamp_max_gen_toks(
            {"max_gen_toks": 32768, "stream": "true"}, 4096, "task_x"
        )
        assert out["max_gen_toks"] == 3072
        assert out["stream"] == "true"

    def test_pass_through_when_within_ceiling(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"max_gen_toks": 256, "stream": "False"}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, 4096, "task_x")
        # Returns original dict unchanged (no copy needed).
        assert out is gen_kwargs

    def test_floor_protects_tiny_max_context(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        # max_context=512 -> 512 - 1024 < 0, floor of 256 kicks in.
        out = run_evals._clamp_max_gen_toks({"max_gen_toks": 32768}, 512, "task_x")
        assert out["max_gen_toks"] == 256

    def test_no_clamp_when_max_context_unset(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"max_gen_toks": 32768}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, None, "task_x")
        assert out is gen_kwargs

    def test_no_clamp_when_max_gen_toks_absent(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"stream": "False"}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, 4096, "task_x")
        assert out is gen_kwargs

    def test_non_numeric_max_gen_toks_passes_through(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"max_gen_toks": "not-a-number"}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, 4096, "task_x")
        assert out is gen_kwargs

    def test_string_numeric_max_gen_toks_clamps(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        # lm-eval task defs sometimes serialize max_gen_toks as a string.
        out = run_evals._clamp_max_gen_toks({"max_gen_toks": "32768"}, 4096, "task_x")
        assert out["max_gen_toks"] == 3072
