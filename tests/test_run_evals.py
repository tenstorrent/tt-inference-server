#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import importlib
from pathlib import Path
from types import SimpleNamespace

import types
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.eval_config import EvalConfig, EvalTask


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


def test_build_eval_command_uses_vllm_model_override_and_hf_tokenizer(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv("VLLM_MODEL", "console/deployed-model")
    monkeypatch.setenv("BASE_URL", "https://console.tenstorrent.com")
    monkeypatch.delenv("DEPLOY_URL", raising=False)

    task = EvalTask(
        task_name="first",
        use_chat_api=True,
        model_kwargs={
            "model": "test/repo",
            "base_url": "http://127.0.0.1:8000/v1/completions",
            "tokenizer_backend": "huggingface",
        },
    )
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
    assert "model=console/deployed-model," in model_args
    assert "tokenizer=test/repo," in model_args
    assert "base_url=https://console.tenstorrent.com/v1/chat/completions" in model_args
    assert "model=test/repo" not in model_args


def test_build_eval_command_translates_external_server_max_gen_toks(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv("BASE_URL", "https://console.tenstorrent.com")
    monkeypatch.delenv("DEPLOY_URL", raising=False)

    task = EvalTask(
        task_name="first",
        use_chat_api=True,
        gen_kwargs={
            "stream": "false",
            "max_gen_toks": "32768",
        },
    )
    model_spec = SimpleNamespace(
        hf_model_repo="test/repo",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(
        limit_samples_mode=None,
        docker_server=False,
        local_server=False,
    )

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="n150",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    gen_kwargs = cmd[cmd.index("--gen_kwargs") + 1]
    assert "max_tokens=32768" in gen_kwargs
    assert "max_gen_toks" not in gen_kwargs


def test_is_external_server_workflow_detects_client_side_runs(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)

    assert (
        run_evals._is_external_server_workflow(
            SimpleNamespace(docker_server=False, local_server=False)
        )
        is True
    )
    assert (
        run_evals._is_external_server_workflow(
            SimpleNamespace(docker_server=True, local_server=False)
        )
        is False
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


def test_configure_openai_api_key_uses_vllm_api_key_for_llm_workflows(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.setenv("VLLM_API_KEY", "vllm-api-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    args = SimpleNamespace(jwt_secret="")

    resolved = run_evals._configure_openai_api_key(
        args=args,
        model_type=run_evals.ModelType.LLM,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert resolved == "vllm-api-key"
    assert run_evals.os.environ["OPENAI_API_KEY"] == "vllm-api-key"


def test_configure_openai_api_key_uses_default_only_when_unset(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
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
