# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path

from llm_module.config import LLMRunConfig, ServerConnection
from llm_module.drivers import vllm as vllm_driver
from llm_module.drivers.vllm import build_vllm_bench_serve_argv


def _config():
    return LLMRunConfig(isl=128, osl=128, max_concurrency=1, num_prompts=8)


def _result_path():
    return Path("/tmp/benchmark_out.json")


def test_remote_console_uses_base_url_and_skips_ready_check():
    server = ServerConnection(
        base_url="https://console.tenstorrent.com:443",
        service_port=443,
        model="moonshotai/Kimi-K2.6",
        auth_token="sk-test",
        is_remote=True,
    )
    cmd, auth_token = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert auth_token == "sk-test"
    assert cmd[cmd.index("--base-url") + 1] == "https://console.tenstorrent.com:443"
    assert cmd[cmd.index("--ready-check-timeout-sec") + 1] == "0"
    assert "--host" not in cmd
    assert "--port" not in cmd
    assert "--extra-body" not in cmd
    header_values = cmd[cmd.index("--header") + 1 :]
    assert "Accept-Encoding=identity" in header_values
    assert "Authorization=Bearer sk-test" in header_values


def test_local_server_uses_host_port_and_truncation():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="/venv/bin/vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert "--base-url" not in cmd
    assert '"truncate_prompt_tokens": "128"' in cmd[cmd.index("--extra-body") + 1]
    header_values = cmd[cmd.index("--header") + 1 :]
    assert header_values == ["Accept-Encoding=identity"]


def test_chat_endpoint_used_when_tokenizer_defines_a_chat_template(monkeypatch):
    monkeypatch.setattr(vllm_driver, "tokenizer_defines_chat_template", lambda s: True)
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/gemma-4-31B-it",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--backend") + 1] == "openai-chat"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/chat/completions"


def test_completions_endpoint_used_when_tokenizer_has_no_chat_template(monkeypatch):
    """A base checkpoint has no chat template, so the chat endpoint would fail its
    pre-flight probe with Bad Request (ChatTemplateResolutionError server-side)."""
    monkeypatch.setattr(vllm_driver, "tokenizer_defines_chat_template", lambda s: False)
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/gemma-4-31B",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--backend") + 1] == "vllm"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/completions"
    assert "openai-chat" not in cmd


def test_unloadable_tokenizer_keeps_the_chat_endpoint():
    """Unknown capability must not silently move every model to /v1/completions."""
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="definitely-not-a-real-org/definitely-not-a-real-model",
        is_remote=False,
    )
    assert vllm_driver.tokenizer_defines_chat_template(server) is True
