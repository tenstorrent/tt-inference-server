# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
from pathlib import Path

from llm_module.config import LLMRunConfig, ServerConnection
from llm_module.drivers.vllm import build_vllm_bench_serve_argv


def _config(**overrides):
    values = dict(isl=128, osl=128, max_concurrency=1, num_prompts=8)
    values.update(overrides)
    return LLMRunConfig(**values)


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
    assert json.loads(cmd[cmd.index("--extra-body") + 1]) == {
        "truncate_prompt_tokens": 128
    }
    assert "--trust-remote-code" not in cmd
    header_values = cmd[cmd.index("--header") + 1 :]
    assert header_values == ["Accept-Encoding=identity"]


def test_local_server_trusts_remote_code_when_spec_opts_in():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=False,
        tokenizer_trust_remote_code=True,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="/venv/bin/vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd.count("--trust-remote-code") == 1
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_remote_server_passes_trust_remote_code_once():
    server = ServerConnection(
        base_url="https://console.tenstorrent.com:443",
        service_port=443,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=True,
        tokenizer_trust_remote_code=True,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd.count("--trust-remote-code") == 1


def test_custom_dataset_path_switches_off_random():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(custom_dataset_path=Path("/tmp/speed_bench_prompts.jsonl")),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--dataset-name") + 1] == "custom"
    assert cmd[cmd.index("--dataset-path") + 1] == "/tmp/speed_bench_prompts.jsonl"
    assert cmd[cmd.index("--custom-output-len") + 1] == "128"
    assert "--disable-shuffle" in cmd
    assert "--skip-chat-template" in cmd
    assert "--random-input-len" not in cmd
    assert "--random-output-len" not in cmd


def test_goodput_constraints_passed_as_separate_tokens():
    # vllm bench serve defines --goodput with nargs="+": KEY:VALUE pairs in
    # milliseconds, one argv token each.
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(goodput="ttft:2000 tpot:20 e2el:20000"),
        server=server,
        result_filename=_result_path(),
    )
    idx = cmd.index("--goodput")
    assert cmd[idx + 1 : idx + 4] == ["ttft:2000", "tpot:20", "e2el:20000"]


def test_no_goodput_flag_without_constraints():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )
    assert "--goodput" not in cmd


def test_without_custom_dataset_the_sweep_stays_random():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--dataset-name") + 1] == "random"
    assert cmd[cmd.index("--random-input-len") + 1] == "128"
