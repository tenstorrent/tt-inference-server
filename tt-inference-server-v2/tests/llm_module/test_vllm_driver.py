# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from dataclasses import replace

import pytest

from llm_module.config import LLMRunConfig, ServerConnection
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
    assert "--random-range-ratio" not in cmd
    assert "--ignore-eos" not in cmd


def test_random_range_and_ignore_eos_are_opt_in():
    server = ServerConnection(
        base_url="https://console.tenstorrent.com",
        service_port=443,
        model="moonshotai/Kimi-K2.6",
        is_remote=True,
    )
    config = replace(_config(), random_range_ratio=0.5, ignore_eos=True)

    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=config,
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--random-range-ratio") + 1] == "0.5"
    assert "--ignore-eos" in cmd


def test_local_random_range_allows_maximum_sampled_input_length():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )

    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=replace(_config(), random_range_ratio=0.5),
        server=server,
        result_filename=_result_path(),
    )

    assert '"truncate_prompt_tokens": "192"' in cmd[cmd.index("--extra-body") + 1]


@pytest.mark.parametrize("ratio", [-0.1, 1.0])
def test_random_range_ratio_must_be_supported(ratio):
    server = ServerConnection(
        base_url="https://console.tenstorrent.com",
        service_port=443,
        model="moonshotai/Kimi-K2.6",
        is_remote=True,
    )

    with pytest.raises(ValueError, match=r"random_range_ratio must be in \[0, 1\)"):
        build_vllm_bench_serve_argv(
            vllm_binary="vllm",
            config=replace(_config(), random_range_ratio=ratio),
            server=server,
            result_filename=_result_path(),
        )
