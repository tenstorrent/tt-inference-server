# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Prefill/decode split benchmarks for the cpp_server.

Replaces test-gate.yml's `cpp-server-prefill-decode-split` job:

  Round 1 (`test_split_plain`)
      Decode (port 8001, SOCKET_HOST=0.0.0.0)
      Prefill (port 8002, connects to decode at 127.0.0.1:9000)
      vllm bench against decode

  Round 2 (`test_split_with_mock_prefill_runner`)
      mock_prefill_runner.py (rank-0 coordinator)
      Prefill server (port 8001, LLM_DEVICE_BACKEND=prefill)
      Decode server (port 8000)
      Verify socket pairing via /tt-liveness
      vllm bench against decode

Each test starts its own servers via the function-scoped `cpp_server` factory;
teardown is automatic. Round 2 also cleans `/dev/shm/tt_ipc_*` before and after.
"""

from __future__ import annotations

import os

import pytest

SHM_SEGMENTS = ("/dev/shm/tt_ipc_p2c", "/dev/shm/tt_ipc_c2p")


def _clean_shm() -> None:
    for path in SHM_SEGMENTS:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            # Permission issue / device — best-effort cleanup, don't mask the
            # actual test failure.
            pass


@pytest.fixture
def shm_clean():
    """Clean /dev/shm/tt_ipc_* before and after the test."""
    _clean_shm()
    yield
    _clean_shm()


def test_split_plain(cpp_server, vllm_bench, assert_thresholds):
    """Round 1: plain prefill/decode split with no rank-0 runner."""
    decode = cpp_server(
        "decode_server",
        port=8001,
        env={
            "LLM_MODE": "decode",
            "SOCKET_HOST": "0.0.0.0",
            "SOCKET_PORT": "9000",
        },
    )
    cpp_server(
        "prefill_server",
        port=8002,
        env={
            "LLM_MODE": "prefill",
            "SOCKET_HOST": "127.0.0.1",
            "SOCKET_PORT": "9000",
        },
    )

    result = vllm_bench(
        label="C++ Server Prefill/Decode Split",
        base_url=decode.base_url,
        result_filename="vllm-bench-split-result.json",
        log_filename="bench_split.log",
    )
    assert_thresholds(result, mean_tpot_ms_max=10, mean_ttft_ms_max=650)


def test_split_with_mock_prefill_runner(
    cpp_server,
    cpp_server_dir,
    python_runner,
    vllm_bench,
    assert_thresholds,
    shm_clean,
):
    """Round 2: split with mock_prefill_runner.py acting as rank-0 coordinator."""
    runner_script = cpp_server_dir / "src" / "runners" / "mock_prefill_runner.py"
    assert runner_script.exists(), (
        f"mock_prefill_runner.py not found at {runner_script}"
    )

    python_runner(
        "mock_prefill_runner",
        runner_script,
        env={
            "TT_IPC_SHM_P2C": "tt_ipc_p2c",
            "TT_IPC_SHM_C2P": "tt_ipc_c2p",
            "LLM_DEVICE_BACKEND": "ttrun",
        },
        cwd=cpp_server_dir,
        settle_sec=2.0,
    )

    prefill = cpp_server(
        "prefill_server_runner",
        port=8001,
        env={
            "TT_IPC_SHM_C2P": "tt_ipc_c2p",
            "TT_IPC_SHM_P2C": "tt_ipc_p2c",
            "TT_LOG_LEVEL": "debug",
            "LLM_MODE": "prefill",
            "LLM_DEVICE_BACKEND": "prefill",
            "SOCKET_HOST": "127.0.0.1",
            "SOCKET_PORT": "9000",
        },
        require={"socket_status": "client:connected"},
        timeout=30.0,
    )
    decode = cpp_server(
        "decode_server_runner",
        port=8000,
        env={
            "LLM_MODE": "decode",
            "SOCKET_HOST": "0.0.0.0",
            "SOCKET_PORT": "9000",
            "TT_LOG_LEVEL": "debug",
        },
        require={"socket_status": "server:connected"},
        timeout=30.0,
    )

    # Cross-check both sides agree the socket is paired before benching.
    from _server import liveness  # local import to keep top-of-file deps minimal

    prefill_status = liveness(prefill)
    decode_status = liveness(decode)
    assert (
        prefill_status and prefill_status.get("socket_status") == "client:connected"
    ), f"prefill liveness: {prefill_status}"
    assert decode_status and decode_status.get("socket_status") == "server:connected", (
        f"decode liveness: {decode_status}"
    )

    result = vllm_bench(
        label="C++ Server Prefill/Decode Split (mock_prefill_runner)",
        base_url=decode.base_url,
        result_filename="vllm-bench-split-runner-result.json",
        log_filename="bench_split_runner.log",
    )
    assert_thresholds(result, mean_tpot_ms_max=10, mean_ttft_ms_max=650)
