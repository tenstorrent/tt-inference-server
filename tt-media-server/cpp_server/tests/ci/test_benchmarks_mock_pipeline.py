# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Mock pipeline backend benchmark for the cpp_server.

Replaces the `mock_pipeline` portion of test-gate.yml's `cpp-server-benchmarks`
job. Lives in its own file so the module-scoped server fixture isolates the
mock_pipeline server from the mock-backend server and they don't collide on
port 8000.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def mock_pipeline_server(cpp_server_binary, cpp_server_dir):
    from _server import start_server, stop_server, wait_for_ready

    artifacts_dir = (
        Path(__file__).resolve().parent / "_artifacts" / "test_benchmarks_mock_pipeline"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / "mock_pipeline_server.log"

    handle = start_server(
        name="mock_pipeline_server",
        binary=cpp_server_binary,
        log_path=log_path,
        port=8000,
        env={"LLM_DEVICE_BACKEND": "mock_pipeline"},
        cwd=cpp_server_dir,
    )
    try:
        wait_for_ready(handle, timeout=30.0)
        yield handle
    finally:
        stop_server(handle)


def test_bench_mock_pipeline(mock_pipeline_server, vllm_bench, assert_thresholds):
    result = vllm_bench(
        label="C++ Server vLLM Bench (mock_pipeline)",
        base_url=mock_pipeline_server.base_url,
        result_filename="vllm-bench-mock-pipeline.json",
        log_filename="bench_mock_pipeline.log",
        random_input_len=32,
        random_output_len=1024,
    )
    assert_thresholds(result, mean_tpot_ms_max=3, mean_ttft_ms_max=395)
