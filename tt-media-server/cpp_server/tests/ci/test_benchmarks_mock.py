# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Mock-backend benchmarks for the cpp_server.

Replaces the four mock-backend steps from .github/workflows/test-gate.yml's
`cpp-server-benchmarks` job:

  - Bench mock backend (vllm bench serve)
  - Smoke test mock backend (non-streaming)
  - Bench mock backend (structured output - json_object)
  - Bench mock backend (structured output - json_schema)

All four tests share a single mock-backend server instance via a module-scoped
fixture, mirroring the original job's start-server-once behavior.
"""

from __future__ import annotations

import concurrent.futures

import pytest
import requests

NON_STREAMING_CONCURRENCY = 4
NON_STREAMING_MAX_TOKENS = 16


@pytest.fixture(scope="module")
def mock_server(cpp_server_binary, cpp_server_dir):
    """One mock-backend server, shared by every test in this module.

    Module-scoped fixtures can't depend on the function-scoped `cpp_server`
    factory in conftest.py, so we drive _server.py directly here. Logs land
    under `_artifacts/<module-name>/mock_server.log` so they don't collide
    with the per-test artifact dirs that the other fixtures create.
    """
    from pathlib import Path

    from _server import start_server, stop_server, wait_for_ready

    artifacts_dir = (
        Path(__file__).resolve().parent / "_artifacts" / "test_benchmarks_mock"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    handle = start_server(
        name="mock_server",
        binary=cpp_server_binary,
        log_path=artifacts_dir / "mock_server.log",
        port=8000,
        env={"LLM_DEVICE_BACKEND": "mock"},
        cwd=cpp_server_dir,
    )
    try:
        wait_for_ready(handle, timeout=30.0)
        yield handle
    finally:
        stop_server(handle)


def test_bench_mock(mock_server, vllm_bench, assert_thresholds):
    result = vllm_bench(
        label="C++ Server vLLM Bench (mock)",
        base_url=mock_server.base_url,
        result_filename="vllm-bench-mock.json",
        log_filename="bench_mock.log",
    )
    assert_thresholds(result, mean_tpot_ms_max=1, mean_ttft_ms_max=150)


def test_non_streaming_smoke(mock_server, api_key):
    """A handful of `stream=false` requests succeed and return well-formed bodies.

    `vllm bench serve` hard-codes stream=true, so this is the only coverage we
    have for the non-streaming response shape.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": NON_STREAMING_MAX_TOKENS,
        "stream": False,
    }
    url = f"{mock_server.base_url}/v1/chat/completions"

    def _send():
        return requests.post(url, json=payload, headers=headers, timeout=30)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=NON_STREAMING_CONCURRENCY
    ) as pool:
        responses = list(pool.map(lambda _: _send(), range(NON_STREAMING_CONCURRENCY)))

    for i, response in enumerate(responses):
        assert response.status_code == 200, (
            f"request {i}: status={response.status_code} body={response.text[:200]}"
        )
        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"request {i}: non-streaming response should be JSON "
            f"(Content-Type={response.headers.get('Content-Type')!r}); "
            "did SSE leak into the non-streaming path?"
        )
        body = response.json()
        assert body.get("object") == "chat.completion", body
        choices = body.get("choices") or []
        assert choices, body
        message = choices[0].get("message") or {}
        assert message.get("role") == "assistant", message
        content = message.get("content")
        assert isinstance(content, str) and content, content
        usage = body.get("usage") or {}
        assert usage.get("prompt_tokens", 0) > 0, usage
        assert usage.get("completion_tokens", 0) > 0, usage


def test_bench_mock_structured_output_json_object(
    mock_server, vllm_bench, assert_thresholds
):
    result = vllm_bench(
        label="C++ Server vLLM Bench (structured output - json_object)",
        base_url=mock_server.base_url,
        result_filename="vllm-bench-structured-output-json-object.json",
        log_filename="bench_structured_output_json_object.log",
        extra_body={"response_format": {"type": "json_object"}},
    )
    assert_thresholds(result, mean_tpot_ms_max=5, mean_ttft_ms_max=150)


def test_bench_mock_structured_output_json_schema(
    mock_server, vllm_bench, assert_thresholds
):
    result = vllm_bench(
        label="C++ Server vLLM Bench (structured output - json_schema)",
        base_url=mock_server.base_url,
        result_filename="vllm-bench-structured-output-json-schema.json",
        log_filename="bench_structured_output_json_schema.log",
        extra_body={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "city": {"type": "string"},
                        },
                        "required": ["name", "age", "city"],
                        "additionalProperties": False,
                    },
                },
            }
        },
    )
    assert_thresholds(result, mean_tpot_ms_max=3, mean_ttft_ms_max=250)
