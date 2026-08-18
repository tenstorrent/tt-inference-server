#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
Smoke test for the C++ server's non-streaming /v1/chat/completions path.

The goal is to detect breakage of the stream=false code path, not to measure
its performance: `vllm bench serve` (which already gates the streaming path
in the test gate) hard-codes stream=true, so without this script we have no
regression coverage for the non-streaming response shape.

Assumes a running C++ server reachable at SERVER_BASE_URL (default
http://127.0.0.1:8000) — the test gate's `cpp-server-benchmarks` job starts
the mock backend before invoking this script.

Usage:
    python cpp_server/tests/test_non_streaming_chat.py
    pytest cpp_server/tests/test_non_streaming_chat.py -sv
"""

import concurrent.futures
import os
import sys

import requests

DEFAULT_API_KEY = "your-secret-key"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528"
REQUEST_TIMEOUT_SEC = 30
MAX_TOKENS = 16
CONCURRENT_REQUESTS = 4


def _base_url() -> str:
    return os.environ.get("SERVER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _auth_headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _build_payload() -> dict:
    return {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }


def _send_chat_completion() -> requests.Response:
    return requests.post(
        f"{_base_url()}/v1/chat/completions",
        json=_build_payload(),
        headers=_auth_headers(),
        timeout=REQUEST_TIMEOUT_SEC,
    )


def _assert_chat_completion_shape(body: dict) -> None:
    assert body.get("object") == "chat.completion", (
        f"object should be 'chat.completion', got {body.get('object')!r}"
    )
    assert body.get("id"), f"missing or empty 'id' field in {body}"
    assert body.get("model"), f"missing or empty 'model' field in {body}"

    choices = body.get("choices") or []
    assert len(choices) >= 1, f"expected at least one choice, got {choices}"
    choice = choices[0]
    assert choice.get("finish_reason"), f"missing finish_reason in choice {choice}"

    message = choice.get("message") or {}
    assert message.get("role") == "assistant", (
        f"choices[0].message.role should be 'assistant', got {message.get('role')!r}"
    )
    content = message.get("content")
    assert isinstance(content, str) and content, (
        f"choices[0].message.content should be a non-empty string, got {content!r}"
    )

    usage = body.get("usage") or {}
    assert usage.get("prompt_tokens", 0) > 0, f"usage missing prompt_tokens: {usage}"
    assert usage.get("completion_tokens", 0) > 0, (
        f"usage missing completion_tokens: {usage}"
    )


def test_non_streaming_returns_well_formed_chat_completion():
    """A single stream=false request returns an OpenAI chat.completion JSON body."""
    response = _send_chat_completion()

    assert response.status_code == 200, (
        f"expected HTTP 200, got {response.status_code}: {response.text}"
    )

    content_type = response.headers.get("Content-Type", "")
    assert "application/json" in content_type, (
        f"non-streaming response should be JSON, got Content-Type={content_type!r} "
        f"(SSE leaked into the non-streaming path?)"
    )

    _assert_chat_completion_shape(response.json())


def test_non_streaming_handles_light_concurrency():
    """A handful of parallel stream=false requests all succeed and are well-formed."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
        responses = list(
            pool.map(lambda _: _send_chat_completion(), range(CONCURRENT_REQUESTS))
        )

    failures = [
        f"request {i}: status={r.status_code} body={r.text[:200]}"
        for i, r in enumerate(responses)
        if r.status_code != 200
    ]
    assert not failures, "; ".join(failures)

    for i, response in enumerate(responses):
        try:
            _assert_chat_completion_shape(response.json())
        except AssertionError as exc:
            raise AssertionError(f"request {i}: {exc}") from exc


def main() -> int:
    tests = [
        (
            "returns_well_formed_chat_completion",
            test_non_streaming_returns_well_formed_chat_completion,
        ),
        (
            "handles_light_concurrency",
            test_non_streaming_handles_light_concurrency,
        ),
    ]
    print(f"Running non-streaming smoke tests against {_base_url()}")
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            failed.append(name)
        else:
            print(f"  PASS {name}")

    if failed:
        print(f"\n{len(failed)}/{len(tests)} tests failed: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
