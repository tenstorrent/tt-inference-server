# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""The standalone launcher isolates cache salts and rejects streaming errors."""

import asyncio
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

# The driver executes this file by path in the benchmark environment.
_spec = importlib.util.spec_from_file_location(
    "vllm_bench_cache_off",
    Path(__file__).resolve().parents[2] / "llm_module" / "vllm_bench_cache_off.py",
)
wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrapper)


@dataclasses.dataclass
class _Input:
    prompt: str
    extra_body: dict | None = None


def test_every_request_gets_a_distinct_salt_including_repeats():
    seen = []

    async def fake_request(request_func_input, session=None, pbar=None):
        seen.append(request_func_input.extra_body["cache_salt"])
        return "ok"

    registry = {"openai-chat": fake_request}
    wrapper.patch_request_funcs(registry)
    inp = _Input(
        prompt="p",
        extra_body={
            "cache_salt": "bench-isl128-osl128-c1-abc",
            "truncate_prompt_tokens": 128,
        },
    )
    for _ in range(
        3
    ):  # the test request and two measured requests with the same prompt/body
        assert asyncio.run(registry["openai-chat"](inp, None, None)) == "ok"
    assert len(set(seen)) == 3
    assert all(s.startswith("bench-isl128-osl128-c1-abc-") for s in seen)
    assert (
        inp.extra_body["cache_salt"] == "bench-isl128-osl128-c1-abc"
    )  # caller's dict untouched


def test_requests_without_a_salt_are_passed_through_unchanged():
    async def fake_request(request_func_input, session=None, pbar=None):
        return request_func_input.extra_body

    registry = {"openai": fake_request}
    wrapper.patch_request_funcs(registry)
    assert asyncio.run(
        registry["openai"](_Input(prompt="p", extra_body={"seed": 42}), None, None)
    ) == {"seed": 42}
    assert asyncio.run(registry["openai"](_Input(prompt="p"), None, None)) is None


def test_patching_is_idempotent():
    async def fake_request(request_func_input, session=None, pbar=None):
        return 1

    registry = {"openai": fake_request}
    wrapper.patch_request_funcs(registry)
    once = registry["openai"]
    wrapper.patch_request_funcs(registry)
    assert registry["openai"] is once


def _endpoint(message_batches):
    class Handler:
        def __init__(self):
            self.batches = iter(message_batches)
            self.chunks = []

        def add_chunk(self, chunk):
            self.chunks.append(chunk)
            return next(self.batches)

    return SimpleNamespace(StreamedResponseHandler=Handler, ASYNC_REQUEST_FUNCS={})


@pytest.mark.parametrize("after_content", [False, True])
@pytest.mark.parametrize(
    "error", [{"message": "context exhausted", "code": 500}, "OOM", {}]
)
def test_stream_errors_raise_even_after_successful_content(after_content, error):
    batches = [["data: " + json.dumps({"error": error})]]
    if after_content:
        batches.insert(0, ['data: {"choices":[{"delta":{"content":"hello"}}]}'])
    endpoint = _endpoint(batches)
    wrapper.patch_stream_errors(endpoint)
    handler = endpoint.StreamedResponseHandler()
    if after_content:
        assert handler.add_chunk(b"first") == batches[0]
    with pytest.raises(RuntimeError, match="Streaming error") as caught:
        handler.add_chunk(b"error")
    assert json.dumps(error) in str(caught.value)


def test_valid_stream_messages_and_parent_buffering_are_preserved():
    # The existing parser still owns chunk assembly; only complete messages
    # are inspected. Empty content and tool calls are not failures.
    complete = [
        ": keepalive",
        'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"id":"call-1"}]}}]}',
        'data: {"choices":[{"delta":{"content":"an error occurred in the story"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: {"usage":{"completion_tokens":1},"error":null}',
        "data: [DONE]",
    ]
    endpoint = _endpoint([[], complete])
    original = endpoint.StreamedResponseHandler
    wrapper.patch_stream_errors(endpoint)
    patched = endpoint.StreamedResponseHandler
    wrapper.patch_stream_errors(endpoint)
    assert endpoint.StreamedResponseHandler is patched
    handler = endpoint.StreamedResponseHandler()
    assert isinstance(handler, original)
    assert handler.add_chunk(b"part one") == []
    assert handler.add_chunk(b"part two") is complete
    assert handler.chunks == [b"part one", b"part two"]


def test_launcher_installs_error_check_before_invoking_vllm(monkeypatch):
    endpoint = _endpoint([['data: {"error":{"message":"server failed"}}']])
    lib = ModuleType("vllm.benchmarks.lib")
    lib.endpoint_request_func = endpoint
    cli = ModuleType("vllm.entrypoints.cli.main")
    called = []

    def main():
        with pytest.raises(RuntimeError, match="server failed"):
            endpoint.StreamedResponseHandler().add_chunk(b"response")
        called.append(True)

    cli.main = main
    monkeypatch.setitem(sys.modules, "vllm.benchmarks.lib", lib)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", cli)
    wrapper.main()
    assert called == [True]
