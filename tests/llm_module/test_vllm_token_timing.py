# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_module import vllm_token_timing


def _run(monkeypatch, events, *, status=200, output_len=2):
    clock = [0.0]
    monkeypatch.setattr(vllm_token_timing.time, "perf_counter", lambda: clock[0])
    recorded = {}

    class Content:
        async def iter_any(self):
            for at, body in events:
                clock[0] = at
                yield ("data: " + json.dumps(body) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

    class Response:
        content = Content()
        reason = "HTTP failure"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    Response.status = status

    class Session:
        def post(self, **kwargs):
            recorded.update(kwargs)
            return Response()

    endpoint = SimpleNamespace(
        _validate_api_url=lambda *a: None,
        _get_chat_content=lambda inp, **kw: [{"type": "text", "text": inp.prompt}],
        _update_payload_common=lambda payload, inp: payload.update(
            **inp.extra_body, ignore_eos=inp.ignore_eos
        ),
        _update_headers_common=lambda *a: None,
        RequestFuncOutput=lambda: SimpleNamespace(
            success=False, error="", ttft=0.0, latency=0.0, output_tokens=0, itl=[]
        ),
        StreamedResponseHandler=lambda: SimpleNamespace(
            add_chunk=lambda chunk: [chunk.decode().strip()]
        ),
    )
    inp = SimpleNamespace(
        api_url="http://localhost/v1/chat/completions",
        model_name=None,
        model="test-model",
        prompt="test prompt",
        # The random dataset's estimate can differ from server-side truncation.
        prompt_len=135,
        output_len=output_len,
        extra_body={"truncate_prompt_tokens": 128, "seed": 42},
        ignore_eos=True,
    )
    result = asyncio.run(
        vllm_token_timing.make_chat_request_func(endpoint)(inp, Session())
    )
    return result, recorded


def _events():
    return [
        (1, {"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
        (3, {"choices": [{"delta": {"content": "one"}}]}),
        (4, {"choices": [{"delta": {"content": ""}}]}),
        (5, {"choices": [{"delta": {"content": " two"}}]}),
        (8, {"choices": [{"delta": {}, "finish_reason": "length"}]}),
        (10, {"choices": [], "usage": {"prompt_tokens": 128, "completion_tokens": 2}}),
    ]


def test_role_finish_and_usage_do_not_count_as_tokens(monkeypatch):
    result, request = _run(monkeypatch, _events())
    assert result.success, result.error
    assert result.ttft == 3
    assert result.latency == 5
    assert (result.latency - result.ttft) / (result.output_tokens - 1) == 2
    assert result.itl == [2]
    assert result.generated_text == "one two"
    assert result.prompt_len == 128
    assert request["json"]["ignore_eos"] is True
    assert request["json"]["seed"] == 42


@pytest.mark.parametrize("missing", [1, 4, 5])
def test_missing_content_finish_or_usage_fails(monkeypatch, missing):
    events = _events()
    if missing == 1:
        events = [e for i, e in enumerate(events) if i not in (1, 3)]
    else:
        events.pop(missing)
    result, _ = _run(monkeypatch, events)
    assert not result.success


@pytest.mark.parametrize(
    "key,value", [("prompt_tokens", 127), ("completion_tokens", 1)]
)
def test_wrong_actual_token_lengths_fail(monkeypatch, key, value):
    events = _events()
    events[-1][1]["usage"][key] = value
    result, _ = _run(monkeypatch, events)
    assert not result.success
    assert "length does not match" in result.error


def test_in_stream_error_fails_even_after_content(monkeypatch):
    result, _ = _run(monkeypatch, _events() + [(11, {"error": {"message": "OOM"}})])
    assert not result.success
    assert "OOM" in result.error


def test_http_error_fails(monkeypatch):
    result, _ = _run(monkeypatch, [], status=503)
    assert not result.success


def test_combined_content_finish_and_usage_is_valid(monkeypatch):
    result, _ = _run(
        monkeypatch,
        [
            (3, {"choices": [{"delta": {"content": "one"}}]}),
            (
                5,
                {
                    "choices": [
                        {"delta": {"content": " two"}, "finish_reason": "length"}
                    ],
                    "usage": {"prompt_tokens": 128, "completion_tokens": 2},
                },
            ),
        ],
    )
    assert result.success, result.error
    assert result.ttft == 3
    assert result.latency == 5
