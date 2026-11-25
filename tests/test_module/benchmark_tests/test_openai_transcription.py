# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The vLLM /v1/audio/transcriptions benchmark client (no server)."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest

from test_module.benchmark_tests import openai_transcription as client


class _Content:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line

        return gen()


class _Response:
    def __init__(self, status=200, lines=(), body=None):
        self.status = status
        self.content = _Content(list(lines))
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    def post(self, url, **kwargs):
        self._calls.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch):
    calls = []

    def install(response):
        monkeypatch.setattr(
            aiohttp, "ClientSession", lambda **kwargs: _Session(response, calls)
        )
        return calls

    return install


def _sse(*events):
    lines = [f"data: {json.dumps(e)}\n".encode() for e in events]
    return lines + [b"\n", b"data: [DONE]\n"]


def _chunk(text):
    return {"object": "transcription.chunk", "choices": [{"delta": {"content": text}}]}


USAGE = {
    "object": "transcription.chunk",
    "choices": [],
    "usage": {"completion_tokens": 40},
}


def _fields(form):
    return {f[0]["name"]: f[2] for f in form._fields}


def test_a_stream_is_a_multipart_upload_that_asks_for_usage(server):
    calls = server(_Response(lines=_sse(_chunk("Hello"), USAGE)))

    asyncio.run(client.transcribe_streaming("http://h", "m", b"mp3", 10.0))

    url, kwargs = calls[0]
    assert url == "http://h/v1/audio/transcriptions"
    fields = _fields(kwargs["data"])
    assert fields["file"] == b"mp3"
    assert fields["model"] == "m"
    assert fields["stream"] == "true"
    assert fields["stream_include_usage"] == "true"


def test_one_clip_reports_ttft_rtr_and_tokens_after_the_first_text(server):
    server(_Response(lines=_sse(_chunk(""), _chunk("Hello"), _chunk(" world"), USAGE)))

    t = asyncio.run(client.transcribe_streaming("http://h", "m", b"mp3", 28.8))

    assert t.ok
    assert 0 <= t.ttft <= t.elapsed
    assert t.rtr == pytest.approx(28.8 / t.elapsed)
    assert t.tsu == pytest.approx(40 / (t.elapsed - t.ttft))


def test_audio_split_into_several_requests_has_no_per_user_rate(server):
    server(_Response(lines=_sse(_chunk("Hello"), USAGE)))

    t = asyncio.run(client.transcribe_streaming("http://h", "m", b"mp3", 54.3))

    assert t.ok and t.rtr is not None
    assert t.tsu is None


def test_a_stream_without_text_fails(server):
    server(_Response(lines=_sse(USAGE)))

    assert not asyncio.run(client.transcribe_streaming("http://h", "m", b"", 3.0)).ok


def test_an_error_status_fails(server):
    server(_Response(status=400, body={"error": "bad"}))

    assert not asyncio.run(client.transcribe_streaming("http://h", "m", b"", 3.0)).ok


def test_the_duration_comes_from_a_verbose_json_request(server):
    calls = server(_Response(body={"duration": 28.795, "text": "x"}))

    assert asyncio.run(client.audio_duration("http://h", "m", b"mp3")) == 28.795
    assert _fields(calls[0][1]["data"])["response_format"] == "verbose_json"
