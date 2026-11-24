# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The vLLM LibriSpeech WER client's requests (no server)."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from test_module.eval_tests import openai_asr_wer


class _Response:
    def __init__(self, text, status=200):
        self.status = status
        self._text = text

    async def json(self):
        return {"text": self._text}

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, calls, status):
        self._calls = calls
        self._status = status

    def post(self, url, **kwargs):
        self._calls.append((url, kwargs))
        wav = {f[0]["name"]: f[2] for f in kwargs["data"]._fields}["file"]
        return _Response(f"text of {wav.decode()}", self._status)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch):
    calls = []

    def install(status=200):
        monkeypatch.setattr(
            aiohttp, "ClientSession", lambda **kwargs: _Session(calls, status)
        )
        return calls

    return install


def test_replies_come_back_in_input_order(server):
    calls = server()
    wavs = [b"a", b"b", b"c"]

    texts = asyncio.run(openai_asr_wer.transcribe_all("http://h", "m", wavs, 2))

    assert texts == ["text of a", "text of b", "text of c"]
    assert [url for url, _ in calls] == ["http://h/v1/audio/transcriptions"] * 3
    fields = {f[0]["name"]: f[2] for f in calls[0][1]["data"]._fields}
    assert fields["model"] == "m"
    assert fields["temperature"] == "0"


def test_a_failed_request_fails_the_eval(server):
    server(status=500)

    with pytest.raises(RuntimeError, match="transcription failed: 500"):
        asyncio.run(openai_asr_wer.transcribe_all("http://h", "m", [b"a"], 1))
