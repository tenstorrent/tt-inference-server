# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The console prompt upsampler's client, against a fake upsampler."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time

import httpx
import pytest
from PIL import Image
from utils.prompt_upsampler import (
    UpsamplerRejected,
    UpsamplerUnavailable,
    upsample,
)

from utils import prompt_upsampler


class FakeUpsampler:
    """An httpx transport answering POST /v1/flux3/upsample: each call takes
    the next of ``responses`` (status, body, headers), or a caption of the
    prompt once they run out. ``hang`` seconds before answering."""

    def __init__(self, *responses, hang: float = 0.0):
        self.responses = list(responses)
        self.hang = hang
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.hang:
            await asyncio.sleep(self.hang)
        if self.responses:
            status, body, headers = self.responses.pop(0)
            return httpx.Response(status, json=body, headers=headers)
        prompt = json.loads(request.content)["prompt"]
        return httpx.Response(
            200,
            json={
                "caption": f"  A detailed caption of {prompt}. ",
                "finish_reason": "stop",
                "revision": "pup-test",
                "request_id": "upc_1",
            },
        )

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def upsampler(monkeypatch):
    """Points the client at a FakeUpsampler: upsampler(*responses, **kwargs)."""
    monkeypatch.setattr(
        prompt_upsampler.settings, "prompt_upsampler_url", "http://upsampler"
    )
    monkeypatch.setattr(prompt_upsampler.settings, "prompt_upsampler_token", "")
    monkeypatch.setattr(
        prompt_upsampler.settings, "prompt_upsampler_timeout_seconds", 5.0
    )
    monkeypatch.setattr(prompt_upsampler, "BACKOFF_SECONDS", 0.01)

    def make(*responses, hang: float = 0.0) -> FakeUpsampler:
        fake = FakeUpsampler(*responses, hang=hang)
        monkeypatch.setattr(prompt_upsampler, "_transport", fake.transport)
        return fake

    return make


def _error(status: int, message: str = "nope", retry_after: str = ""):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return (status, {"error": {"message": message, "type": "x"}}, headers)


def test_caption_and_request(upsampler, monkeypatch):
    monkeypatch.setattr(prompt_upsampler.settings, "prompt_upsampler_token", "sekret")
    fake = upsampler()
    result = asyncio.run(upsample("a cat", [Image.new("RGB", (1024, 1024), "red")]))
    assert result.caption == "A detailed caption of a cat."
    assert (result.revision, result.request_id, result.finish_reason) == (
        "pup-test",
        "upc_1",
        "stop",
    )
    (request,) = fake.requests
    assert request.url.path == "/v1/flux3/upsample"
    assert request.headers["Authorization"] == "Bearer sekret"
    (body,) = fake.bodies()
    assert body["prompt"] == "a cat"
    (uri,) = body["images"]
    assert uri.startswith("data:image/jpeg;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert (image.format, image.size) == ("JPEG", (1024, 1024))


def test_text_only_sends_no_images_and_no_token(upsampler):
    fake = upsampler()
    asyncio.run(upsample("a cat"))
    assert fake.bodies() == [{"prompt": "a cat", "images": []}]
    assert "Authorization" not in fake.requests[0].headers


def test_transient_errors_are_retried(upsampler):
    fake = upsampler(_error(503, retry_after="0.01"), _error(429))
    assert asyncio.run(upsample("a cat")).caption == "A detailed caption of a cat."
    assert len(fake.requests) == 3


def test_gives_up_after_the_retries(upsampler):
    fake = upsampler(*[_error(500)] * 5)
    with pytest.raises(UpsamplerUnavailable, match="HTTP 500; 3 attempts.*safe"):
        asyncio.run(upsample("a cat"))
    assert len(fake.requests) == 3


def test_does_not_wait_past_its_deadline(upsampler):
    fake = upsampler(_error(503, retry_after="60"))
    started = time.monotonic()
    with pytest.raises(UpsamplerUnavailable, match="1 attempt in"):
        asyncio.run(upsample("a cat"))
    assert time.monotonic() - started < 1 and len(fake.requests) == 1


def test_times_out(upsampler, monkeypatch):
    monkeypatch.setattr(
        prompt_upsampler.settings, "prompt_upsampler_timeout_seconds", 0.2
    )
    upsampler(hang=5)
    started = time.monotonic()
    with pytest.raises(UpsamplerUnavailable, match="timed out"):
        asyncio.run(upsample("a cat"))
    assert time.monotonic() - started < 2


def test_unreachable(upsampler, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(prompt_upsampler, "_transport", httpx.MockTransport(refuse))
    with pytest.raises(UpsamplerUnavailable, match="unreachable"):
        asyncio.run(upsample("a cat"))


@pytest.mark.parametrize("status", [401, 403])
def test_refused_credentials_are_unavailable_not_retried(upsampler, status):
    fake = upsampler(_error(status))
    with pytest.raises(UpsamplerUnavailable, match="credentials"):
        asyncio.run(upsample("a cat"))
    assert len(fake.requests) == 1


def test_a_rejected_request_is_not_retried(upsampler):
    fake = upsampler(_error(400, "the prompt is 9100 tokens; the limit is 8191"))
    with pytest.raises(UpsamplerRejected, match="HTTP 400.*9100 tokens"):
        asyncio.run(upsample("a cat"))
    assert len(fake.requests) == 1
    assert not hasattr(UpsamplerRejected, "error_code")


@pytest.mark.parametrize("body", [{"caption": "  "}, {"finish_reason": "stop"}])
def test_an_empty_caption_is_unavailable(upsampler, body):
    upsampler((200, body, {}))
    with pytest.raises(UpsamplerUnavailable, match="empty caption"):
        asyncio.run(upsample("a cat"))


def test_not_configured(monkeypatch):
    monkeypatch.setattr(prompt_upsampler.settings, "prompt_upsampler_url", "")
    with pytest.raises(UpsamplerUnavailable, match="not configured"):
        asyncio.run(upsample("a cat"))
    assert UpsamplerUnavailable.error_code == "upsampler_unavailable"
