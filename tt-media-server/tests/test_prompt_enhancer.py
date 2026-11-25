# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the Ideogram 4 prompt enhancer's client, against a mock of
the magic-prompt API."""

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr
from utils.prompt_enhancer import (
    REJECTED,
    UNAVAILABLE,
    IdeogramMagicPrompt,
    PromptEnhancementError,
    get_ideogram_prompt_enhancer,
)

_URL = "https://enhancer.test/v1/ideogram-v4/magic-prompt"
_CAPTION = {"compositional_deconstruction": {"background": "b", "elements": []}}


def _enhancer(*responses, max_retries=2):
    """An enhancer whose calls get ``responses`` in turn: an httpx.Response,
    or an exception to raise. Returns (enhancer, calls, sleeps)."""
    responses = list(responses)
    calls, sleeps = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def sleep(seconds):
        sleeps.append(seconds)

    enhancer = IdeogramMagicPrompt(
        api_key="test-key",
        url=_URL,
        timeout_seconds=5,
        max_in_flight=2,
        max_retries=max_retries,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    return enhancer, calls, sleeps


def _ok(caption=_CAPTION):
    return httpx.Response(200, json={"json_prompt": caption, "aspect_ratio": "16x9"})


async def test_sends_the_prompt_and_ratio_and_returns_the_caption():
    enhancer, calls, _ = _enhancer(_ok())
    assert await enhancer.enhance("a cat", "16:9") == _CAPTION
    (request,) = calls
    assert str(request.url) == _URL
    assert request.headers["Api-Key"] == "test-key"
    assert json.loads(request.content) == {
        "text_prompt": "a cat",
        "aspect_ratio": "16x9",
    }


async def test_retries_429_after_retry_after():
    enhancer, calls, sleeps = _enhancer(
        httpx.Response(429, headers={"Retry-After": "3"}), _ok()
    )
    assert await enhancer.enhance("a cat", "1:1") == _CAPTION
    assert len(calls) == 2
    assert sleeps == [3.0]


async def test_retry_after_is_capped():
    enhancer, _, sleeps = _enhancer(
        httpx.Response(503, headers={"Retry-After": "600"}), _ok()
    )
    await enhancer.enhance("a cat", "1:1")
    assert sleeps == [30.0]


async def test_gives_up_after_the_retries():
    enhancer, calls, sleeps = _enhancer(
        httpx.ReadTimeout("slow"),
        httpx.Response(500),
        httpx.ConnectError("down"),
        max_retries=2,
    )
    with pytest.raises(PromptEnhancementError, match="unavailable") as error:
        await enhancer.enhance("a cat", "1:1")
    assert error.value.code == "prompt_enhancement_error"
    assert str(error.value) == UNAVAILABLE
    assert len(calls) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize(
    ("status", "message"), [(400, REJECTED), (401, UNAVAILABLE), (404, UNAVAILABLE)]
)
async def test_other_errors_fail_at_once(status, message):
    enhancer, calls, sleeps = _enhancer(httpx.Response(status))
    with pytest.raises(PromptEnhancementError) as error:
        await enhancer.enhance("a cat", "1:1")
    assert str(error.value) == message
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "response",
    [httpx.Response(200, text="not json"), httpx.Response(200, json=["a list"])],
)
async def test_a_malformed_body_is_no_caption(response):
    enhancer, _, _ = _enhancer(response)
    assert await enhancer.enhance("a cat", "1:1") is None


def test_needs_a_key():
    with pytest.raises(ValueError, match="IDEOGRAM_API_KEY"):
        IdeogramMagicPrompt(
            api_key="", url=_URL, timeout_seconds=5, max_in_flight=1, max_retries=0
        )


def test_off_is_none(monkeypatch):
    monkeypatch.setattr("utils.prompt_enhancer._enhancer", None)
    settings = SimpleNamespace(
        ideogram_prompt_enhancer="none", ideogram_api_key=SecretStr("")
    )
    assert get_ideogram_prompt_enhancer(settings) is None


def test_on_without_a_key_fails(monkeypatch):
    monkeypatch.setattr("utils.prompt_enhancer._enhancer", None)
    settings = SimpleNamespace(
        ideogram_prompt_enhancer="ideogram-api",
        ideogram_api_key=SecretStr(""),
        ideogram_magic_prompt_url=_URL,
        prompt_enhancer_timeout_seconds=5,
        prompt_enhancer_max_in_flight=1,
        prompt_enhancer_max_retries=0,
    )
    with pytest.raises(ValueError, match="IDEOGRAM_API_KEY"):
        get_ideogram_prompt_enhancer(settings)
