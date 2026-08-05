# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DiffusionGemma block-serving and request-admission regression gates."""

import json

import pytest
import requests

from test_fixtures.conftest import _get_bearer_token

CANVAS_LENGTH = 256
SIMPLE_MESSAGES = [
    {
        "role": "user",
        "content": "Think briefly, then answer the arithmetic question: 2 + 2.",
    }
]


def _auth_headers():
    headers = {"Content-Type": "application/json"}
    if token := _get_bearer_token():
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_root(endpoint_url):
    return endpoint_url.split("/v1/", maxsplit=1)[0]


def _assert_healthy(endpoint_url):
    response = requests.get(
        f"{_api_root(endpoint_url)}/health",
        headers=_auth_headers(),
        timeout=30,
    )
    response.raise_for_status()


def _completion_request(endpoint_url, model_name, prompt, *, max_tokens):
    return requests.post(
        f"{_api_root(endpoint_url)}/v1/completions",
        headers=_auth_headers(),
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        },
        timeout=1800,
    )


def _stream_response(response):
    reasoning_parts = []
    content_parts = []
    usage = None
    finish_reason = None

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            reasoning_parts.append(reasoning)
        if delta.get("content"):
            content_parts.append(delta["content"])
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    return {
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "usage": usage,
        "finish_reason": finish_reason,
    }


def test_single_canvas_sync_completion(report_test, api_client, endpoint_url):
    response = api_client(
        {
            "messages": SIMPLE_MESSAGES,
            "max_tokens": CANVAS_LENGTH,
            "ignore_eos": True,
        },
        timeout=600,
    )

    assert response["usage"]["completion_tokens"] == CANVAS_LENGTH
    message = response["choices"][0]["message"]
    assert (
        message.get("content")
        or message.get("reasoning_content")
        or message.get("reasoning")
    )
    _assert_healthy(endpoint_url)


def test_second_canvas_is_trimmed_at_request_limit(
    report_test, api_client, endpoint_url
):
    # 300 commits one full 256-token canvas and only 44 tokens from the next.
    # This catches both one-token scheduler accounting and missing stop trimming.
    response = api_client(
        {
            "messages": SIMPLE_MESSAGES,
            "max_tokens": 300,
            "ignore_eos": True,
        },
        timeout=900,
    )

    assert response["usage"]["completion_tokens"] == 300
    assert response["choices"][0]["finish_reason"] == "length"
    _assert_healthy(endpoint_url)


def test_streaming_separates_reasoning_and_final_answer(
    report_test, api_client, endpoint_url
):
    response = api_client(
        {
            "messages": SIMPLE_MESSAGES,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 1024,
        },
        stream=True,
        timeout=1200,
    )
    result = _stream_response(response)

    assert result["reasoning"], "thinking tokens were not exposed as reasoning"
    assert result["content"], "reasoning parser did not expose a final answer"
    assert result["usage"] is not None
    assert result["usage"]["completion_tokens"] <= 1024
    assert result["finish_reason"] in {"stop", "length"}
    _assert_healthy(endpoint_url)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("logprobs", True),
        ("response_format", {"type": "json_object"}),
        ("bad_words", ["forbidden"]),
        ("seed", 42),
        ("temperature", 0.5),
        ("top_p", 0.9),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
        ("n", 2),
    ],
)
def test_unsupported_sampling_parameter_is_4xx_without_engine_exit(
    report_test, api_client, endpoint_url, parameter, value
):
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        api_client(
            {
                "messages": SIMPLE_MESSAGES,
                "max_tokens": CANVAS_LENGTH,
                parameter: value,
            },
            timeout=60,
        )

    assert "400" in str(exc_info.value) or "422" in str(exc_info.value)
    _assert_healthy(endpoint_url)


def test_exact_context_and_canvas_fit(report_test, endpoint_url, max_context, request):
    model_name = request.config.getoption("--model-name")
    response = _completion_request(
        endpoint_url,
        model_name,
        [2] * (max_context - CANVAS_LENGTH),
        max_tokens=CANVAS_LENGTH,
    )
    response.raise_for_status()

    body = response.json()
    assert body["usage"]["prompt_tokens"] == max_context - CANVAS_LENGTH
    assert body["usage"]["completion_tokens"] == CANVAS_LENGTH
    _assert_healthy(endpoint_url)


def test_prompt_must_reserve_a_full_canvas(
    report_test, endpoint_url, max_context, request
):
    model_name = request.config.getoption("--model-name")
    response = _completion_request(
        endpoint_url,
        model_name,
        [2] * (max_context - CANVAS_LENGTH + 1),
        max_tokens=1,
    )

    assert response.status_code in {400, 422}, response.text
    assert "256" in response.text or "canvas" in response.text.lower()
    _assert_healthy(endpoint_url)
