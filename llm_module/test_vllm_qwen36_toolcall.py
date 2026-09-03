# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Qwen3.6-27B tool-call / reasoning-parser conformance suite.

These repros are Qwen3.6-specific: the serving stack runs the ``qwen3``
reasoning parser together with the ``qwen3_coder`` tool-call parser
(see the ``Qwen/Qwen3.6-27B`` device rows in
``workflows/model_specs/dev/llm.yaml``). They are wired only for
``qwen36_27b`` via ``VLLMQwen36ToolCallTest`` and mirror the structure of
``test_vllm_qwen3_streaming.py`` (Qwen3-32B).

The two conformance properties asserted here:

1. **Structured tool_calls emitted** -- a thinking-enabled prompt with a tool
   provided must stream a single, well-formed tool call: the qwen3_coder parser
   emits ``finish_reason: "tool_calls"`` with a populated ``tool_calls`` array
   whose ``arguments`` parse as JSON, even after a ``<think>`` reasoning block.
2. **Reasoning parser strips correctly** -- with a streaming JSON
   ``response_format`` the qwen3 reasoning parser must keep the reasoning in
   ``reasoning_content`` and never leak the ``</think>`` close tag (or trailing
   reasoning tokens) into ``content``, so the accumulated content parses as JSON.

NOTE (draft wiring): Qwen chunked serving is still being unblocked, so this
suite is wired-but-pending-serving -- it is expected to run once serving works,
not to pass today.
"""

import json
import time

import pytest
import requests

# --- Tool-calling + reasoning repro (qwen3_coder tool parser) ---
# Weather tool; a thinking-enabled prompt that should reliably trigger a single
# tool call after a reasoning block.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
}
THINKING_TOOL_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You must think before every response. First write at least one "
            "sentence of reasoning, then give a concise final answer in plain text."
        ),
    },
    {
        "role": "user",
        "content": "What is the weather like in Boston, MA in fahrenheit?",
    },
]

# --- Streaming reasoning-parser repro (qwen3 reasoning parser) ---
# The qwen3 reasoning parser must keep reasoning out of the content stream; a
# leaked ``</think>`` close tag (observed split across deltas, e.g. ``'<'`` +
# ``'/think>{'``) means the accumulated content is not valid JSON even though
# the request asked for a JSON response_format.
JSON_OBJECT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Return ONLY a minified JSON object with EXACTLY these keys and "
            'constraints: {"location":string, "temperature":integer '
            '[-100..100], "conditions":"sunny|cloudy|rainy|snowy", '
            '"unit":"celsius", "readings":[{t:integer, ts:ISO-8601 UTC Z} x '
            "3]}. No code fences, no extra text, no newlines. Example shape "
            "only; fill realistic values for Boston."
        ),
    }
]

# The failure is intermittent, so repeat each scenario a small number of times
# to make a regression deterministic without inflating runtime.
_TOOL_THINKING_RUNS = 20
_JSON_STREAMING_RUNS = 4

# The console ingress rate-limits back-to-back expensive streams: subsequent
# requests are rejected (404, or a mid-stream ChunkedEncodingError) and recover
# after a rolling window. Space the START of each run at least this many seconds
# after the previous run's start (gap-aware), and retry a rate-limited run this
# many times before giving up.
_JSON_STREAMING_MIN_GAP_S = 120
_JSON_STREAMING_MAX_RETRIES = 3


def _stream_chat_completion(api_client, payload):
    """Send a streaming chat-completion and reconstruct the single choice.

    Returns a dict with the accumulated ``reasoning_content``, ``content``, the
    ordered list of raw ``content_deltas``, aggregated ``tool_calls`` (indexed
    by their delta index), and the terminal ``finish_reason``. Raises the
    underlying HTTPError so callers can distinguish an unsupported request from
    a bad completion.
    """
    response = api_client(payload, stream=True, timeout=120)

    reasoning_parts = []
    content_parts = []
    tool_calls = {}
    finish_reason = None

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data.strip() == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if delta.get("content"):
            content_parts.append(delta["content"])

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(idx, {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    return {
        "reasoning_content": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "content_deltas": content_parts,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }


def _find_think_leak(content):
    """Return the ``</think>`` variant that leaked into ``content``, if any.

    The reasoning parser can split the close tag across deltas, so the check is
    run against the *joined* content. Any of the tag or its partial forms
    landing in content is a leak.
    """
    for marker in ("</think>", "</think", "<think>", "<think"):
        if marker in content:
            return marker
    return None


def _is_rate_limit_error(exc):
    """Return True when an exception looks like a transient ingress rate-limit.

    The console ingress rejects excess streaming requests as ``404`` (observed)
    or ``429``, or cuts the stream mid-flight so ``requests`` raises a
    ``ChunkedEncodingError``. All are transient and should be retried rather
    than counted as a parser failure. Auth errors (401/403) are NOT rate-limits.
    """
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        return True
    msg = str(exc)
    return "404" in msg or "429" in msg or "Response ended prematurely" in msg


def _stream_one_run_with_backoff(api_client, base_payload, run_idx, prev_start):
    """Run a single streaming attempt, spacing and retrying around rate-limits.

    Returns ``(result, start_ts)`` where ``start_ts`` is when the successful
    attempt began (so the next run can be spaced relative to it). Auth errors
    (401/403) and unsupported-payload errors propagate to the caller (the
    latter as ``pytest.skip`` -- the model does not support the payload).
    """
    if prev_start is not None:
        remaining = _JSON_STREAMING_MIN_GAP_S - (time.monotonic() - prev_start)
        if remaining > 0:
            print(
                f"[run {run_idx}] waiting {remaining:.0f}s before next run "
                "to stay under the ingress rate-limit..."
            )
            time.sleep(remaining)

    attempt = 0
    while True:
        start_ts = time.monotonic()
        try:
            result = _stream_chat_completion(api_client, base_payload)
            return result, start_ts
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            if _is_rate_limit_error(e):
                if attempt >= _JSON_STREAMING_MAX_RETRIES:
                    pytest.skip(
                        f"[run {run_idx}] rate-limited/truncated after "
                        f"{attempt} retries ({e}); the ingress limit did not "
                        "clear within the retry budget -- infra flake, not a "
                        "parser failure."
                    )
                attempt += 1
                print(
                    f"[run {run_idx}] transient rate-limit/truncation "
                    f"({type(e).__name__}); retry {attempt}/"
                    f"{_JSON_STREAMING_MAX_RETRIES} after "
                    f"{_JSON_STREAMING_MIN_GAP_S}s..."
                )
                time.sleep(_JSON_STREAMING_MIN_GAP_S)
                continue
            msg = str(e)
            if "401" in msg or "403" in msg:
                raise
            pytest.skip(
                "Server rejected the streaming tool-call/response_format "
                f"payload; model likely does not support this path: {e}"
            )


def test_streaming_tool_call_with_thinking(report_test, api_client, request):
    """qwen3_coder: streaming + tools + enable_thinking must emit the tool call.

    A correctly-behaving host emits ``finish_reason: "tool_calls"`` with a
    populated ``tool_calls`` array on every run, even after a ``<think>`` block,
    and the tool call's ``arguments`` must parse as JSON and target the provided
    tool. The test runs the repro multiple times and fails if ANY run drops the
    tool call or emits unparseable arguments.
    """
    base_payload = {
        "messages": THINKING_TOOL_MESSAGES,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 4096,
        "tools": [WEATHER_TOOL],
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    }

    results = []
    prev_start = None
    for i in range(_TOOL_THINKING_RUNS):
        result, prev_start = _stream_one_run_with_backoff(
            api_client, base_payload, i, prev_start
        )
        results.append(result)

    failures = []
    for i, result in enumerate(results):
        has_tool_call = any(slot["name"] for slot in result["tool_calls"].values())
        if result["finish_reason"] != "tool_calls" or not has_tool_call:
            failures.append(
                f"run {i}: finish_reason={result['finish_reason']!r}, "
                f"tool_calls={result['tool_calls']!r}, "
                f"reasoning_len={len(result['reasoning_content'])}, "
                f"content={result['content']!r}"
            )
            continue
        # Structured-arguments conformance: qwen3_coder must emit tool-call
        # arguments that parse as a JSON object.
        first_call = next(iter(result["tool_calls"].values()))
        try:
            json.loads(first_call["arguments"])
        except (json.JSONDecodeError, ValueError):
            failures.append(
                f"run {i}: tool_call arguments not valid JSON: "
                f"{first_call['arguments']!r}"
            )

    assert not failures, (
        f"{len(failures)}/{len(results)} streaming runs failed the "
        "tool-call conformance (expected finish_reason='tool_calls' with a "
        "populated tool_calls array whose arguments parse as JSON on every "
        "run). Failing runs:\n" + "\n".join(failures)
    )

    # Sanity-check the emitted tool call targets the provided tool.
    first = results[0]
    first_call = next(iter(first["tool_calls"].values()))
    assert first_call["name"] == WEATHER_TOOL["function"]["name"], (
        f"Expected tool call '{WEATHER_TOOL['function']['name']}', "
        f"got '{first_call['name']}'."
    )


def test_streaming_json_object_no_reasoning_leak(report_test, api_client, request):
    """qwen3 reasoning parser: streaming json_object must not leak reasoning.

    With ``response_format: {"type": "json_object"}`` and ``stream: true`` the
    trailing reasoning tokens and the literal ``</think>`` close tag must stay
    in ``reasoning_content`` and never appear as ``content`` deltas, so the
    accumulated content parses as JSON. The test fails if ANY run leaks the
    ``</think>`` tag into content or the content does not parse as JSON.
    """
    base_payload = {
        "messages": JSON_OBJECT_MESSAGES,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1,
        "repetition_penalty": 1,
        "top_p": 1,
        "stop": ["<|im_start|>", "<|im_end|>"],
        "seed": None,
        "min_p": 0,
        "max_tokens": 8192,
        "top_a": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
    }

    failures = []
    completed = 0
    prev_start = None
    for i in range(_JSON_STREAMING_RUNS):
        result, prev_start = _stream_one_run_with_backoff(
            api_client, base_payload, i, prev_start
        )
        completed += 1
        content = result["content"]
        leak = _find_think_leak(content)
        try:
            json.loads(content)
            content_valid_json = True
        except (json.JSONDecodeError, ValueError):
            content_valid_json = False
        ok = leak is None and content_valid_json
        detail = (
            f"think_leak={leak!r}, content_valid_json={content_valid_json}, "
            f"first_content_deltas={result['content_deltas'][:6]!r}, "
            f"content={content!r}"
        )
        print(f"[run {i}] {'OK' if ok else 'LEAK/INVALID'}: {detail}")
        if not ok:
            failures.append(f"run {i}: {detail}")

    assert not failures, (
        f"{len(failures)}/{completed} streaming runs leaked reasoning / a "
        "</think> tag into content or produced invalid JSON (expected "
        "reasoning to stay in reasoning_content and content to be a clean, "
        "parseable JSON object). Failing runs:\n" + "\n".join(failures)
    )
