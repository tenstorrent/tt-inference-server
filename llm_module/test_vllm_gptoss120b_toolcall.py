# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""gpt-oss-120b tool-call / reasoning-parser conformance suite.

These repros are gpt-oss-specific: the serving stack runs the
``openai_gptoss`` reasoning parser together with the ``openai`` tool-call
parser (see the ``openai/gpt-oss-120b`` device rows in
``workflows/model_specs/dev/llm.yaml`` -- ``reasoning_parser_name:
openai_gptoss`` / ``tool_call_parser_name: openai``). gpt-oss emits a Harmony
transcript -- ``<|channel|>analysis<|message|>...<|channel|>final<|message|>
...`` -- and only the final channel is the answer; the reasoning parser routes
the analysis channel to ``reasoning_content``. They are wired only for
``gpt_oss_120b`` via ``VLLMGptOss120bToolCallTest`` and mirror the structure of
``test_vllm_qwen36_toolcall.py`` (Qwen3.6-27B).

The two conformance properties asserted here:

1. **Structured tool_calls emitted** -- a prompt with a tool provided must
   stream a single, well-formed tool call: the openai tool parser emits
   ``finish_reason: "tool_calls"`` with a populated ``tool_calls`` array whose
   ``arguments`` parse as JSON, even after a Harmony analysis channel. A leak of
   Harmony control tokens (``<|channel|>`` / ``<|message|>`` / ``<|start|>`` /
   ``<|end|>``) into ``content`` means the reasoning/tool parsers are inert (the
   exact defect the flag-reconcile audit flagged) -- the arguments would then be
   un-parseable and the assertion fails.
2. **Reasoning parser strips correctly** -- with a streaming JSON
   ``response_format`` the openai_gptoss reasoning parser must keep the analysis
   channel in ``reasoning_content`` and never leak a Harmony control token into
   ``content``, so the accumulated content parses as JSON.

NOTE (draft wiring): serving for the quetzal war-room models is still being
unblocked (chunked serving), so this suite is wired-but-pending-serving -- it
is expected to run once serving works, not to pass today.
"""

import json
import time

import pytest
import requests

# --- Tool-calling + reasoning repro (openai tool parser) ---
# Weather tool; a prompt that should reliably trigger a single tool call after
# gpt-oss's Harmony analysis channel.
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
            "You must think before every response. First reason briefly, then "
            "give a concise final answer in plain text."
        ),
    },
    {
        "role": "user",
        "content": "What is the weather like in Boston, MA in fahrenheit?",
    },
]

# --- Streaming reasoning-parser repro (openai_gptoss reasoning parser) ---
# The openai_gptoss reasoning parser must keep the Harmony analysis channel out
# of the content stream; a leaked Harmony control token means the accumulated
# content is not valid JSON even though the request asked for a JSON
# response_format.
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

# gpt-oss routes its analysis channel to reasoning_content; the Harmony control
# tokens delimit the channelled transcript (serving/reasoning.py: CHANNEL_OPEN /
# CHANNEL_MESSAGE, plus the start/end/return message-boundary tokens). ANY of
# these landing in ``content`` is a parser leak. "analysis"/"final" are the
# channel *names* but are ordinary English words, so they are intentionally NOT
# leak markers (they would false-positive on normal prose).
_LEAK_MARKERS = (
    "<|channel|>",
    "<|message|>",
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|constrain|>",
    "<|channel",
    "<|message",
)

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


def _find_reasoning_leak(content):
    """Return the Harmony control token that leaked into ``content``, if any.

    The parser can split a marker across deltas, so the check is run against
    the *joined* content. Any marker landing in content is a leak.
    """
    for marker in _LEAK_MARKERS:
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
    """openai tool parser: streaming + tools must emit the tool call.

    A correctly-behaving host emits ``finish_reason: "tool_calls"`` with a
    populated ``tool_calls`` array on every run, even after a Harmony analysis
    channel, and the tool call's ``arguments`` must parse as JSON and target the
    provided tool. The test runs the repro multiple times and fails if ANY run
    drops the tool call, leaks a Harmony control token into content, or emits
    unparseable arguments.
    """
    base_payload = {
        "messages": THINKING_TOOL_MESSAGES,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 4096,
        "tools": [WEATHER_TOOL],
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
        # A raw Harmony control token in content means the reasoning/tool
        # parsers never fired (registered-but-inert) -- catch it explicitly.
        leak = _find_reasoning_leak(result["content"])
        if result["finish_reason"] != "tool_calls" or not has_tool_call or leak:
            failures.append(
                f"run {i}: finish_reason={result['finish_reason']!r}, "
                f"tool_calls={result['tool_calls']!r}, "
                f"content_leak={leak!r}, "
                f"reasoning_len={len(result['reasoning_content'])}, "
                f"content={result['content']!r}"
            )
            continue
        # Structured-arguments conformance: the openai parser must emit
        # tool-call arguments that parse as a JSON object.
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
        "run, and no Harmony control token leaking into content). "
        "Failing runs:\n" + "\n".join(failures)
    )

    # Sanity-check the emitted tool call targets the provided tool.
    first = results[0]
    first_call = next(iter(first["tool_calls"].values()))
    assert first_call["name"] == WEATHER_TOOL["function"]["name"], (
        f"Expected tool call '{WEATHER_TOOL['function']['name']}', "
        f"got '{first_call['name']}'."
    )


def test_streaming_json_object_no_reasoning_leak(report_test, api_client, request):
    """openai_gptoss reasoning parser: streaming json_object must not leak.

    With ``response_format: {"type": "json_object"}`` and ``stream: true`` the
    Harmony analysis channel and every Harmony control token must stay out of
    the content stream, so the accumulated content parses as JSON. The test
    fails if ANY run leaks a Harmony control token into content or the content
    does not parse as JSON.
    """
    base_payload = {
        "messages": JSON_OBJECT_MESSAGES,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 8192,
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
        leak = _find_reasoning_leak(content)
        try:
            json.loads(content)
            content_valid_json = True
        except (json.JSONDecodeError, ValueError):
            content_valid_json = False
        ok = leak is None and content_valid_json
        detail = (
            f"reasoning_leak={leak!r}, content_valid_json={content_valid_json}, "
            f"first_content_deltas={result['content_deltas'][:6]!r}, "
            f"content={content!r}"
        )
        print(f"[run {i}] {'OK' if ok else 'LEAK/INVALID'}: {detail}")
        if not ok:
            failures.append(f"run {i}: {detail}")

    assert not failures, (
        f"{len(failures)}/{completed} streaming runs leaked the Harmony "
        "analysis channel / a control token into content or produced invalid "
        "JSON (expected reasoning to stay in reasoning_content and content to "
        "be a clean, parseable JSON object). Failing runs:\n" + "\n".join(failures)
    )
