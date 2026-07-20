# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import json
import math
import time
from collections import Counter

import pytest
import requests


# --- Helper Functions ---
def tokenize(text):
    """Tokenize by whitespace (simple, portable)."""
    return text.lower().split()


def repetition_stats(text):
    """Compute repetition metrics."""
    tokens = tokenize(text)
    counts = Counter(tokens)

    return {
        "len": len(tokens),
        "unique": len(set(tokens)),
        "unique_ratio": len(set(tokens)) / len(tokens) if tokens else 0,
        "most_common": counts.most_common(3),
        "entropy": shannon_entropy(tokens),
    }


def shannon_entropy(tokens):
    total = len(tokens)
    if total == 0:
        return 0
    counts = Counter(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# --- Reusable Prompts ---
BASE_PROMPT = [{"role": "user", "content": "Tell me a short joke."}]
REPRO_PROMPT = [
    {"role": "user", "content": "What is the capital of France? Be concise."}
]
PENALTY_PROMPTS = {
    "repeat_trap": [{"role": "user", "content": "Write a very repetitive story."}],
    "natural_repetition": [
        {
            "role": "user",
            "content": "Write a paragraph about bananas using simple language.",
        }
    ],
    "semantic_repetition": [
        {
            "role": "user",
            "content": "Write a creative story about a dragon named Blaze.",
        }
    ],
}

# --- Tool-calling + reasoning repro (issue: Qwen3-32B streaming) ---
# Weather tool matching the issue repro; a thinking-enabled prompt that should
# reliably trigger a single tool call.
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

# --- Streaming reasoning-parser repro (issues 1 & 2: Qwen3-32B) ---
# The qwen3 reasoning parser mishandles the reasoning->content boundary in
# streaming mode, emitting the trailing reasoning tokens and the literal
# ``</think>`` close tag as *content* deltas instead of ``reasoning_content``.
# The accumulated content is therefore not valid JSON even though the request
# asked for a JSON response_format. Non-streaming is clean, which isolates the
# defect to the incremental path (the marker is observed split across deltas,
# e.g. ``'<'`` + ``'/think>{'``).
#
# The prompt below is copied verbatim from the issue's issue-1 repro; it is the
# demanding prompt that reliably triggers the streaming reasoning-boundary edge
# case (a simplified prompt did not leak, per the issue's controls).
JSON_OBJECT_MESSAGES = [
    {
        "role": "user",
        "content": (
            'Return ONLY a minified JSON object with EXACTLY these keys and '
            'constraints: {"location":string, "temperature":integer '
            '[-100..100], "conditions":"sunny|cloudy|rainy|snowy", '
            '"unit":"celsius", "readings":[{t:integer, ts:ISO-8601 UTC Z} x '
            '3]}. No code fences, no extra text, no newlines. Example shape '
            "only; fill realistic values for Boston."
        ),
    }
]

# issue-2 repro prompt + strict json_schema, copied verbatim from the issue.
JSON_SCHEMA_MESSAGES = [
    {"role": "user", "content": "What is the weather like in London?"}
]
WEATHER_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "enum": ["London", "New York"]},
                "temperature": {"type": "number"},
                "conditions": {"type": "string"},
            },
            "required": ["location", "temperature", "conditions"],
        },
    },
}

# The failure is observed 4/4 (issue 1) / repeatedly (issue 2), but keep a small
# repeat count so an occasional clean run does not mask the regression.
_JSON_STREAMING_RUNS = 4

# The console ingress rate-limits after a couple of expensive back-to-back
# streams: subsequent requests are rejected (observed as 404, or a mid-stream
# truncation surfacing as ChunkedEncodingError) and recover after a rolling
# window. To avoid tripping it we space the START of each run at least this many
# seconds after the previous run's start (gap-aware: the stream's own elapsed
# time counts toward the gap, so a run that already took 40s only sleeps the
# remaining ~80s). Tune here if the endpoint's limit differs.
_JSON_STREAMING_MIN_GAP_S = 120

# If a run is still rejected/truncated despite the gap, wait one more gap and
# retry that same run this many times before giving up. Only cleanly-completed
# streams are recorded and asserted against.
_JSON_STREAMING_MAX_RETRIES = 3


# --- Test Functions ---
@pytest.mark.parametrize("n_val", [2, 3])
def test_n(report_test, api_client, n_val, request):
    """Tests the 'n' parameter (number of choices)."""
    payload = {"messages": BASE_PROMPT, "n": n_val, "max_tokens": 32}
    response = api_client(payload)

    try:
        assert "choices" in response, "choices field is not in response"
        assert len(response["choices"]) == n_val, (
            f"length of choices field does not match n_val: {n_val}"
        )
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("max_val", [5, 10])
def test_max_tokens(report_test, api_client, max_val, request):
    """Tests the 'max_tokens' parameter."""
    payload = {"messages": BASE_PROMPT, "max_tokens": max_val}
    response = api_client(payload)

    completion_token_count = response["usage"].get("completion_tokens")
    if not completion_token_count:
        raise ValueError("Response did not contain completion_tokens field in usage.")

    assert completion_token_count <= max_val, (
        f"Generated {completion_token_count} tokens, which is greater than allowed amount of {max_val}."
    )


@pytest.mark.parametrize("stop_seq", [["Stop"], ["END", "DONE"]])
def test_stop(report_test, api_client, stop_seq, request):
    """Tests the 'stop' parameter."""
    prompt = [{"role": "user", "content": "Count to 5 and then say 'StopIt'."}]
    payload = {"messages": prompt, "stop": stop_seq, "max_tokens": 1024}
    response = api_client(payload)

    output_text = response["choices"][0]["message"]["content"]
    for seq in stop_seq:
        assert seq not in output_text, f"Sequence {seq} was in {output_text}"


def test_seed_reproducibility(report_test, api_client, request):
    """Tests the 'seed' parameter for reproducibility."""
    payload = {"messages": REPRO_PROMPT, "seed": 42, "temperature": 0.5}

    response1 = api_client(payload)
    response2 = api_client(payload)

    output1 = response1["choices"][0]["message"]["content"]
    output2 = response2["choices"][0]["message"]["content"]
    assert output1 and output1 == output2, (
        f"Seed did not produce reproducible results. Output 1: '{output1}', Output 2: '{output2}'"
    )


def test_coherence_verbatim_echo(report_test, api_client, request):
    """Coherence guard: the model must echo an exact sentence verbatim.

    The other tests here only check structural/determinism properties (n,
    max_tokens, stop, seed, logprobs, penalties). A backend regression that
    corrupts the forward pass (e.g. a broken on-device decode trace) still
    emits deterministic, well-formed-but-garbage tokens, so all of those pass
    while the model is actually producing gibberish. Requiring a verbatim echo
    catches that class of bug directly: a model that cannot reproduce a simple
    sentence is not generating coherent text.
    """
    sentence = "The quick brown fox jumps over the lazy dog."
    prompt = [
        {
            "role": "user",
            "content": (
                "Repeat the following sentence exactly, with no extra words, "
                f"no quotes, and no commentary: {sentence}"
            ),
        }
    ]
    payload = {"messages": prompt, "max_tokens": 32, "temperature": 0}
    response = api_client(payload)

    output_text = response["choices"][0]["message"]["content"]
    assert sentence in output_text, (
        "Coherence guard failed: model did not echo the sentence verbatim "
        "(likely gibberish from a corrupted forward pass). Expected to find "
        f"{sentence!r} in output, got: {output_text!r}"
    )


@pytest.mark.asyncio
async def test_non_uniform_seeding(report_test, api_client, request):
    """
    Tests that vLLM Chat endpoint respects per-request seeding
    under concurrent load.

    Pattern:
    - 32 concurrent requests.
    - Seeds: [1, 0, 2, 0, 3, 0 ...]
    - Expectation:
        - All seed=0 requests -> Identical content.
        - All seed!=0 requests -> Unique content.
    """

    total_requests = 32
    seeds = []

    # Generate pattern: 1, 0, 2, 0, 3, 0...
    for i in range(1, (total_requests // 2) + 1):
        seeds.append(i)
        seeds.append(0)

    async def get_chat_response(seed_val):
        try:
            # We use a temperature > 0 to prove that the seed is
            # actually forcing the determinism.
            payload = {
                "messages": [
                    {"role": "user", "content": "Generate a list of 10 random colors."}
                ],
                "max_tokens": 50,
                "temperature": 0.9,
                "seed": seed_val,
            }
            response = await asyncio.to_thread(api_client, payload)
            return {
                "seed": seed_val,
                "content": response["choices"][0]["message"]["content"].strip(),
                "id": response["id"],
            }
        except Exception as e:
            pytest.fail(f"Request failed for seed {seed_val}: {str(e)}")

    # Fire all requests concurrently
    # This floods the server, forcing dynamic batching if enabled on vLLM
    tasks = [get_chat_response(s) for s in seeds]
    results = await asyncio.gather(*tasks)

    # Separate results
    zero_seed_contents = [r["content"] for r in results if r["seed"] == 0]
    unique_seed_contents = [r["content"] for r in results if r["seed"] != 0]
    assert len(zero_seed_contents) == total_requests / 2
    assert len(unique_seed_contents) == total_requests / 2

    # --- ASSERTIONS ---

    # 1. Verify Determinism (Seed = 0)
    # Using a set to find unique values. Length should be exactly 1.
    unique_zero_outputs = set(zero_seed_contents)
    assert len(unique_zero_outputs) == 1, (
        f"Determinism Failed for seed=0.\n"
        f"Expected 1 unique output, found {len(unique_zero_outputs)}.\n"
        f"Outputs: {unique_zero_outputs}"
    )

    # 2. Verify Entropy (Seed != 0)
    # The set length should equal the list length (all unique).
    unique_varied_outputs = set(unique_seed_contents)
    assert len(unique_varied_outputs) == len(unique_seed_contents), (
        f"Entropy Failed for non-zero seeds.\n"
        f"Expected {len(unique_seed_contents)} unique outputs, found {len(unique_varied_outputs)}.\n"
        f"Collisions detected in: {unique_seed_contents}"
    )


def test_logprobs(report_test, api_client, request):
    """Tests the 'logprobs' parameter."""
    payload = {"messages": BASE_PROMPT, "logprobs": True, "max_tokens": 100}
    response = api_client(payload)

    try:
        assert "logprobs" in response["choices"][0]
        assert response["choices"][0]["logprobs"] is not None
        assert "content" in response["choices"][0]["logprobs"]
        assert len(response["choices"][0]["logprobs"]["content"]) > 0
    except AssertionError as e:
        msg = f"Logprobs data was missing or malformed: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize(
    "param_name, param_value",
    [
        ("temperature", 0.0),
        ("top_k", 1),
        ("top_p", 0.01),  # A very low top_p should also be deterministic
    ],
)
def test_determinism_parameters(
    report_test, api_client, param_name, param_value, request
):
    """Tests parameters that should force deterministic output."""
    payload = {"messages": REPRO_PROMPT, param_name: param_value}

    # If testing top_k or top_p, set temperature high to prove they are working
    if param_name != "temperature":
        payload["temperature"] = 1.0

    response1 = api_client(payload)
    response2 = api_client(payload)

    output1 = response1["choices"][0]["message"]["content"]
    output2 = response2["choices"][0]["message"]["content"]
    assert output1 and output1 == output2, (
        f"{param_name}={param_value} was not deterministic. Output 1: '{output1}', Output 2: '{output2}'"
    )


@pytest.mark.parametrize("prompt_name, messages", PENALTY_PROMPTS.items())
@pytest.mark.parametrize(
    "penalty_param, penalty_val",
    [
        ("presence_penalty", 1.2),
        ("frequency_penalty", 1.2),
        ("repetition_penalty", 1.5),  # vLLM implements this, OpenAI uses the other two
    ],
)
def test_penalties(
    report_test, api_client, prompt_name, messages, penalty_param, penalty_val, request
):
    """Tests repetition, presence, and frequency penalties."""

    # Baseline run (no penalty)
    payload_base = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
        "seed": 1234,
    }
    response_base = api_client(payload_base, timeout=None)

    # Test run (with penalty)
    payload_test = payload_base.copy()
    payload_test[penalty_param] = penalty_val
    response_test = api_client(payload_test, timeout=None)

    # Compute baseline and test statistics
    text_base = response_base["choices"][0]["message"]["content"]
    text_test = response_test["choices"][0]["message"]["content"]

    base_stats = repetition_stats(text_base)
    test_stats = repetition_stats(text_test)

    try:
        # 1. Diversity should not decrease (penalties push diversity up)
        assert test_stats["unique_ratio"] >= base_stats["unique_ratio"] * 0.90, (
            "Penalty unexpectedly reduced diversity."
        )

        # 2. Heavy repetition should decrease
        #    For repetition-heavy prompts, penalties reduce top-token dominance
        if prompt_name == "repeat_trap" and penalty_param != "presence_penalty":
            most_common_penalty = test_stats["most_common"][0][1]
            most_common_baseline = base_stats["most_common"][0][1]
            assert most_common_penalty <= most_common_baseline, (
                "Penalty didn't reduce repetition on repetition-trap prompt."
            )

        # 3. Length differences accepted but should not be identical
        assert test_stats["len"] != base_stats["len"], (
            "Penalty had no measurable effect on output length."
        )

        # For vLLM-specific repetition_penalty, check more aggressive behavior
        if penalty_param == "repetition_penalty":
            assert test_stats["unique_ratio"] > base_stats["unique_ratio"], (
                "vLLM repetition_penalty did not increase diversity enough."
            )
    except AssertionError as e:
        msg = f"Test failed: {str(e)}. Base: {text_base}, Test: {text_test}"
        raise AssertionError(msg)


def _stream_chat_completion(api_client, payload):
    """Send a streaming chat-completion and reconstruct the single choice.

    Returns a dict with the accumulated ``reasoning_content``, ``content``, the
    ordered list of raw ``content_deltas`` (each per-chunk ``delta.content``
    string, preserved so tests can inspect the reasoning->content boundary),
    aggregated ``tool_calls`` (indexed by their delta index), and the terminal
    ``finish_reason`` seen on the stream. Raises the underlying HTTPError so
    callers can distinguish an unsupported request (e.g. a model that does not
    accept ``chat_template_kwargs``) from a bad completion.
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


# Number of repetitions. The failure is intermittent (~40% in the issue), so a
# single run can pass by luck; repeating makes the regression deterministic.
_TOOL_THINKING_RUNS = 20


def test_streaming_tool_call_with_thinking(report_test, api_client, request):
    """Streaming + tools + enable_thinking must still emit the tool call.

    Reproduces the Qwen3-32B serving-side bug where the qwen3 reasoning parser
    and the hermes tool parser interact so that, after a ``<think>`` block, the
    tool call is dropped and the stream ends with ``finish_reason: "stop"`` and
    an empty ``tool_calls`` array. A correctly-behaving host emits
    ``finish_reason: "tool_calls"`` with a populated ``tool_calls`` array every
    time (the issue's Nebius reference passes 20/20).

    The test runs the exact repro payload multiple times and fails if ANY run
    drops the tool call, so it catches the intermittent failure today and will
    pass once the parser interaction is fixed.
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

    # If the server/model does not support thinking via chat_template_kwargs,
    # this scenario is not applicable — skip rather than fail so the shared
    # suite stays green for non-reasoning models. Auth failures (401/403) are
    # a configuration problem, not "unsupported", so let them surface.
    try:
        first = _stream_chat_completion(api_client, base_payload)
    except requests.exceptions.HTTPError as e:
        # The api_client fixture re-raises a new HTTPError without .response, so
        # detect auth failures from the message (requests uses "<code> Client
        # Error"). Auth is a config problem — surface it instead of skipping.
        msg = str(e)
        if "401" in msg or "403" in msg:
            raise
        pytest.skip(
            "Server rejected the streaming tools + enable_thinking payload; "
            f"model likely does not support chat_template_kwargs thinking: {e}"
        )

    failures = []
    results = [first]
    for _ in range(_TOOL_THINKING_RUNS - 1):
        results.append(_stream_chat_completion(api_client, base_payload))

    for i, result in enumerate(results):
        has_tool_call = any(slot["name"] for slot in result["tool_calls"].values())
        if result["finish_reason"] != "tool_calls" or not has_tool_call:
            failures.append(
                f"run {i}: finish_reason={result['finish_reason']!r}, "
                f"tool_calls={result['tool_calls']!r}, "
                f"reasoning_len={len(result['reasoning_content'])}, "
                f"content={result['content']!r}"
            )

    assert not failures, (
        f"{len(failures)}/{len(results)} streaming runs dropped the tool call "
        "after the reasoning block (expected finish_reason='tool_calls' with a "
        "populated tool_calls array on every run). Failing runs:\n"
        + "\n".join(failures)
    )

    # Sanity-check the emitted tool call targets the provided tool.
    first_call = next(iter(first["tool_calls"].values()))
    assert first_call["name"] == WEATHER_TOOL["function"]["name"], (
        f"Expected tool call '{WEATHER_TOOL['function']['name']}', "
        f"got '{first_call['name']}'."
    )


def _find_think_leak(content):
    """Return the ``</think>`` variant that leaked into ``content``, if any.

    The reasoning parser splits the close tag across deltas (observed as
    ``'<' + '/think>{'`` and ``'\\n</think' + '>{'``), so the check is run
    against the *joined* content rather than per-delta. Any of the tag or its
    partial forms landing in content is a leak.
    """
    for marker in ("</think>", "</think", "<think>", "<think"):
        if marker in content:
            return marker
    return None


def _is_rate_limit_error(exc):
    """Return True when an exception looks like a transient ingress rate-limit.

    The console ingress rejects excess streaming requests as ``404`` (observed)
    or ``429``, or cuts the stream mid-flight so ``requests`` raises a
    ``ChunkedEncodingError`` ("Response ended prematurely"). All are transient
    and recover after the rolling window, so they should be retried rather than
    counted as a parser failure. Auth errors (401/403) are NOT rate-limits.
    """
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        return True
    msg = str(exc)
    return "404" in msg or "429" in msg or "Response ended prematurely" in msg


def _evaluate_json_run(result):
    """Return ``(ok, detail)`` for a single completed streaming run.

    ``ok`` is True when reasoning stayed out of content (no ``</think>``) and
    the accumulated content parses as JSON. ``detail`` is a human-readable
    one-liner used for both the live ``-s`` log and the failure report.
    """
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
    return ok, detail


def _stream_one_run_with_backoff(api_client, base_payload, run_idx, prev_start):
    """Run a single streaming attempt, spacing and retrying around rate-limits.

    Waits until at least ``_JSON_STREAMING_MIN_GAP_S`` have elapsed since
    ``prev_start`` (gap-aware: the previous run's own stream time counts toward
    the gap) before starting, then streams once. On a transient rate-limit /
    truncation it waits a full gap and retries, up to
    ``_JSON_STREAMING_MAX_RETRIES`` times. Auth errors (401/403) and
    unsupported-payload errors propagate to the caller.

    Returns ``(result, start_ts)`` where ``start_ts`` is when the successful
    attempt began (so the next run can be spaced relative to it).
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
        except (requests.exceptions.HTTPError, requests.exceptions.ChunkedEncodingError) as e:
            if _is_rate_limit_error(e):
                if attempt >= _JSON_STREAMING_MAX_RETRIES:
                    pytest.skip(
                        f"[run {run_idx}] rate-limited/truncated after "
                        f"{attempt} retries ({e}); the ingress limit did not "
                        "clear within the retry budget — infra flake, not a "
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
            # Auth (401/403) is a config problem — surface it. Any other HTTP
            # error means the model likely does not support this payload.
            msg = str(e)
            if "401" in msg or "403" in msg:
                raise
            pytest.skip(
                "Server rejected the streaming response_format payload; model "
                f"likely does not support this response_format/reasoning path: {e}"
            )


def _run_streaming_json_test(api_client, base_payload):
    """Run the streaming JSON repro N times and assert no reasoning leak.

    Shared by the ``json_object`` (issue 1) and ``json_schema`` (issue 2)
    tests: both must keep reasoning in ``reasoning_content`` and emit only the
    JSON object as ``content`` (no ``</think>`` tag, and the accumulated
    content must parse cleanly).

    Each run is evaluated and logged the moment it completes (visible with
    ``-s``) rather than buffering all runs first, and runs are spaced out with
    a gap-aware buffer + retry so the transient ingress rate-limit doesn't
    masquerade as a parser failure. If the server/model does not support the
    reasoning path the runner skips (non-auth HTTP error); auth errors surface.
    """
    failures = []
    completed = 0
    prev_start = None

    for i in range(_JSON_STREAMING_RUNS):
        result, prev_start = _stream_one_run_with_backoff(
            api_client, base_payload, i, prev_start
        )
        completed += 1
        ok, detail = _evaluate_json_run(result)
        # Report each run immediately as it is received.
        print(f"[run {i}] {'OK' if ok else 'LEAK/INVALID'}: {detail}")
        if not ok:
            failures.append(f"run {i}: {detail}")

    assert not failures, (
        f"{len(failures)}/{completed} streaming runs leaked reasoning / a "
        "</think> tag into content or produced invalid JSON (expected "
        "reasoning to stay in reasoning_content and content to be a clean, "
        "parseable JSON object). Failing runs:\n" + "\n".join(failures)
    )


def test_streaming_json_object_no_reasoning_leak(report_test, api_client, request):
    """Issue 1: streaming response_format json_object must not leak reasoning.

    Reproduces the Qwen3-32B streaming reasoning-parser bug: with
    ``response_format: {"type": "json_object"}`` and ``stream: true`` the
    trailing reasoning tokens and the literal ``</think>`` close tag are emitted
    as *content* deltas instead of ``reasoning_content``, so the accumulated
    content is not valid JSON (observed 4/4 in the issue). Non-streaming is
    clean, isolating the defect to ``extract_reasoning_content_streaming``.

    Payload matches the issue's issue-1 repro verbatim; the test fails if ANY
    run leaks the ``</think>`` tag into content or the content does not parse as
    JSON, and will pass once the streaming reasoning boundary is fixed.
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
    _run_streaming_json_test(api_client, base_payload)


def test_streaming_json_schema_no_reasoning_leak(report_test, api_client, request):
    """Issue 2: streaming response_format json_schema must not leak reasoning.

    Same streaming reasoning-parser defect as issue 1, but with a strict
    ``response_format: {"type": "json_schema", ...}``: the ``</think>`` close
    tag leaks into content and the accumulated content is not valid JSON, while
    non-streaming stays clean. Payload matches the issue's issue-2 repro
    verbatim.

    The test fails if ANY run leaks the ``</think>`` tag into content or the
    content does not parse as JSON, and will pass once the streaming reasoning
    boundary is fixed.
    """
    base_payload = {
        "messages": JSON_SCHEMA_MESSAGES,
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
        "response_format": WEATHER_JSON_SCHEMA,
    }
    _run_streaming_json_test(api_client, base_payload)
