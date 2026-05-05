# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""End-to-end assertions on what the API/Service layer hands to the runner.

Each test:
    1. starts with a cleared event buffer (handled by the `client` fixture
       in `conftest.py`)
    2. drives one or more HTTP requests
    3. fetches the resulting events via `GET /debug/runner-events`
    4. asserts inline -- the request and the assertion live in the same
       scope, no fixture files to maintain

Run with:
    pytest tests/recorder/test_runner_events.py -v

The `server_url` session fixture boots a single-worker mock server with
`TT_RUNNER_RECORDER_ENABLED=1` once for the whole module; teardown sends
SIGTERM. See `conftest.py` for fixture details.

Goals:
    - Refactor-safety net for API / controllers / services / sessions /
      disaggregation glue. Anything below the producer side of the task
      queue is out of scope.

Anti-goals:
    - Not a model-output correctness test. Token contents are
      represented only by `tokens_xx64` (xxhash) + `tokens_len`.
"""

from __future__ import annotations

from typing import Any

from conftest import RunnerEventClient


MODEL = "deepseek-ai/DeepSeek-R1-0528"


def _exactly_one_task(client: RunnerEventClient) -> dict[str, Any]:
    events = client.get_events()
    task_events = [e for e in events if e["kind"] == "task_submitted"]
    assert len(task_events) == 1, (
        f"expected exactly 1 task_submitted event, got: {events}"
    )
    return task_events[0]


# ---------------------------------------------------------------------------
# non-streaming
# ---------------------------------------------------------------------------


def test_non_streaming_submits_single_task(client: RunnerEventClient) -> None:
    client.chat(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }
    ).raise_for_status()

    event = _exactly_one_task(client)
    assert event["kind"] == "task_submitted"
    assert event["continuation"] is False
    assert event["disaggregated"] is False
    assert event["sampling"]["max_tokens"] == 8
    assert event["sampling"]["temperature"] == 0
    assert event["sampling"]["response_format"] == "TEXT"
    assert event["sampling"]["tools_count"] == 0
    assert event["sampling"]["tool_choice_type"] is None
    assert event["num_prompt_tokens"] > 0
    assert event["tokens_len"] == event["num_prompt_tokens"]


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


def test_streaming_submits_single_task(client: RunnerEventClient) -> None:
    chunks = client.stream_chat(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": True,
        }
    )
    assert chunks > 0

    event = _exactly_one_task(client)
    assert event["continuation"] is False
    assert event["sampling"]["max_tokens"] == 8


# ---------------------------------------------------------------------------
# multi-turn / prefix cache
# ---------------------------------------------------------------------------


def test_multi_turn_second_call_is_continuation(
    client: RunnerEventClient,
) -> None:
    first = client.chat(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "turn one"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        }
    )
    first.raise_for_status()
    assistant_text = first.json()["choices"][0]["message"].get("content", "")

    first_events = client.get_events()
    assert len(first_events) == 1
    first_seq = first_events[0]["seq"]
    assert first_events[0]["continuation"] is False

    client.chat(
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "turn one"},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": "turn two"},
            ],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        }
    ).raise_for_status()

    new_events = client.get_events(since_seq=first_seq)
    assert len(new_events) == 1
    assert new_events[0]["continuation"] is True, (
        "second turn should reuse the prefix-cached session"
    )
    assert new_events[0]["sampling"]["max_tokens"] == 4


# ---------------------------------------------------------------------------
# tool calling
# ---------------------------------------------------------------------------


def test_tool_choice_auto_propagates_to_runner(
    client: RunnerEventClient,
) -> None:
    client.chat(
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "what is the weather in NYC?"}
            ],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Look up the current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    ).raise_for_status()

    event = _exactly_one_task(client)
    assert event["sampling"]["tools_count"] == 1
    assert event["sampling"]["tool_choice_type"] == "auto"


# ---------------------------------------------------------------------------
# structured output
# ---------------------------------------------------------------------------


def test_response_format_json_schema_propagates(
    client: RunnerEventClient,
) -> None:
    client.chat(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "give me JSON"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        }
    ).raise_for_status()

    event = _exactly_one_task(client)
    assert event["sampling"]["response_format"] == "JSON_SCHEMA"
    assert event["sampling"]["json_schema_present"] is True


# ---------------------------------------------------------------------------
# determinism: same prompt -> same fingerprint
# ---------------------------------------------------------------------------


def test_identical_requests_produce_identical_token_fingerprints(
    client: RunnerEventClient,
) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "deterministic"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }

    client.chat(payload).raise_for_status()
    first = _exactly_one_task(client)

    client.clear_events()
    client.chat(payload).raise_for_status()
    second = _exactly_one_task(client)

    assert first["tokens_xx64"] == second["tokens_xx64"]
    assert first["tokens_len"] == second["tokens_len"]
