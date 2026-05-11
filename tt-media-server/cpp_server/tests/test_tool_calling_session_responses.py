# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Protocol smoke test: replays a Codex-style tool-using session on /v1/responses.

Sibling of test_tool_calling_session.py. That file targets the OpenAI
Chat Completions wire format (which Claude Code does *not* speak —
Claude Code speaks Anthropic Messages). This file targets the OpenAI
Responses API at /v1/responses, which is what Codex CLI actually drives.

What this exercises
-------------------
- POST /v1/responses with the Responses-style request shape:
  `instructions` (system prompt), `input` (array of items), top-level
  `tools` array, `stream=true`.
- Tool definitions in real Codex/Responses format — flattened, with
  `type`, `name`, `description`, `parameters` at the same level (NOT
  nested under a `function` key the way Chat Completions does it).
- Multi-turn replay: assistant tool_call + tool_result items are
  appended to the running `input` array, mirroring how Codex
  accumulates the conversation locally and resends it each turn.
- The typed SSE event stream the server emits for /v1/responses
  (`response.created`, `response.output_text.delta`,
  `response.completed`, ...).

Wire-format note
----------------
At the time of writing, this server's `ResponsesRequest::toMessages()`
ingests each input array item via `ChatMessage::fromJson`, so it expects
chat-completion-style message objects (`role` + `content` + optional
`tool_calls` / `tool_call_id`) as input items — not the strict OpenAI
Responses item shape (`type: "message"` with `content: [{"type":
"input_text", ...}]`, `type: "function_call"`, `type:
"function_call_output"`). We use the chat-message shape that the server
actually accepts today; if the server ever grows native Responses-item
ingestion, the SESSION_TURNS data here is where to update.

Usage
-----
Assumes a running C++ server reachable at SERVER_BASE_URL (default
http://127.0.0.1:8000). Start one separately, e.g.:

    LLM_DEVICE_BACKEND=mock ./build/tt_media_server_cpp -p 8000

Then:

    pytest cpp_server/tests/test_tool_calling_session_responses.py -sv
    python cpp_server/tests/test_tool_calling_session_responses.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field

import requests

DEFAULT_API_KEY = "your-secret-key"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528"
REQUEST_TIMEOUT_SEC = 60
MAX_OUTPUT_TOKENS = 128


# ── tools in OpenAI Responses (Codex) format ─────────────────────────────
#
# Note the *flattened* shape: `type`, `name`, `description`, `parameters`
# all at the same level. Chat Completions nests these under a `function`
# key; the Responses API does not. This is what Codex CLI actually sends.

TOOLS = [
    {
        "type": "function",
        "name": "Bash",
        "description": (
            "Execute a shell command in the user's working directory. "
            "Returns combined stdout/stderr."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command line to execute.",
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable description of what the command does.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "Read",
        "description": "Read the contents of a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path of the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "type": "function",
        "name": "Edit",
        "description": "Replace an exact string in a file with another string.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path of the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Text to replace it with.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "type": "function",
        "name": "Grep",
        "description": "Search for a regex pattern across files under a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search under.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob to filter files (e.g. '*.py').",
                },
            },
            "required": ["pattern"],
        },
    },
]


# ── the scripted session: same five turns as the chat-completions test ───
#
# `instructions` carries the system prompt (the server maps it onto a
# system ChatMessage in toMessages()). Each Turn's `appended_items` gets
# concatenated to the running `input` array before that turn's request
# fires. Items are chat-message-shaped (see the wire-format note above).

INSTRUCTIONS = (
    "You are a coding assistant with access to Bash, Read, Edit, and Grep "
    "tools. Use them to investigate and modify the user's codebase. Keep "
    "responses concise."
)


@dataclass
class Turn:
    description: str
    appended_items: list[dict] = field(default_factory=list)


SESSION_TURNS: list[Turn] = [
    Turn(
        description="user asks the assistant to find auth helper",
        appended_items=[
            {
                "role": "user",
                "content": (
                    "There's a function in this repo that builds the auth "
                    "header for outbound requests. Find it, bump the default "
                    "timeout from 5 to 10 seconds, and confirm the change."
                ),
            },
        ],
    ),
    Turn(
        description="assistant called Grep; we feed it the search results",
        appended_items=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_grep_1",
                        "type": "function",
                        "function": {
                            "name": "Grep",
                            "arguments": json.dumps(
                                {
                                    "pattern": "def build_auth_header",
                                    "path": "src",
                                    "glob": "*.py",
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_grep_1",
                "content": (
                    "src/http/auth.py:42:def build_auth_header(token, "
                    "timeout=5):\n"
                ),
            },
        ],
    ),
    Turn(
        description="assistant called Read; we feed it the file contents",
        appended_items=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": json.dumps(
                                {
                                    "file_path": "/repo/src/http/auth.py",
                                    "offset": 35,
                                    "limit": 20,
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": (
                    "    35: import time\n"
                    "    36: from typing import Optional\n"
                    "    37:\n"
                    "    38: from .session import Session\n"
                    "    39:\n"
                    "    40: # NOTE: keep in sync with retry budget in settings.py\n"
                    "    41:\n"
                    "    42: def build_auth_header(token, timeout=5):\n"
                    '    43:     """Build the Authorization header dict."""\n'
                    '    44:     return {"Authorization": f"Bearer {token}",\n'
                    '    45:             "X-Timeout": str(timeout)}\n'
                ),
            },
        ],
    ),
    Turn(
        description="assistant called Edit; we feed it the edit confirmation",
        appended_items=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_edit_1",
                        "type": "function",
                        "function": {
                            "name": "Edit",
                            "arguments": json.dumps(
                                {
                                    "file_path": "/repo/src/http/auth.py",
                                    "old_string": "def build_auth_header(token, timeout=5):",
                                    "new_string": "def build_auth_header(token, timeout=10):",
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_edit_1",
                "content": (
                    "Edited /repo/src/http/auth.py (1 replacement).\n"
                    "  - def build_auth_header(token, timeout=5):\n"
                    "  + def build_auth_header(token, timeout=10):\n"
                ),
            },
        ],
    ),
    Turn(
        description="assistant called Bash to verify; we feed grep output back",
        appended_items=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bash_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps(
                                {
                                    "command": "grep -n 'def build_auth_header' src/http/auth.py",
                                    "description": "Confirm the new signature is in place.",
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_bash_1",
                "content": "42:def build_auth_header(token, timeout=10):\n",
            },
        ],
    ),
]


# ── HTTP plumbing ────────────────────────────────────────────────────────


def _base_url() -> str:
    return os.environ.get("SERVER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL)


def _auth_headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _build_payload(input_items: list[dict]) -> dict:
    return {
        "model": _model(),
        "instructions": INSTRUCTIONS,
        "input": input_items,
        "tools": TOOLS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }


def _send_turn(input_items: list[dict]) -> requests.Response:
    return requests.post(
        f"{_base_url()}/v1/responses",
        json=_build_payload(input_items),
        headers=_auth_headers(),
        stream=True,
        timeout=REQUEST_TIMEOUT_SEC,
    )


# ── SSE parsing for typed `event: <name>\ndata: <json>` streams ──────────


@dataclass
class SseEvent:
    name: str
    data: dict


@dataclass
class ParsedStream:
    events: list[SseEvent] = field(default_factory=list)
    unexpected_lines: list[str] = field(default_factory=list)


def _parse_typed_sse(response: requests.Response) -> ParsedStream:
    """Parse an `event:` + `data:` SSE stream into a list of SseEvent.

    The Responses formatter writes one event per record:

        event: response.output_text.delta\n
        data: {...json...}\n
        \n

    Records are separated by blank lines. We do NOT expect a `[DONE]`
    sentinel — terminal status is signaled by `response.completed` or
    `response.incomplete`.
    """
    parsed = ParsedStream()
    pending_event: str | None = None
    pending_data: str | None = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        if raw_line == "":
            if pending_event is not None and pending_data is not None:
                parsed.events.append(
                    SseEvent(name=pending_event, data=json.loads(pending_data))
                )
            pending_event = None
            pending_data = None
            continue
        if raw_line.startswith("event: "):
            pending_event = raw_line[len("event: ") :].strip()
        elif raw_line.startswith("data: "):
            pending_data = raw_line[len("data: ") :]
        else:
            parsed.unexpected_lines.append(raw_line)
    # Flush trailing record if the server didn't terminate with a blank line.
    if pending_event is not None and pending_data is not None:
        parsed.events.append(
            SseEvent(name=pending_event, data=json.loads(pending_data))
        )
    return parsed


# ── per-turn protocol invariants ─────────────────────────────────────────


TERMINAL_EVENTS = {"response.completed", "response.incomplete"}

REQUIRED_EVENT_PREFIXES = (
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
)


def _assert_turn_invariants(turn_index: int, stream: ParsedStream) -> None:
    where = f"turn[{turn_index}]"
    assert stream.events, (
        f"{where}: no SSE events parsed (unexpected_lines={stream.unexpected_lines!r})"
    )
    assert not stream.unexpected_lines, (
        f"{where}: unexpected SSE lines (not `event:` / `data:` / blank): "
        f"{stream.unexpected_lines!r}"
    )

    # Each event's `data` payload must echo its event `type` and carry a
    # monotonically increasing `sequence_number`.
    last_seq = -1
    for i, ev in enumerate(stream.events):
        assert ev.data.get("type") == ev.name, (
            f"{where} event[{i}]: data.type ({ev.data.get('type')!r}) "
            f"!= header event ({ev.name!r})"
        )
        seq = ev.data.get("sequence_number")
        assert isinstance(seq, int), (
            f"{where} event[{i}] ({ev.name}): missing/non-int sequence_number"
        )
        assert seq > last_seq, (
            f"{where} event[{i}] ({ev.name}): sequence_number {seq} not "
            f"strictly greater than previous {last_seq}"
        )
        last_seq = seq

    names = [ev.name for ev in stream.events]

    # First event must be response.created and carry the in-progress
    # response object.
    assert names[0] == "response.created", (
        f"{where}: first event should be response.created, got {names[0]!r}"
    )
    created_resp = stream.events[0].data.get("response", {})
    assert created_resp.get("object") == "response", (
        f"{where}: response.created.response.object should be 'response', got "
        f"{created_resp.get('object')!r}"
    )
    assert created_resp.get("status") == "in_progress", (
        f"{where}: response.created.response.status should be 'in_progress', got "
        f"{created_resp.get('status')!r}"
    )
    assert created_resp.get("id"), f"{where}: response.created.response.id missing"

    # Every expected lifecycle event must appear at least once.
    missing = [name for name in REQUIRED_EVENT_PREFIXES if name not in names]
    assert not missing, f"{where}: missing required SSE events {missing!r}"

    # Last event must be a terminal one.
    assert names[-1] in TERMINAL_EVENTS, (
        f"{where}: terminal event should be one of {TERMINAL_EVENTS}, got "
        f"{names[-1]!r}"
    )
    terminal = stream.events[-1]
    terminal_resp = terminal.data.get("response", {})
    expected_status = "completed" if terminal.name == "response.completed" else "incomplete"
    assert terminal_resp.get("status") == expected_status, (
        f"{where}: terminal response.status should be {expected_status!r}, got "
        f"{terminal_resp.get('status')!r}"
    )

    # No `[DONE]` terminator (the Responses formatter doesn't emit one).
    assert all(
        "[DONE]" not in (line or "")
        for ev in stream.events
        for line in [ev.name, json.dumps(ev.data)]
    ), f"{where}: Responses stream unexpectedly contained `[DONE]`"

    # The accumulated text from output_text.delta events should equal the
    # final `output_text.done.text` and the message item's content text.
    deltas = [
        ev.data.get("delta", "")
        for ev in stream.events
        if ev.name == "response.output_text.delta"
    ]
    accumulated = "".join(deltas)
    done_events = [
        ev for ev in stream.events if ev.name == "response.output_text.done"
    ]
    assert done_events, f"{where}: missing response.output_text.done"
    done_text = done_events[-1].data.get("text", "")
    assert done_text == accumulated, (
        f"{where}: output_text.done.text ({done_text!r}) does not match "
        f"accumulated deltas ({accumulated!r})"
    )

    # Usage on the terminal event should be reasonable.
    usage = terminal_resp.get("usage")
    if usage is not None:  # usage may be omitted on `incomplete` in some paths
        assert usage.get("input_tokens", 0) > 0, (
            f"{where}: terminal usage missing input_tokens: {usage!r}"
        )
        assert usage.get("output_tokens", 0) >= 0, (
            f"{where}: terminal usage missing output_tokens: {usage!r}"
        )
        total = usage.get("total_tokens", 0)
        assert total == usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        ), f"{where}: total_tokens inconsistent: {usage!r}"


# ── the actual test ──────────────────────────────────────────────────────


def test_codex_style_tool_calling_session_responses():
    """Replay the canned tool-using session as N /v1/responses requests.

    Each turn appends its prior assistant tool_call + tool result to the
    running `input` array and resends — mirroring Codex's "resend the
    whole history every turn" mode (we don't exercise
    `previous_response_id` here; that's a separate axis).
    """
    input_items: list[dict] = []

    for i, turn in enumerate(SESSION_TURNS):
        input_items.extend(copy.deepcopy(turn.appended_items))

        response = _send_turn(input_items)
        assert response.status_code == 200, (
            f"turn[{i}] ({turn.description}): HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
        assert "text/event-stream" in response.headers.get("Content-Type", ""), (
            f"turn[{i}]: expected SSE, got Content-Type="
            f"{response.headers.get('Content-Type')!r}"
        )

        stream = _parse_typed_sse(response)
        _assert_turn_invariants(i, stream)


def main() -> int:
    print(
        f"Replaying Codex-style /v1/responses session against {_base_url()} "
        f"(model={_model()})"
    )
    try:
        test_codex_style_tool_calling_session_responses()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1
    print(f"  PASS ({len(SESSION_TURNS)} turns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
