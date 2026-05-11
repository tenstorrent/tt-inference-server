# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Protocol smoke test: replays a single Claude-Code-style tool-using session.

What this is
------------
A multi-turn chat session — modeled after how Claude Code drives an LLM
backend in one user task — replayed against the C++ server as a series of
streaming /v1/chat/completions requests, one per turn. The session
exercises:

- A non-trivial `tools` array (Bash, Read, Edit, Grep) with realistic JSON
  schemas, sent on every request.
- Assistant turns that include `tool_calls` (so the server must accept
  prior tool_calls in history without choking).
- `role="tool"` messages with `tool_call_id` carrying the simulated tool
  output back in.
- Streaming responses (SSE) with `stream_options.include_usage=true`.

What this is NOT
----------------
Not a quality/golden test. The assistant text and any tool_calls the
server emits are model-dependent (mock backend will produce random-ish
output, llama backend will produce something coherent). We assert
*protocol* invariants only — that every turn comes back as a well-formed
chat.completion.chunk SSE stream — so this works against any backend.

Usage
-----
Assumes a running C++ server reachable at SERVER_BASE_URL (default
http://127.0.0.1:8000). Start one separately, e.g.:

    LLM_DEVICE_BACKEND=mock ./build/tt_media_server_cpp -p 8000

Then:

    pytest cpp_server/tests/test_tool_calling_session.py -sv
    python cpp_server/tests/test_tool_calling_session.py
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
MAX_TOKENS = 128


# ── tools the simulated Claude Code session has available ────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
]


# ── the scripted session: one user task, five turns ──────────────────────
#
# Story: user asks the assistant to locate an authentication helper, look
# at it, tweak a constant, and verify the change. Each TURN below is what
# we *append* to the running messages array before that turn's request.
#
# Turn 1 — just the user prompt.
# Turn 2+ — the previous turn's "assistant tool_call" (hardcoded — pretends
#           the model called the tool) plus the "tool" message carrying
#           the simulated tool output.
#
# By hardcoding the assistant tool_calls in history instead of using
# whatever the server actually returned, the test stays decoupled from
# model behavior and works against any backend.

SYSTEM_PROMPT = (
    "You are a coding assistant with access to Bash, Read, Edit, and Grep "
    "tools. Use them to investigate and modify the user's codebase. Keep "
    "responses concise."
)


@dataclass
class Turn:
    description: str
    appended_messages: list[dict] = field(default_factory=list)


SESSION_TURNS: list[Turn] = [
    Turn(
        description="user asks the assistant to find auth helper",
        appended_messages=[
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
        appended_messages=[
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
        appended_messages=[
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
                    '    42: def build_auth_header(token, timeout=5):\n'
                    '    43:     """Build the Authorization header dict."""\n'
                    '    44:     return {"Authorization": f"Bearer {token}",\n'
                    '    45:             "X-Timeout": str(timeout)}\n'
                ),
            },
        ],
    ),
    Turn(
        description="assistant called Edit; we feed it the edit confirmation",
        appended_messages=[
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
        appended_messages=[
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
                "content": (
                    "42:def build_auth_header(token, timeout=10):\n"
                ),
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


def _build_payload(messages: list[dict]) -> dict:
    return {
        "model": _model(),
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _send_turn(messages: list[dict]) -> requests.Response:
    return requests.post(
        f"{_base_url()}/v1/chat/completions",
        json=_build_payload(messages),
        headers=_auth_headers(),
        stream=True,
        timeout=REQUEST_TIMEOUT_SEC,
    )


# ── SSE parsing (same shape as test_golden_stream.py) ────────────────────


@dataclass
class ParsedStream:
    chunks: list[dict] = field(default_factory=list)
    ended_with_done: bool = False
    raw_events: list[str] = field(default_factory=list)


def _parse_sse(response: requests.Response) -> ParsedStream:
    parsed = ParsedStream()
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("data: "):
            parsed.raw_events.append(line)
            continue
        payload = line[len("data: ") :]
        parsed.raw_events.append(payload)
        if payload == "[DONE]":
            parsed.ended_with_done = True
            continue
        parsed.chunks.append(json.loads(payload))
    return parsed


# ── per-turn protocol invariants ─────────────────────────────────────────


def _assert_turn_invariants(turn_index: int, stream: ParsedStream) -> None:
    where = f"turn[{turn_index}]"
    assert stream.chunks, f"{where}: no chunks parsed (raw={stream.raw_events!r})"
    assert stream.ended_with_done, f"{where}: stream did not end with `data: [DONE]`"

    for i, chunk in enumerate(stream.chunks):
        assert chunk.get("object") == "chat.completion.chunk", (
            f"{where} chunk[{i}].object should be chat.completion.chunk, got {chunk!r}"
        )
        assert chunk.get("id"), f"{where} chunk[{i}] missing id"
        assert chunk.get("model"), f"{where} chunk[{i}] missing model"

    role = next(
        (
            ch["choices"][0]["delta"].get("role")
            for ch in stream.chunks
            if ch.get("choices") and ch["choices"][0].get("delta", {}).get("role")
        ),
        None,
    )
    assert role == "assistant", f"{where}: first role should be assistant, got {role!r}"

    finish_reasons = [
        ch["choices"][0].get("finish_reason")
        for ch in stream.chunks
        if ch.get("choices") and ch["choices"][0].get("finish_reason") is not None
    ]
    assert len(finish_reasons) == 1, (
        f"{where}: expected exactly one finish_reason, got {finish_reasons!r}"
    )
    assert finish_reasons[0] in {"stop", "length", "tool_calls"}, (
        f"{where}: unexpected finish_reason={finish_reasons[0]!r}"
    )

    usage_chunks = [ch for ch in stream.chunks if ch.get("usage")]
    assert usage_chunks, f"{where}: include_usage=true set but no usage chunk arrived"
    usage = usage_chunks[-1]["usage"]
    assert usage.get("prompt_tokens", 0) > 0, f"{where}: prompt_tokens missing: {usage!r}"
    assert usage.get("completion_tokens", 0) >= 0, (
        f"{where}: completion_tokens missing: {usage!r}"
    )

    # If any tool_call deltas arrived, they must at least carry an index +
    # be well-shaped (the server may stream tool_call args incrementally).
    for ch in stream.chunks:
        if not ch.get("choices"):
            continue
        tcs = ch["choices"][0].get("delta", {}).get("tool_calls")
        if not tcs:
            continue
        for j, tc in enumerate(tcs):
            assert "index" in tc, (
                f"{where}: tool_call delta missing 'index': {tc!r}"
            )
            if "function" in tc:
                fn = tc["function"]
                assert isinstance(fn, dict), f"{where}: tool_call.function must be dict"
                if "arguments" in fn:
                    assert isinstance(fn["arguments"], str), (
                        f"{where}: tool_call.function.arguments must be a JSON string"
                    )


# ── the actual test ──────────────────────────────────────────────────────


def test_claude_code_style_tool_calling_session():
    """Replay the canned tool-using session as N streaming requests.

    The messages array grows turn by turn, mirroring how Claude Code
    accumulates the conversation locally and resends the full history on
    every call. Each request must come back as a well-formed SSE stream.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, turn in enumerate(SESSION_TURNS):
        # Append this turn's new messages (user prompt, or prior
        # assistant tool_call + tool result pair).
        messages.extend(copy.deepcopy(turn.appended_messages))

        response = _send_turn(messages)
        assert response.status_code == 200, (
            f"turn[{i}] ({turn.description}): HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
        assert "text/event-stream" in response.headers.get("Content-Type", ""), (
            f"turn[{i}]: expected SSE, got Content-Type="
            f"{response.headers.get('Content-Type')!r}"
        )

        stream = _parse_sse(response)
        _assert_turn_invariants(i, stream)


def main() -> int:
    print(f"Replaying tool-calling session against {_base_url()} (model={_model()})")
    try:
        test_claude_code_style_tool_calling_session()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1
    print(f"  PASS ({len(SESSION_TURNS)} turns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
