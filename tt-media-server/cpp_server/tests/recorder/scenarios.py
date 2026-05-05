#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Drive a curated set of HTTP requests against a running server.

Used together with `run_recorder_e2e.sh` to exercise the API/service layer
with a deterministic sequence of requests. The server, when launched with
`TT_RUNNER_RECORDER_MODE=record`, writes a JSONL trace of what crossed the
producer side of the task / cancel queues; in `assert` mode it compares
against the committed fixture instead.

Scenarios are intentionally simple and sequential. They are NOT meant to
test correctness of the model output; they only drive the API surface so
that the recorder captures what the API+service layer hands to the
worker/runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Iterable

import requests

DEFAULT_API_KEY = "your-secret-key"
DEFAULT_TIMEOUT = 120


def auth_headers() -> dict[str, str]:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def wait_for_ready(base_url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/tt-liveness", timeout=2)
            if resp.ok and resp.json().get("model_ready") is True:
                return
        except (requests.ConnectionError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server not ready at {base_url} after {timeout}s")


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        f"{base_url}{path}",
        headers=auth_headers(),
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )


def consume_stream(base_url: str, payload: dict[str, Any]) -> int:
    """POST a streaming chat completion and drain the SSE stream.

    Returns the number of received `data:` chunks (final `[DONE]` excluded).
    The recorder captures what the server submits to the runner, so we don't
    inspect the chunks themselves here; we just have to consume the stream
    to its natural end so the request lifecycle completes deterministically.
    """
    chunks = 0
    with requests.post(
        f"{base_url}/v1/chat/completions",
        headers=auth_headers(),
        json=payload,
        stream=True,
        timeout=DEFAULT_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue
            data = raw[len("data: "):]
            if data.strip() == "[DONE]":
                break
            chunks += 1
    return chunks


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_non_streaming(base_url: str) -> None:
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [
            {"role": "user", "content": "say hi"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    resp = post_json(base_url, "/v1/chat/completions", payload)
    resp.raise_for_status()


def scenario_streaming(base_url: str) -> None:
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [
            {"role": "user", "content": "stream please"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
    }
    consume_stream(base_url, payload)


def scenario_multi_turn(base_url: str) -> None:
    """Two-turn conversation that exercises the prefix-cache / session path.

    The second turn includes the assistant message from the first plus a new
    user message, which is what triggers `LLMController::resolveSession` to
    take the prefix-cache branch and submit a `continuation` Sequence.
    """
    first = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [
            {"role": "user", "content": "turn one"},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    resp = post_json(base_url, "/v1/chat/completions", first)
    resp.raise_for_status()
    body = resp.json()
    assistant_text = body["choices"][0]["message"].get("content", "")

    second = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": "turn two"},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    resp = post_json(base_url, "/v1/chat/completions", second)
    resp.raise_for_status()


def scenario_tool_calling(base_url: str) -> None:
    """tool_choice=auto with a tools array set."""
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [
            {"role": "user", "content": "what is the weather in NYC?"},
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
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    resp = post_json(base_url, "/v1/chat/completions", payload)
    resp.raise_for_status()


SCENARIOS = [
    ("non_streaming", scenario_non_streaming),
    ("streaming", scenario_streaming),
    ("multi_turn", scenario_multi_turn),
    ("tool_calling", scenario_tool_calling),
]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of scenario names to run.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_url = f"http://{args.host}:{args.port}"
    print(f"[scenarios] waiting for server at {base_url}")
    wait_for_ready(base_url)

    selected = SCENARIOS
    if args.only:
        wanted = set(args.only.split(","))
        selected = [s for s in SCENARIOS if s[0] in wanted]
        if not selected:
            print(f"[scenarios] --only={args.only!r} matched no scenarios")
            return 2

    for name, fn in selected:
        print(f"[scenarios] running: {name}")
        try:
            fn(base_url)
        except Exception as exc:  # noqa: BLE001 -- driver script, propagate via exit code
            print(f"[scenarios] FAILED {name}: {exc}")
            return 1

    print("[scenarios] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
