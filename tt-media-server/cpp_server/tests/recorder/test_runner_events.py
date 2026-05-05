#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""End-to-end assertions on what the API/Service layer hands to the runner.

Each scenario:
    1. clears the in-memory event log via DELETE /debug/runner-events
    2. drives one or more HTTP requests
    3. fetches the resulting events via GET /debug/runner-events
    4. asserts inline -- request and assertion live next to each other

The recorder is enabled by setting `TT_RUNNER_RECORDER_ENABLED=1` on the
server process (see `run_recorder_e2e.sh`).

Goals:
    - Refactor-safety net for the API / controllers / services / sessions /
      disaggregation glue. Anything below the producer side of the task
      queue is out of scope.
    - Self-contained: no fixture files, no record/replay step. Every
      expectation is in code, right next to the request that produced it.

Anti-goals:
    - Not a model-output correctness test. Token contents are intentionally
      represented only by `tokens_xx64` (xxhash) + `tokens_len`.
"""

from __future__ import annotations

import argparse
import sys
import time
import unittest
from typing import Any

import requests

DEFAULT_TIMEOUT = 60
DEFAULT_API_KEY = "your-secret-key"


class RunnerEventClient:
    """Thin HTTP client around `/v1/...` and `/debug/runner-events`."""

    def __init__(self, base_url: str, api_key: str = DEFAULT_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def wait_for_ready(self, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.base_url}/tt-liveness", timeout=2)
                if resp.ok and resp.json().get("model_ready") is True:
                    return
            except (requests.ConnectionError, ValueError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Server not ready at {self.base_url} after {timeout}s")

    def clear_events(self) -> int:
        resp = requests.delete(
            f"{self.base_url}/debug/runner-events", timeout=5
        )
        resp.raise_for_status()
        return resp.json()["last_seq"]

    def get_events(self, since_seq: int = 0) -> list[dict[str, Any]]:
        url = f"{self.base_url}/debug/runner-events"
        if since_seq:
            url += f"?since_seq={since_seq}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        body = resp.json()
        return body["events"]

    def chat(self, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

    def stream_chat(self, payload: dict[str, Any]) -> int:
        chunks = 0
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                if raw[len("data: "):].strip() == "[DONE]":
                    break
                chunks += 1
        return chunks


# Populated by main() before unittest.main().
_CLIENT: RunnerEventClient | None = None


def client() -> RunnerEventClient:
    assert _CLIENT is not None, "client() called before main() initialized it"
    return _CLIENT


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunnerEvents(unittest.TestCase):
    """Each test owns its slice of the event log via clear_events()."""

    def setUp(self) -> None:
        self.client = client()
        self.client.clear_events()

    def _exactly_one_task(self) -> dict[str, Any]:
        events = self.client.get_events()
        task_events = [e for e in events if e["kind"] == "task_submitted"]
        self.assertEqual(
            len(task_events),
            1,
            f"expected exactly 1 task_submitted event, got: {events}",
        )
        return task_events[0]

    # ---- non-streaming ----------------------------------------------------

    def test_non_streaming_submits_single_task(self) -> None:
        resp = self.client.chat(
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": False,
            }
        )
        resp.raise_for_status()

        event = self._exactly_one_task()
        self.assertEqual(event["kind"], "task_submitted")
        self.assertFalse(event["continuation"])
        self.assertFalse(event["disaggregated"])
        self.assertEqual(event["sampling"]["max_tokens"], 8)
        self.assertEqual(event["sampling"]["temperature"], 0)
        self.assertEqual(event["sampling"]["response_format"], "TEXT")
        self.assertEqual(event["sampling"]["tools_count"], 0)
        self.assertIsNone(event["sampling"]["tool_choice_type"])
        self.assertGreater(event["num_prompt_tokens"], 0)
        self.assertEqual(event["tokens_len"], event["num_prompt_tokens"])

    # ---- streaming --------------------------------------------------------

    def test_streaming_submits_single_task(self) -> None:
        chunks = self.client.stream_chat(
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "messages": [{"role": "user", "content": "stream please"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": True,
            }
        )
        self.assertGreater(chunks, 0)

        event = self._exactly_one_task()
        self.assertFalse(event["continuation"])
        self.assertEqual(event["sampling"]["max_tokens"], 8)

    # ---- multi-turn / prefix cache ---------------------------------------

    def test_multi_turn_second_call_is_continuation(self) -> None:
        first = self.client.chat(
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "messages": [{"role": "user", "content": "turn one"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": False,
            }
        )
        first.raise_for_status()
        assistant_text = first.json()["choices"][0]["message"].get("content", "")

        # Snapshot the seq of the first task so we can isolate the second.
        first_events = self.client.get_events()
        self.assertEqual(len(first_events), 1)
        first_seq = first_events[0]["seq"]
        self.assertFalse(first_events[0]["continuation"])

        second = self.client.chat(
            {
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
        )
        second.raise_for_status()

        new_events = self.client.get_events(since_seq=first_seq)
        self.assertEqual(len(new_events), 1)
        self.assertTrue(
            new_events[0]["continuation"],
            "second turn should reuse the prefix-cached session",
        )
        self.assertEqual(new_events[0]["sampling"]["max_tokens"], 4)

    # ---- tool calling -----------------------------------------------------

    def test_tool_choice_auto_propagates_to_runner(self) -> None:
        self.client.chat(
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
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

        event = self._exactly_one_task()
        self.assertEqual(event["sampling"]["tools_count"], 1)
        self.assertEqual(event["sampling"]["tool_choice_type"], "auto")

    # ---- structured output ------------------------------------------------

    def test_response_format_json_schema_propagates(self) -> None:
        self.client.chat(
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
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

        event = self._exactly_one_task()
        self.assertEqual(event["sampling"]["response_format"], "JSON_SCHEMA")
        self.assertTrue(event["sampling"]["json_schema_present"])

    # ---- determinism: same request twice -> same fingerprint -------------

    def test_identical_requests_produce_identical_token_fingerprints(self) -> None:
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "messages": [{"role": "user", "content": "deterministic"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        }

        # Force fresh sessions for both calls so the prefix-cache layer does
        # not turn the second call into a `continuation` (which would short
        # the prompt).
        self.client.chat(payload).raise_for_status()
        first = self._exactly_one_task()

        self.client.clear_events()
        self.client.chat(payload).raise_for_status()
        second = self._exactly_one_task()

        self.assertEqual(first["tokens_xx64"], second["tokens_xx64"])
        self.assertEqual(first["tokens_len"], second["tokens_len"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    global _CLIENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--unittest-args", nargs="*", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)

    _CLIENT = RunnerEventClient(f"http://{args.host}:{args.port}", args.api_key)
    print(f"[runner-events] waiting for server at {_CLIENT.base_url}")
    _CLIENT.wait_for_ready()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestRunnerEvents)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
