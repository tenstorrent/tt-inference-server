#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Integration tests for the cancellation endpoint.

Tests:
  1. Explicit cancel  — DELETE /v1/requests/{id} stops an in-flight stream;
                        the stream closes with finish_reason="cancelled".
  2. Unknown cancel   — DELETE on a finished/unknown id returns 404.
  3. Cancel not found after natural finish — stream completes, then
                        DELETE on the same id returns 404.
  4. Double cancel    — two concurrent DELETEs: exactly one returns 200,
                        the other 404.

Usage:
  python tests/test_cancel_endpoint.py --base-url http://127.0.0.1:8000

Design note
-----------
The mock runner generates tokens very quickly (~8 µs each).  To avoid the
cancel arriving after natural completion, tests 1 and 4 use a large
max_tokens value and consume the stream in a background thread so the
DELETE fires from the main thread as soon as the first chunk's id is known.
"""

import argparse
import json
import os
import sys
import threading

import requests

DEFAULT_API_KEY = "your-secret-key"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Large enough for a cancel to arrive mid-stream.  When the server is started
# with MOCK_TOKEN_DELAY_US=1000, 200 tokens ≈ 200 ms — plenty of time for the
# DELETE to arrive before the stream finishes.
STREAM_MAX_TOKENS = 200


def _headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _stream_request(
    base_url: str, max_tokens: int = STREAM_MAX_TOKENS
) -> requests.Response:
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "prompt": "Hello, world!",
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }
    return requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        headers=_headers(),
        stream=True,
        timeout=30,
    )


def _parse_sse_chunks(response: requests.Response):
    """Yield parsed JSON objects from an SSE stream, stopping at [DONE]."""
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data.strip() == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            pass


def _consume_stream(
    response: requests.Response, result: dict, task_id_event: threading.Event
):
    """
    Read an SSE stream to completion in a background thread.

    Stores the first task id in result["task_id"] and sets task_id_event,
    then continues until it finds a chunk with a finish_reason, storing
    that in result["finish_reason"].
    """
    for chunk in _parse_sse_chunks(response):
        if result["task_id"] is None:
            result["task_id"] = chunk.get("id")
            task_id_event.set()
        choices = chunk.get("choices", [])
        if choices and choices[0].get("finish_reason") is not None:
            result["finish_reason"] = choices[0]["finish_reason"]
            break


# ---------------------------------------------------------------------------
# Test 1: explicit cancel stops an in-flight stream
# ---------------------------------------------------------------------------


def test_explicit_cancel(base_url: str) -> bool:
    print("\n=== Test 1: explicit cancel ===")

    response = _stream_request(base_url)
    if response.status_code != 200:
        print(
            f"  FAIL — stream request returned {response.status_code}: {response.text}"
        )
        return False

    result = {"task_id": None, "finish_reason": None}
    task_id_event = threading.Event()

    consume_thread = threading.Thread(
        target=_consume_stream, args=(response, result, task_id_event), daemon=True
    )
    consume_thread.start()

    # Cancel as soon as task_id is known to minimise the race window.
    if not task_id_event.wait(timeout=5):
        print("  FAIL — timed out waiting for first chunk")
        return False

    task_id = result["task_id"]
    print(f"  task_id: {task_id}")

    cancel_resp = requests.delete(
        f"{base_url}/v1/requests/{task_id}",
        headers=_headers(),
        timeout=5,
    )
    cancel_status = cancel_resp.status_code
    print(f"  DELETE status: {cancel_status}")

    consume_thread.join(timeout=15)
    finish_reason = result["finish_reason"]
    print(f"  finish_reason: {finish_reason}")

    if cancel_status != 200:
        print(f"  FAIL — expected DELETE 200, got {cancel_status}")
        return False

    if finish_reason != "cancelled":
        print(f"  FAIL — expected finish_reason='cancelled', got '{finish_reason}'")
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 2: DELETE on an unknown id returns 404
# ---------------------------------------------------------------------------


def test_cancel_unknown_id(base_url: str) -> bool:
    print("\n=== Test 2: cancel unknown id ===")

    fake_id = "does-not-exist-000000000000"
    resp = requests.delete(
        f"{base_url}/v1/requests/{fake_id}",
        headers=_headers(),
        timeout=5,
    )
    print(f"  DELETE status: {resp.status_code}")

    if resp.status_code != 404:
        print(f"  FAIL — expected 404, got {resp.status_code}")
        return False

    body = resp.json()
    if body.get("cancelled") is not False:
        print(f"  FAIL — expected cancelled=false in body, got: {body}")
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 3: DELETE after natural completion returns 404
# ---------------------------------------------------------------------------


def test_cancel_after_finish(base_url: str) -> bool:
    print("\n=== Test 3: cancel after natural finish ===")

    # Small max_tokens so the stream finishes quickly.
    response = _stream_request(base_url, max_tokens=4)
    if response.status_code != 200:
        print(f"  FAIL — stream request returned {response.status_code}")
        return False

    task_id = None
    last_finish_reason = None

    for chunk in _parse_sse_chunks(response):
        if task_id is None:
            task_id = chunk.get("id")
        choices = chunk.get("choices", [])
        if choices and choices[0].get("finish_reason") is not None:
            last_finish_reason = choices[0]["finish_reason"]

    print(f"  task_id: {task_id}, finish_reason: {last_finish_reason}")

    if last_finish_reason not in ("stop", "length"):
        print(f"  FAIL — expected natural finish_reason, got '{last_finish_reason}'")
        return False

    resp = requests.delete(
        f"{base_url}/v1/requests/{task_id}",
        headers=_headers(),
        timeout=5,
    )
    print(f"  POST-FINISH DELETE status: {resp.status_code}")

    if resp.status_code != 404:
        print(f"  FAIL — expected 404 after natural finish, got {resp.status_code}")
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 4: concurrent cancels — exactly one succeeds
# ---------------------------------------------------------------------------


def test_double_cancel(base_url: str) -> bool:
    print("\n=== Test 4: double cancel (idempotency) ===")

    response = _stream_request(base_url)
    if response.status_code != 200:
        print(f"  FAIL — stream request returned {response.status_code}")
        return False

    result = {"task_id": None, "finish_reason": None}
    task_id_event = threading.Event()

    consume_thread = threading.Thread(
        target=_consume_stream, args=(response, result, task_id_event), daemon=True
    )
    consume_thread.start()

    if not task_id_event.wait(timeout=5):
        print("  FAIL — timed out waiting for task_id")
        return False

    task_id = result["task_id"]
    statuses = [None, None]

    def do_cancel(idx):
        r = requests.delete(
            f"{base_url}/v1/requests/{task_id}", headers=_headers(), timeout=5
        )
        statuses[idx] = r.status_code

    t1 = threading.Thread(target=do_cancel, args=(0,))
    t2 = threading.Thread(target=do_cancel, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"  cancel 1 status: {statuses[0]}, cancel 2 status: {statuses[1]}")

    consume_thread.join(timeout=15)

    if sorted(statuses) != [200, 404]:
        print(f"  FAIL — expected one 200 and one 404, got {statuses}")
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Integration tests for the cancel endpoint"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token (defaults to OPENAI_API_KEY env or 'your-secret-key')",
    )
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    base_url = args.base_url.rstrip("/")
    print(f"Cancel endpoint integration tests — {base_url}")

    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"Server not healthy ({health.status_code}), aborting.")
            return 1
    except requests.RequestException as e:
        print(f"Cannot reach server: {e}")
        return 1

    results = {
        "explicit_cancel": test_explicit_cancel(base_url),
        "cancel_unknown_id": test_cancel_unknown_id(base_url),
        "cancel_after_finish": test_cancel_after_finish(base_url),
        "double_cancel": test_double_cancel(base_url),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
