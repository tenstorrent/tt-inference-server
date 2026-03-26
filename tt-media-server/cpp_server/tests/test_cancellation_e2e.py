# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
End-to-end tests for request cancellation over HTTP.

Verifies that when a streaming client drops its connection, the C++ server:
  1. Does not crash (health endpoint remains responsive).
  2. Continues serving subsequent requests correctly after the disconnect.
  3. Handles multiple rapid disconnects without degradation.

These tests run against a live C++ server started with LLM_DEVICE_BACKEND=mock.
Set SERVER_BASE_URL to point at a running server (default: http://127.0.0.1:8000).
"""

import json
import os
import threading
import time

import requests

SERVER_BASE_URL = os.environ.get("SERVER_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("OPENAI_API_KEY", "your-secret-key")

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def _payload(max_tokens: int) -> dict:
    return {
        "model": "test",
        "prompt": "Hello",
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _health_ok() -> bool:
    try:
        resp = requests.get(
            f"{SERVER_BASE_URL}/health",
            timeout=5,
        )
        return resp.status_code == 200 and resp.json().get("status") == "healthy"
    except Exception:
        return False


def _stream_and_drop(max_tokens: int, drop_after: int) -> int:
    """
    Open a streaming completion, read up to *drop_after* SSE data lines, then
    close the connection abruptly.  Returns the number of data chunks received
    before the close.
    """
    resp = requests.post(
        f"{SERVER_BASE_URL}/v1/completions",
        json=_payload(max_tokens),
        headers=_HEADERS,
        stream=True,
        timeout=30,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    received = 0
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except Exception:
                continue
            if not chunk.get("choices"):
                continue  # skip usage/empty-choices events
            received += 1
            if received >= drop_after:
                # Close the socket without consuming the rest — simulates
                # an abrupt TCP close from the client side.
                resp.close()
                return received
    except Exception:
        pass
    return received


def _stream_full(max_tokens: int) -> int:
    """Stream a completion to completion and return the total data-chunk count."""
    resp = requests.post(
        f"{SERVER_BASE_URL}/v1/completions",
        json=_payload(max_tokens),
        headers=_HEADERS,
        stream=True,
        timeout=30,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    received = 0
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except Exception:
            continue
        if chunk.get("choices"):  # skip usage/empty-choices events
            received += 1
    return received


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Server stays healthy after a single mid-stream disconnect
# ─────────────────────────────────────────────────────────────────────────────


def test_health_after_single_disconnect():
    """Server health endpoint returns 'healthy' after one dropped connection."""
    received = _stream_and_drop(max_tokens=200, drop_after=5)

    assert received == 5, f"Expected to drop after exactly 5 tokens, got {received}"

    # Allow the server a moment to process the TCP close event.
    time.sleep(0.3)

    assert _health_ok(), "Health check failed after a single disconnect"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Subsequent request completes correctly after a disconnect
# ─────────────────────────────────────────────────────────────────────────────


def test_subsequent_request_completes_after_disconnect():
    """A new request submitted after a disconnect receives all expected tokens."""
    expected = 30

    _stream_and_drop(max_tokens=200, drop_after=5)
    time.sleep(0.3)

    received = _stream_full(max_tokens=expected)

    assert received == expected, (
        f"Follow-up request received {received} tokens instead of {expected}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Multiple rapid disconnects don't degrade the server
# ─────────────────────────────────────────────────────────────────────────────


def test_multiple_rapid_disconnects_no_degradation():
    """Repeated rapid disconnects leave the server able to serve a full request."""
    num_disconnects = 8
    final_tokens = 25

    for _ in range(num_disconnects):
        _stream_and_drop(max_tokens=100, drop_after=3)

    time.sleep(0.5)

    assert _health_ok(), (
        f"Health check failed after {num_disconnects} rapid disconnects"
    )

    received = _stream_full(max_tokens=final_tokens)

    assert received == final_tokens, (
        f"Server degraded after {num_disconnects} disconnects: "
        f"expected {final_tokens} tokens, got {received}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Disconnect at the very first token
# ─────────────────────────────────────────────────────────────────────────────


def test_disconnect_at_first_token():
    """Dropping a connection after the very first token doesn't hang the server."""
    received = _stream_and_drop(max_tokens=100, drop_after=1)
    assert received == 1

    time.sleep(0.3)

    assert _health_ok(), "Health check failed after disconnect at first token"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Concurrent disconnect and new request
# ─────────────────────────────────────────────────────────────────────────────


def test_concurrent_disconnect_and_new_request():
    """A new request started concurrently with a disconnect completes in full."""
    expected = 20
    drop_result: list[int] = []
    new_result: list[int] = []

    def _drop():
        drop_result.append(_stream_and_drop(max_tokens=200, drop_after=5))

    t = threading.Thread(target=_drop)
    t.start()

    # Small stagger so the disconnect is in-flight while the new request starts.
    time.sleep(0.05)
    new_result.append(_stream_full(max_tokens=expected))

    t.join(timeout=30)

    assert drop_result, "Drop thread did not complete"
    assert drop_result[0] == 5, f"Expected to drop after 5 tokens, got {drop_result[0]}"
    assert new_result[0] == expected, (
        f"Concurrent request received {new_result[0]} tokens instead of {expected}"
    )
