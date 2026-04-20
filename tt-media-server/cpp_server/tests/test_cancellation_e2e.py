#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
End-to-end tests for request cancellation via client disconnect.

These tests verify that the server correctly handles client disconnections
during streaming and remains healthy afterwards.

Usage:
    python test_cancellation_e2e.py [--host HOST] [--port PORT]

Requires a running server (use run_e2e_with_server.sh for automated setup).
"""

import argparse
import os
import sys
import time

import requests

DEFAULT_API_KEY = "your-secret-key"


def _auth_headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _create_test_session(base_url: str) -> str:
    """Create a session for testing with a pre-allocated slot. Returns session_id."""
    try:
        # Create a session with slot_id to avoid memory manager dependency in tests
        resp = requests.post(
            f"{base_url}/v1/sessions",
            json={"slot_id": 0},  # Use slot 0 for testing
            headers=_auth_headers(),
            timeout=5,
        )
        if resp.status_code == 201:
            session_data = resp.json()
            return session_data.get("session_id", "")
    except Exception as e:
        print(f"  Warning: Could not create session: {e}")
    return ""


def _wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Wait for the server to become ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def _streaming_request(
    base_url: str, max_tokens: int = 50, session_id: str = None
) -> dict:
    """Build a streaming chat completion request payload."""
    request = {
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if session_id:
        request["session_id"] = session_id
    return request


def _complete_streaming_request(
    base_url: str, max_tokens: int = 10, session_id: str = ""
) -> list[str]:
    """Make a streaming request and collect all SSE chunks."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=_streaming_request(base_url, max_tokens, session_id=session_id),
        headers=_auth_headers(),
        stream=True,
        timeout=30,
    )
    resp.raise_for_status()

    chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            chunks.append(data)
    return chunks


def test_server_healthy_after_disconnect(base_url: str, session_id: str = "") -> bool:
    """Disconnect mid-stream and verify the server stays healthy."""
    print("\n=== Test: Server healthy after disconnect ===")
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(base_url, max_tokens=200, session_id=session_id),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp.raise_for_status()

        # Read a few chunks then close abruptly
        count = 0
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                count += 1
                if count >= 3:
                    break
        resp.close()

        # Give the server a moment to process the disconnect
        time.sleep(0.5)

        # Health check
        health = requests.get(f"{base_url}/health", timeout=5)
        ok = health.status_code == 200
        print(f"  Disconnected after {count} chunks, health={health.status_code}")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_request_completes_after_disconnect(
    base_url: str, session_id: str = ""
) -> bool:
    """After a disconnect, a new request should complete normally."""
    print("\n=== Test: Request completes after disconnect ===")
    try:
        # First: disconnect mid-stream
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(base_url, max_tokens=200, session_id=session_id),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                break
        resp.close()
        time.sleep(0.5)

        # Second: complete a full request
        chunks = _complete_streaming_request(
            base_url, max_tokens=5, session_id=session_id
        )
        ok = len(chunks) > 0
        print(f"  Got {len(chunks)} chunks from follow-up request")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_multiple_rapid_disconnects(base_url: str, session_id: str = "") -> bool:
    """Multiple rapid disconnects should not degrade the server."""
    print("\n=== Test: Multiple rapid disconnects ===")
    try:
        for i in range(5):
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json=_streaming_request(
                    base_url, max_tokens=200, session_id=session_id
                ),
                headers=_auth_headers(),
                stream=True,
                timeout=10,
            )
            resp.raise_for_status()
            # Read 1 chunk and disconnect
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    break
            resp.close()

        time.sleep(1.0)

        # Server should still be healthy
        health = requests.get(f"{base_url}/health", timeout=5)
        ok = health.status_code == 200
        print(f"  5 rapid disconnects, health={health.status_code}")

        # And a full request should work
        chunks = _complete_streaming_request(
            base_url, max_tokens=5, session_id=session_id
        )
        ok = ok and len(chunks) > 0
        print(f"  Follow-up request: {len(chunks)} chunks")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_disconnect_at_first_token(base_url: str, session_id: str = "") -> bool:
    """Disconnect immediately after receiving the very first SSE event."""
    print("\n=== Test: Disconnect at first token ===")
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(base_url, max_tokens=200, session_id=session_id),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp.raise_for_status()
        # Close immediately after first data line
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                resp.close()
                break

        time.sleep(0.5)

        health = requests.get(f"{base_url}/health", timeout=5)
        ok = health.status_code == 200
        print(f"  Disconnected at first token, health={health.status_code}")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_concurrent_disconnect_and_new_request(
    base_url: str, session_id: str = ""
) -> bool:
    """Start a request, disconnect, and immediately start another."""
    print("\n=== Test: Concurrent disconnect and new request ===")
    try:
        # Start streaming
        resp1 = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(base_url, max_tokens=200, session_id=session_id),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp1.raise_for_status()
        for line in resp1.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                break
        resp1.close()

        # Immediately start a new request (no sleep)
        chunks = _complete_streaming_request(
            base_url, max_tokens=5, session_id=session_id
        )
        ok = len(chunks) > 0
        print(f"  Immediate follow-up: {len(chunks)} chunks")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cancellation E2E tests")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"Running cancellation E2E tests against {base_url}")

    if not _wait_for_server(base_url):
        print("ERROR: Server not ready within timeout")
        sys.exit(1)

    # Create a test session with pre-allocated slot to avoid memory manager dependency
    session_id = _create_test_session(base_url)
    if session_id:
        print(f"Created test session: {session_id}")
    else:
        print("Running tests without session (server may not support sessions)")

    tests = [
        lambda url: test_server_healthy_after_disconnect(url, session_id),
        lambda url: test_request_completes_after_disconnect(url, session_id),
        lambda url: test_multiple_rapid_disconnects(url, session_id),
        lambda url: test_disconnect_at_first_token(url, session_id),
        lambda url: test_concurrent_disconnect_and_new_request(url, session_id),
    ]

    passed = 0
    failed = 0
    for test in tests:
        if test(base_url):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
