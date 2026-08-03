#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
End-to-end tests for request cancellation via client disconnect.

Chat traffic goes through the Dynamo frontend (/v1/chat/completions).
Readiness / post-disconnect health use /v1/models (Dynamo has no /health).

Usage:
    python cancellation_e2e_test.py [--host HOST] [--port PORT] [--model MODEL]

Requires etcd + Dynamo frontend + a registered mock worker
(use run_e2e_with_server.sh, or test-gate's Dynamo bootstrap).
"""

import argparse
import os
import sys
import time

import requests

DEFAULT_API_KEY = "your-secret-key"
DEFAULT_MODEL = os.environ.get("DYNAMO_MODEL", "deepseek-ai/DeepSeek-R1-0528")


def _auth_headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _wait_for_frontend(base_url: str, model: str, timeout: int = 30) -> bool:
    """Wait until Dynamo frontend lists the model."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=2)
            if resp.status_code == 200 and model in resp.text:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def _frontend_healthy(base_url: str, model: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        return resp.status_code == 200 and model in resp.text
    except requests.RequestException:
        return False


def _streaming_request(model: str, max_tokens: int = 50) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": max_tokens,
        "stream": True,
    }


def _complete_streaming_request(
    base_url: str, model: str, max_tokens: int = 10
) -> list[str]:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=_streaming_request(model, max_tokens),
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


def test_server_healthy_after_disconnect(base_url: str, model: str) -> bool:
    """Disconnect mid-stream and verify the frontend still lists the model."""
    print("\n=== Test: Server healthy after disconnect ===")
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(model, max_tokens=200),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp.raise_for_status()

        count = 0
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                count += 1
                if count >= 3:
                    break
        resp.close()

        time.sleep(0.5)

        ok = _frontend_healthy(base_url, model)
        print(f"  Disconnected after {count} chunks, models_ok={ok}")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_request_completes_after_disconnect(base_url: str, model: str) -> bool:
    """After a disconnect, a new request should complete normally."""
    print("\n=== Test: Request completes after disconnect ===")
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(model, max_tokens=200),
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

        chunks = _complete_streaming_request(base_url, model, max_tokens=5)
        ok = len(chunks) > 0
        print(f"  Got {len(chunks)} chunks from follow-up request")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_multiple_rapid_disconnects(base_url: str, model: str) -> bool:
    """Multiple rapid disconnects should not degrade the server."""
    print("\n=== Test: Multiple rapid disconnects ===")
    try:
        for _ in range(5):
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json=_streaming_request(model, max_tokens=200),
                headers=_auth_headers(),
                stream=True,
                timeout=10,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    break
            resp.close()

        time.sleep(1.0)

        ok = _frontend_healthy(base_url, model)
        print(f"  5 rapid disconnects, models_ok={ok}")

        chunks = _complete_streaming_request(base_url, model, max_tokens=5)
        ok = ok and len(chunks) > 0
        print(f"  Follow-up request: {len(chunks)} chunks")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_disconnect_at_first_token(base_url: str, model: str) -> bool:
    """Disconnect immediately after receiving the very first SSE event."""
    print("\n=== Test: Disconnect at first token ===")
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(model, max_tokens=200),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                resp.close()
                break

        time.sleep(0.5)

        ok = _frontend_healthy(base_url, model)
        print(f"  Disconnected at first token, models_ok={ok}")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_concurrent_disconnect_and_new_request(base_url: str, model: str) -> bool:
    """Start a request, disconnect, and immediately start another."""
    print("\n=== Test: Concurrent disconnect and new request ===")
    try:
        resp1 = requests.post(
            f"{base_url}/v1/chat/completions",
            json=_streaming_request(model, max_tokens=200),
            headers=_auth_headers(),
            stream=True,
            timeout=10,
        )
        resp1.raise_for_status()
        for line in resp1.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                break
        resp1.close()

        chunks = _complete_streaming_request(base_url, model, max_tokens=5)
        ok = len(chunks) > 0
        print(f"  Immediate follow-up: {len(chunks)} chunks")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cancellation E2E tests (Dynamo)")
    parser.add_argument(
        "--host",
        default=os.environ.get("DYNAMO_HOST", "127.0.0.1"),
        help="Dynamo frontend host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DYNAMO_PORT", "8080")),
        help="Dynamo frontend port",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"Running cancellation E2E tests against Dynamo frontend {base_url}")
    print(f"  model={args.model}")

    if not _wait_for_frontend(base_url, args.model):
        print("ERROR: Dynamo frontend / model not ready within timeout")
        sys.exit(1)

    tests = [
        test_server_healthy_after_disconnect,
        test_request_completes_after_disconnect,
        test_multiple_rapid_disconnects,
        test_disconnect_at_first_token,
        test_concurrent_disconnect_and_new_request,
    ]

    passed = 0
    failed = 0
    for test in tests:
        if test(base_url, args.model):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
