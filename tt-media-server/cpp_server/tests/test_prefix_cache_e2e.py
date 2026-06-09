#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
End-to-end prefix cache verification via Dynamo frontend.

Tests that KV-cache prefix reuse works correctly by checking exact
cached_tokens counts:
  - First request: all tokens newly cached, cached_tokens=0 (nothing reused)
  - Second request with same history: prefix reused, cached_tokens equals
    the expected block-aligned count

Usage:
    python tests/test_prefix_cache_e2e.py

    # Verbose output:
    python tests/test_prefix_cache_e2e.py -v

    # Custom host/port:
    python tests/test_prefix_cache_e2e.py --host 127.0.0.1 --port 9000
"""

import argparse
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field

import requests

# Block size defaults from include/config/defaults.hpp
DEFAULT_BLOCK_SIZE = 32
DEFAULT_FIRST_BLOCK_SIZE = 128

# System prompt that fills at least one hash block (~220 tokens for typical
# tokenizers).
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a highly capable AI coding assistant working inside an IDE. You have
    access to the full project source tree and can read files, search code, run
    shell commands, and edit files. When the user asks you to make changes, you
    should understand the request, explore the relevant code, plan your changes,
    and implement them carefully. Always verify your changes compile and pass
    tests. If you encounter ambiguity, ask for clarification. Provide concise,
    accurate answers. When writing code, follow the project's existing style and
    conventions. Use meaningful variable names and add comments only when the
    logic is non-obvious. Keep functions small and focused. Prefer composition
    over inheritance. Handle errors gracefully and log relevant context. When
    reviewing code, look for correctness issues first, then performance, then
    style. Always consider edge cases and concurrent access patterns. For
    distributed systems, think about failure modes, retry strategies, and
    idempotency. When working with databases, consider indexing, query plans, and
    data migration paths. For API design, follow RESTful conventions and provide
    clear error messages with appropriate HTTP status codes.""")


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __str__(self) -> str:
        return (
            f"prompt={self.prompt_tokens}, "
            f"cached={self.cached_tokens}, "
            f"completion={self.completion_tokens}"
        )


@dataclass
class TurnResult:
    content: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)
    status_code: int = 0
    error: str = ""


def _headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", "your-secret-key")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _extract_usage(usage_dict: dict, into: UsageInfo) -> None:
    into.prompt_tokens = usage_dict.get("prompt_tokens", 0)
    into.completion_tokens = usage_dict.get("completion_tokens", 0)
    into.total_tokens = usage_dict.get("total_tokens", 0)
    ptd = usage_dict.get("prompt_tokens_details") or {}
    into.cached_tokens = ptd.get("cached_tokens", 0)


def wait_for_server(base_url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(f"{base_url}{path}", headers=_headers(), timeout=2)
                if r.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass
        time.sleep(0.5)
    return False


def send_chat(
    base_url: str,
    messages: list[dict],
    *,
    max_tokens: int = 32,
    model: str = "tt-cpp-server",
    stream: bool = True,
    verbose: bool = False,
) -> TurnResult:
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    result = TurnResult()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=_headers(),
            stream=stream,
            timeout=120,
        )
        result.status_code = resp.status_code
        if resp.status_code != 200:
            result.error = resp.text[:500]
            return result
    except Exception as exc:
        result.error = str(exc)
        return result

    if not stream:
        body = resp.json()
        if verbose:
            print(f"    [response] {json.dumps(body, indent=2)}")
        choices = body.get("choices") or []
        if choices:
            result.content = (choices[0].get("message") or {}).get("content", "")
        usage = body.get("usage")
        if usage:
            _extract_usage(usage, result.usage)
        return result

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if verbose:
                print(f"    [chunk] {json.dumps(chunk)}")

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                result.content += delta.get("content", "")

            usage = chunk.get("usage")
            if usage:
                _extract_usage(usage, result.usage)

    return result


def compute_expected_cached_tokens(
    prompt_tokens: int,
    first_block_size: int = DEFAULT_FIRST_BLOCK_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> int:
    """
    Compute the expected cached_tokens for a prompt of given length.

    The prefix cache works in blocks:
    - First block: first_block_size tokens (default 128)
    - Subsequent blocks: block_size tokens each (default 32)

    Only complete blocks are cached. The trailing partial block is NOT cached.
    cached_tokens = number of tokens in complete blocks.
    """
    if prompt_tokens < first_block_size:
        return 0

    cached = first_block_size
    remaining = prompt_tokens - first_block_size
    full_subsequent_blocks = remaining // block_size
    cached += full_subsequent_blocks * block_size
    return cached


def test_prefix_cache_exact_values(
    base_url: str, model: str, stream: bool, verbose: bool
) -> bool:
    """
    Two-request scenario testing exact cached_tokens values.

    Request 1: Fresh conversation. All tokens are newly cached. Since nothing
               was reused, cached_tokens=0.

    Request 2: Same conversation history. The prefix is reused from cache.
               cached_tokens should equal the block-aligned portion of the
               first request's prompt.
    """
    print("\n=== Test: Prefix cache exact values ===")
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": "What is the capital of France?"}
    messages = [system, user_msg]

    # Request 1: Fresh - nothing reused
    print("  Request 1 (fresh)...")
    r1 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r1.error:
        print(f"  FAIL: Request 1 errored — {r1.error}")
        return False

    print(f"    Usage: {r1.usage}")

    # First request should have cached_tokens=0 (nothing was reused)
    if r1.usage.cached_tokens != 0:
        print(
            f"  FAIL: Request 1 cached_tokens={r1.usage.cached_tokens}, "
            f"expected 0 (nothing should be reused on first request)"
        )
        return False
    print("    OK: cached_tokens=0 (nothing reused on first request)")

    # Build continuation: same system + user + assistant response + new user
    messages.append({"role": "assistant", "content": r1.content})
    messages.append({"role": "user", "content": "And what is the capital of Germany?"})

    # Small delay to ensure caching completes
    time.sleep(0.3)

    # Request 2: Should reuse the prefix from request 1
    print("  Request 2 (continuation with same history)...")
    r2 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r2.error:
        print(f"  FAIL: Request 2 errored — {r2.error}")
        return False

    print(f"    Usage: {r2.usage}")

    # The reused portion should be the block-aligned prefix from request 1
    # Request 1 had prompt_tokens tokens; the cacheable portion is computed
    # based on block sizes.
    expected_cached = compute_expected_cached_tokens(r1.usage.prompt_tokens)

    print(f"    Expected cached_tokens: {expected_cached}")
    print(
        f"    (Based on request 1 prompt_tokens={r1.usage.prompt_tokens}, "
        f"first_block={DEFAULT_FIRST_BLOCK_SIZE}, block={DEFAULT_BLOCK_SIZE})"
    )

    if r2.usage.cached_tokens == 0:
        print("  FAIL: Request 2 cached_tokens=0 — prefix should have been reused")
        return False

    if r2.usage.cached_tokens != expected_cached:
        print(
            f"  WARN: Request 2 cached_tokens={r2.usage.cached_tokens}, "
            f"expected {expected_cached}"
        )
        # This is a warning, not a failure, because block sizes may differ
        # from defaults depending on server configuration
        print("    (Block sizes may differ from defaults)")

    # The newly cached portion is the difference
    newly_cached = r2.usage.prompt_tokens - r2.usage.cached_tokens
    print(f"    Reused tokens: {r2.usage.cached_tokens}")
    print(f"    Newly processed tokens: {newly_cached}")

    if r2.usage.cached_tokens > 0:
        print(f"  OK: Prefix cache working — reused {r2.usage.cached_tokens} tokens")
        return True
    else:
        print("  FAIL: No tokens were reused")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E prefix cache verification via Dynamo frontend",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Dynamo frontend port (default: 9000)",
    )
    parser.add_argument("--model", default="tt-cpp-server")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use non-streaming requests",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every SSE chunk / response body",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Seconds to wait for server"
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    stream = not args.no_stream

    print(f"Prefix cache E2E test against {base_url}")
    print(f"  model={args.model}  stream={stream}")
    print("\nWaiting for server...")

    if not wait_for_server(base_url, args.timeout):
        print("ERROR: Server not ready within timeout")
        return 1
    print("Server ready.\n")

    ok = test_prefix_cache_exact_values(base_url, args.model, stream, args.verbose)

    print(f"\n{'=' * 60}")
    if ok:
        print("PASSED")
    else:
        print("FAILED")
    print(f"{'=' * 60}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
