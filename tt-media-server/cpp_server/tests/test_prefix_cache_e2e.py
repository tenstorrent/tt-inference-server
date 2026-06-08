#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
End-to-end prefix cache verification.

Sends multi-turn conversations and inspects
``usage.prompt_tokens_details.cached_tokens`` to verify that KV-cache
prefix reuse is working across turns.

Can target either:
  - cpp_server directly (port 8000, default) — always works, exercises the
    same LLMPipeline / SessionManager / prefix-cache code.
  - Dynamo frontend (port 9000) — requires etcd + registered backend.

Usage:
    # Default: hit cpp_server on :8000
    python tests/test_prefix_cache_e2e.py

    # Verbose output (show every SSE chunk / response body):
    python tests/test_prefix_cache_e2e.py -v

    # Hit the Dynamo frontend (needs etcd + backend registered):
    python tests/test_prefix_cache_e2e.py --port 9000 --model tt-cpp-server

    # Dump raw response for debugging:
    python tests/test_prefix_cache_e2e.py --dump

    # Non-streaming mode (simpler response parsing):
    python tests/test_prefix_cache_e2e.py --no-stream
"""

import argparse
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------------------
# A long-ish system prompt so the tokenized prefix comfortably fills at least
# one hash block (default first-block size is 64–256 tokens depending on
# config).  ~220 BPE tokens for DeepSeek / Llama tokenizers.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def cache_pct(self) -> float:
        return (
            (self.cached_tokens / self.prompt_tokens * 100)
            if self.prompt_tokens
            else 0.0
        )


@dataclass
class TurnResult:
    content: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)
    status_code: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", "your-secret-key")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _extract_usage(usage_dict: dict, into: UsageInfo) -> None:
    """Fill *into* from an OpenAI ``usage`` JSON object."""
    into.prompt_tokens = usage_dict.get("prompt_tokens", 0)
    into.completion_tokens = usage_dict.get("completion_tokens", 0)
    into.total_tokens = usage_dict.get("total_tokens", 0)
    ptd = usage_dict.get("prompt_tokens_details") or {}
    into.cached_tokens = ptd.get("cached_tokens", 0)
    ctd = usage_dict.get("completion_tokens_details") or {}
    into.reasoning_tokens = ctd.get("reasoning_tokens", 0)


def wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Block until *base_url* responds to a health or models probe."""
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
    """Send a chat completion and return content + usage."""
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

    # --- non-streaming ---------------------------------------------------
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

    # --- streaming (SSE) -------------------------------------------------
    # Collect raw lines so we can diagnose format mismatches when nothing
    # parses.  The Dynamo frontend may use bare `data:` lines, named SSE
    # events (`event: …\ndata: …`), or something else entirely.
    raw_lines: list[str] = []
    parsed_any = False

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        raw_lines.append(line)

        # Standard OpenAI SSE: `data: {json}`
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            parsed_any = True
            if verbose:
                print(f"    [chunk] {json.dumps(chunk)}")

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                result.content += delta.get("content", "")

            usage = chunk.get("usage")
            if usage:
                _extract_usage(usage, result.usage)

    # If nothing parsed, dump the first few raw lines so the user can see
    # what the server is actually sending.
    if not parsed_any and raw_lines:
        sample = raw_lines[:20]
        print("    [DIAG] No SSE chunks parsed. Raw response lines:")
        for ln in sample:
            print(f"    [DIAG]   {ln!r}")
        if len(raw_lines) > 20:
            print(f"    [DIAG]   … ({len(raw_lines) - 20} more lines)")
    elif parsed_any and result.usage.prompt_tokens == 0 and result.content == "":
        # Chunks parsed but nothing useful extracted — show a sample.
        sample = raw_lines[:10]
        print("    [DIAG] Chunks parsed but content/usage empty. Sample lines:")
        for ln in sample:
            print(f"    [DIAG]   {ln!r}")

    return result


def _print_turn(turn_num: int, result: TurnResult, label: str = "") -> None:
    tag = f"  Turn {turn_num}" + (f" ({label})" if label else "")
    u = result.usage
    preview = result.content[:80].replace("\n", " ")
    if len(result.content) > 80:
        preview += "…"
    print(f"{tag}:")
    print(f"    prompt_tokens     = {u.prompt_tokens}")
    print(f"    cached_tokens     = {u.cached_tokens}")
    print(f"    completion_tokens = {u.completion_tokens}")
    print(f"    cache_hit_ratio   = {u.cache_pct():.1f}%")
    print(f"    content           = {preview!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_turn_prefix_cache(
    base_url: str, model: str, stream: bool, verbose: bool
) -> bool:
    """3-turn conversation.  Turn 1 is fresh; turns 2 & 3 should show
    growing ``cached_tokens``."""
    print("\n=== Test: Multi-turn prefix cache ===")
    system = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system, {"role": "user", "content": "What is the capital of France?"}]

    r1 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r1.error:
        print(f"  FAIL: Turn 1 errored — {r1.error}")
        return False
    _print_turn(1, r1, "fresh")

    messages.append({"role": "assistant", "content": r1.content})
    messages.append({"role": "user", "content": "And what is the capital of Germany?"})
    r2 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r2.error:
        print(f"  FAIL: Turn 2 errored — {r2.error}")
        return False
    _print_turn(2, r2, "continuation")

    messages.append({"role": "assistant", "content": r2.content})
    messages.append(
        {
            "role": "user",
            "content": "Which of those two cities is larger by population?",
        }
    )
    r3 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r3.error:
        print(f"  FAIL: Turn 3 errored — {r3.error}")
        return False
    _print_turn(3, r3, "continuation")

    ok = True
    if r1.usage.cached_tokens != 0:
        print(
            f"  WARN: Turn 1 cached_tokens={r1.usage.cached_tokens} (expected 0 for fresh turn)"
        )

    if r2.usage.cached_tokens == 0:
        print("  FAIL: Turn 2 cached_tokens=0 — prefix should have been reused")
        ok = False
    else:
        print(f"  OK: Turn 2 reused {r2.usage.cached_tokens} cached tokens")

    if r3.usage.cached_tokens == 0:
        print("  FAIL: Turn 3 cached_tokens=0 — prefix should have been reused")
        ok = False
    elif r3.usage.cached_tokens <= r2.usage.cached_tokens:
        print(
            f"  WARN: Turn 3 cached_tokens ({r3.usage.cached_tokens}) did not grow "
            f"vs Turn 2 ({r2.usage.cached_tokens})"
        )
    else:
        print(
            f"  OK: Turn 3 cached tokens grew {r2.usage.cached_tokens} → {r3.usage.cached_tokens}"
        )

    return ok


def test_identical_prompt_caches(
    base_url: str, model: str, stream: bool, verbose: bool
) -> bool:
    """Send the exact same prompt twice.  Second request should hit the
    prefix cache."""
    print("\n=== Test: Identical prompt caches ===")
    system = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [
        system,
        {"role": "user", "content": "Explain how TCP works in one paragraph."},
    ]

    r1 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r1.error:
        print(f"  FAIL: Request 1 errored — {r1.error}")
        return False
    _print_turn(1, r1, "first send")

    time.sleep(0.3)

    r2 = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
    if r2.error:
        print(f"  FAIL: Request 2 errored — {r2.error}")
        return False
    _print_turn(2, r2, "identical repeat")

    ok = True
    if r2.usage.cached_tokens == 0:
        print("  FAIL: Identical prompt on request 2 has cached_tokens=0")
        ok = False
    else:
        print(f"  OK: Identical repeat reused {r2.usage.cached_tokens} cached tokens")
    return ok


def test_shared_prefix_different_suffix(
    base_url: str, model: str, stream: bool, verbose: bool
) -> bool:
    """Two single-turn requests that share the system prompt but differ in the
    user message.  The shared prefix should be cached on the second request."""
    print("\n=== Test: Shared system prompt, different user messages ===")
    system = {"role": "system", "content": SYSTEM_PROMPT}

    r1 = send_chat(
        base_url,
        [system, {"role": "user", "content": "What is Python?"}],
        model=model,
        stream=stream,
        verbose=verbose,
    )
    if r1.error:
        print(f"  FAIL: Request 1 errored — {r1.error}")
        return False
    _print_turn(1, r1, "user: Python")

    time.sleep(0.3)

    r2 = send_chat(
        base_url,
        [system, {"role": "user", "content": "What is Rust?"}],
        model=model,
        stream=stream,
        verbose=verbose,
    )
    if r2.error:
        print(f"  FAIL: Request 2 errored — {r2.error}")
        return False
    _print_turn(2, r2, "user: Rust")

    ok = True
    if r2.usage.cached_tokens == 0:
        print("  FAIL: Shared system-prompt prefix not cached on request 2")
        ok = False
    else:
        print(
            f"  OK: Shared system-prompt prefix reused {r2.usage.cached_tokens} cached tokens"
        )
    return ok


def test_different_conversations_no_sharing(
    base_url: str, model: str, stream: bool, verbose: bool
) -> bool:
    """Two completely unrelated conversations should not share cache."""
    print("\n=== Test: Different conversations don't share cache ===")

    ra = send_chat(
        base_url,
        [
            {"role": "system", "content": "You are a marine biologist."},
            {
                "role": "user",
                "content": "Tell me about the migration patterns of blue whales.",
            },
        ],
        model=model,
        stream=stream,
        verbose=verbose,
    )
    if ra.error:
        print(f"  FAIL: Conv A errored — {ra.error}")
        return False
    _print_turn(1, ra, "conv A — marine biology")

    time.sleep(0.3)

    rb = send_chat(
        base_url,
        [
            {
                "role": "system",
                "content": "You are an astrophysicist specializing in dark matter.",
            },
            {
                "role": "user",
                "content": "What evidence supports the existence of dark matter?",
            },
        ],
        model=model,
        stream=stream,
        verbose=verbose,
    )
    if rb.error:
        print(f"  FAIL: Conv B errored — {rb.error}")
        return False
    _print_turn(2, rb, "conv B — astrophysics")

    if rb.usage.cached_tokens > 0:
        print(
            f"  WARN: Unrelated conversation got cached_tokens={rb.usage.cached_tokens} "
            "(unexpected — possible hash collision or shared BOS prefix)"
        )
    else:
        print("  OK: Unrelated conversation correctly has cached_tokens=0")
    return True


def test_five_turn_growing_cache(
    base_url: str, model: str, stream: bool, verbose: bool
) -> bool:
    """Five-turn conversation tracking how cached_tokens grows each turn."""
    print("\n=== Test: 5-turn growing conversation ===")
    system = {"role": "system", "content": SYSTEM_PROMPT}
    questions = [
        "What is a linked list?",
        "How does it differ from an array?",
        "When should I prefer a linked list?",
        "What about a skip list?",
        "Summarize the trade-offs in one sentence.",
    ]
    messages: list[dict] = [system]
    results: list[TurnResult] = []

    for i, q in enumerate(questions, 1):
        messages.append({"role": "user", "content": q})
        r = send_chat(base_url, messages, model=model, stream=stream, verbose=verbose)
        if r.error:
            print(f"  FAIL: Turn {i} errored — {r.error}")
            return False
        _print_turn(i, r)
        results.append(r)
        messages.append({"role": "assistant", "content": r.content})

    print()
    print("  Turn | prompt_tokens | cached_tokens | cache %")
    print("  -----+---------------+---------------+--------")
    for i, r in enumerate(results, 1):
        u = r.usage
        print(
            f"  {i:>4} | {u.prompt_tokens:>13} | {u.cached_tokens:>13} | {u.cache_pct():>5.1f}%"
        )

    ok = True
    for i in range(1, len(results)):
        if results[i].usage.cached_tokens == 0:
            print(f"  FAIL: Turn {i + 1} cached_tokens=0 — expected prefix reuse")
            ok = False

    if ok:
        print("  OK: All continuation turns show prefix reuse")
    return ok


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _dump_one_request(base_url: str, model: str, stream: bool) -> int:
    """Send a single request and dump the raw response for debugging."""
    print("=== Dump mode: sending one request and showing raw response ===\n")
    system = {"role": "system", "content": "You are a helpful assistant."}
    messages = [system, {"role": "user", "content": "Say hello."}]
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": 16,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    print(f"POST {base_url}/v1/chat/completions")
    print(f"  payload: {json.dumps(payload, indent=2)}\n")

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=_headers(),
        stream=stream,
        timeout=60,
    )
    print(f"HTTP {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('Content-Type', '?')}")
    print()

    if not stream:
        print("Response body:")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:2000])
        return 0

    print("SSE lines:")
    count = 0
    for line in resp.iter_lines(decode_unicode=True):
        print(f"  {line!r}")
        count += 1
        if count > 100:
            print("  … (truncated)")
            break
    print(f"\nTotal lines: {count}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E prefix cache verification for Dynamo frontend + cpp_server",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000 = cpp_server direct)",
    )
    parser.add_argument("--model", default="tt-cpp-server")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use non-streaming requests (stream=false)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every SSE chunk / response body",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Seconds to wait for server readiness"
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Send one request and dump raw response, then exit",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    stream = not args.no_stream

    print(f"Prefix cache E2E tests against {base_url}")
    print(f"  model={args.model}  stream={stream}")
    print("\nWaiting for server…")

    if not wait_for_server(base_url, args.timeout):
        print("ERROR: Server not ready within timeout")
        return 1
    print("Server ready.\n")

    # --dump: send one request and show everything the server returns.
    if args.dump:
        return _dump_one_request(base_url, args.model, stream)

    tests = [
        ("multi_turn_prefix_cache", test_multi_turn_prefix_cache),
        ("identical_prompt_caches", test_identical_prompt_caches),
        ("shared_prefix_different_suffix", test_shared_prefix_different_suffix),
        ("different_conversations_no_sharing", test_different_conversations_no_sharing),
        ("five_turn_growing_cache", test_five_turn_growing_cache),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            ok = fn(base_url, args.model, stream, args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  ERROR in {name}: {exc}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
