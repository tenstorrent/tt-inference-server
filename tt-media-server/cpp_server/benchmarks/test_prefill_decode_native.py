#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""No-Dynamo smoke test: drives /v1/chat/completions at the decode worker's
native REST port (DECODE_URL, default :8001).
Both ingresses feed the same LLMPipeline, so the disaggregation/prefix-cache
behavior is covered by test_prefill_decode.py; this just proves the native
ingress reaches the pipeline and round-trips.

Two native-only constraints: the server enforces Bearer auth (OPENAI_API_KEY,
default "your-secret-key"), and it tokenizes server-side, so the model must ship
a tokenizer.json.

    MODEL=deepseek-ai/DeepSeek-R1-0528 benchmarks/run_stack.sh up
    MODEL=deepseek-ai/DeepSeek-R1-0528 pytest -v benchmarks/test_prefill_decode_native.py
"""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_prefill_decode import PREFILL_LOG, _log, _worker_health  # noqa: E402

DECODE_URL = os.environ.get("DECODE_URL", "http://127.0.0.1:8001").rstrip("/")
PREFILL_URL = os.environ.get("PREFILL_URL", "http://127.0.0.1:8002").rstrip("/")
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-0528")
API_KEY = os.environ.get("API_KEY", "your-secret-key")


def _req(method, path, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(DECODE_URL + path, data=data, method=method)
    if payload is not None:
        r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + API_KEY)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def _chat(text, max_tokens=16, timeout=120):
    status, body = _req(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    assert status == 200, "HTTP %s: %s" % (status, body)
    u = body["usage"]
    msg = body["choices"][0]["message"]
    # native LLMController emits reasoning text under "reasoning"; the frontend
    # normalizes it to "content". Accept all three so the check is ingress-agnostic.
    out = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content")
    return {
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "output": out,
    }


def _ensure_server(timeout_s=30):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            status, body = _req("GET", "/health", timeout=5)
            if status == 200 and body.get("status") == "healthy":
                return
            last = "/health=%s" % body
        except Exception as e:
            last = repr(e)
        time.sleep(1)
    raise AssertionError(
        "decode worker not healthy at %s within %ds (%s)"
        % (DECODE_URL, timeout_s, last)
    )


# Decode + prefill healthy, socket up, chat round-trips on :8001
def test_native_roundtrip_no_dynamo():
    _log(
        "--------- Native smoke: decode+prefill health + direct :8001 round-trip (no Dynamo) ---------"
    )
    _ensure_server()
    assert os.path.exists(PREFILL_LOG), (
        "prefill log %s missing — start the disaggregated stack: MODEL=%s run_stack.sh up"
        % (PREFILL_LOG, MODEL)
    )
    status, body = _worker_health(DECODE_URL)
    assert status == 200 and body.get("status") == "healthy", (
        "decode worker unhealthy at %s: HTTP %s %s" % (DECODE_URL, status, body)
    )
    assert "socket_status" in body, (
        "decode has no inter-server socket — stack is not disaggregated: %s" % body
    )
    _log(
        "decode healthy at %s socket_status=%s"
        % (DECODE_URL, body.get("socket_status"))
    )
    status, body = _worker_health(PREFILL_URL)
    assert status == 200 and body.get("status") == "healthy", (
        "prefill worker unhealthy at %s: HTTP %s %s" % (PREFILL_URL, status, body)
    )
    r = _chat("Hello, are you healthy?")
    _log(
        "native round-trip: prompt=%s completion=%s output=%r"
        % (r["prompt_tokens"], r["completion_tokens"], (r["output"] or "")[:80])
    )
    assert r["prompt_tokens"] > 0 and r["completion_tokens"] > 0 and r["output"], r


if __name__ == "__main__":
    import traceback

    try:
        test_native_roundtrip_no_dynamo()
        print("PASS  test_native_roundtrip_no_dynamo")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
