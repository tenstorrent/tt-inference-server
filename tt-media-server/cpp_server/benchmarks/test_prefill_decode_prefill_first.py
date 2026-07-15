#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Prefill-first + Dynamo-native routing integration tests.

Bring the stack up with:

    DYNAMO_NATIVE_ROUTING=1 USE_PREFILL_FIRST_DISAGGREGATION=1 \\
        MODEL=deepseek-ai/DeepSeek-R1-0528 benchmarks/run_stack.sh up

Or via the harness default:

    benchmarks/run_tests.sh

Asserts the orchestration path:
  1. Dynamo hits prefill first (Native prefill-first log)
  2. Prefill reserves a decode slot over ZMQ
  3. Decode grants the slot
  4. Chat completes with distinct Dynamo worker ids and decode max_tokens budget
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_prefill_decode import (  # noqa: E402
    DECODE_LOG,
    PREFILL_LOG,
    TARGET,
    _ensure_server,
    _log,
    _offset,
    _since,
    _worker_health,
)

DECODE_URL = os.environ.get("DECODE_URL", "http://127.0.0.1:8001").rstrip("/")
PREFILL_URL = os.environ.get("PREFILL_URL", "http://127.0.0.1:8002").rstrip("/")
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-0528")
API_KEY = os.environ.get("API_KEY", "")
LOG_SETTLE_S = 0.8
PREFILL_FIRST = os.environ.get("USE_PREFILL_FIRST_DISAGGREGATION", "0") == "1"

_NATIVE_PREFILL_FIRST = re.compile(
    r"\[DynamoEndpoint\] Native prefill-first path taskId=(\d+) tokens=(\d+)"
)
_SLOT_RESERVE = re.compile(
    r"\[DisaggregationService\] Prefill-first slot reservation taskId=(\d+)"
)
_SLOT_GRANTED = re.compile(
    r"\[DisaggregationService\] Slot reservation granted taskId=(\d+) slotId=(\d+)"
)
_SLOT_REQUEST_DECODE = re.compile(
    r"\[DisaggregationService\] Slot reservation request taskId=(\d+)"
)
_DECODE_MAX_TOKENS = re.compile(
    r"\[DynamoEndpoint\] Using decode max_tokens=(\d+) \(prefill remaining_tokens="
)


def _req(method, path, payload=None, timeout=120, base=TARGET):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(base + path, data=data, method=method)
    if payload is not None:
        r.add_header("Content-Type", "application/json")
    if API_KEY:
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
    return body


def _wait_log(path, offset, pattern, deadline_s=15):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        m = pattern.search(_since(path, offset))
        if m:
            return m
        time.sleep(0.2)
    return None


def test_prefill_first_orchestration_via_dynamo():
    mode = "prefill-first" if PREFILL_FIRST else "native"
    _log("--------- Dynamo native route (%s) via :8080 ---------" % mode)
    _ensure_server()

    assert os.path.exists(PREFILL_LOG) and os.path.exists(DECODE_LOG), (
        "missing worker logs; start with "
        "DYNAMO_NATIVE_ROUTING=1 [USE_PREFILL_FIRST_DISAGGREGATION=1] run_stack.sh up"
    )

    status, body = _worker_health(DECODE_URL)
    assert status == 200 and body.get("status") == "healthy", body
    if PREFILL_FIRST:
        assert "socket_status" in body, (
            "decode missing inter-server socket (required for prefill-first): %s" % body
        )

    status, body = _worker_health(PREFILL_URL)
    assert status == 200 and body.get("status") == "healthy", body

    pre_off = _offset(PREFILL_LOG)
    dec_off = _offset(DECODE_LOG)

    max_tokens = 16
    resp = _chat("Say hello in one short sentence.", max_tokens=max_tokens)
    usage = resp["usage"]
    nvext = resp.get("nvext") or {}
    workers = nvext.get("worker_id") or {}
    timing = nvext.get("timing") or {}

    _log(
        "chat ok: completion=%s prompt=%s finish=%s ttft_ms=%s "
        "prefill_worker=%s decode_worker=%s"
        % (
            usage.get("completion_tokens"),
            usage.get("prompt_tokens"),
            resp["choices"][0].get("finish_reason"),
            timing.get("ttft_ms"),
            workers.get("prefill_worker_id"),
            workers.get("decode_worker_id"),
        )
    )

    time.sleep(LOG_SETTLE_S)
    pre_tail = _since(PREFILL_LOG, pre_off)
    dec_tail = _since(DECODE_LOG, dec_off)

    if PREFILL_FIRST:
        m_pf = _NATIVE_PREFILL_FIRST.search(pre_tail) or _wait_log(
            PREFILL_LOG, pre_off, _NATIVE_PREFILL_FIRST
        )
        assert m_pf, (
            "expected Native prefill-first path in prefill log; recent:\n%s"
            % pre_tail[-2000:]
        )
        _log(
            "prefill-first path taskId=%s tokens=%s" % (m_pf.group(1), m_pf.group(2))
        )

        assert _SLOT_RESERVE.search(pre_tail) or _wait_log(
            PREFILL_LOG, pre_off, _SLOT_RESERVE
        ), "expected Prefill-first slot reservation in prefill log"

        assert _SLOT_REQUEST_DECODE.search(dec_tail) or _wait_log(
            DECODE_LOG, dec_off, _SLOT_REQUEST_DECODE
        ), "expected Slot reservation request in decode log"

        m_grant = _SLOT_GRANTED.search(pre_tail) or _wait_log(
            PREFILL_LOG, pre_off, _SLOT_GRANTED
        )
        assert m_grant, "expected Slot reservation granted in prefill log"
        _log(
            "slot granted taskId=%s slotId=%s" % (m_grant.group(1), m_grant.group(2))
        )

        # Decode hop should honor client max_tokens, not prefill's forced 1.
        m_budget = _DECODE_MAX_TOKENS.search(dec_tail)
        if m_budget:
            assert int(m_budget.group(1)) == max_tokens, m_budget.group(0)
            _log("decode budget override max_tokens=%s" % m_budget.group(1))
        assert usage.get("completion_tokens", 0) > 1, (
            "expected >1 completion tokens with max_tokens=%s (got %s); "
            "decode likely still stuck on remaining_tokens=1"
            % (max_tokens, usage.get("completion_tokens"))
        )

    prefill_id = workers.get("prefill_worker_id")
    decode_id = workers.get("decode_worker_id")
    assert prefill_id is not None and decode_id is not None, nvext
    assert prefill_id != decode_id, (
        "expected distinct Dynamo workers for remote prefill; got %s" % workers
    )
    assert timing.get("prefill_time_ms", 0) > 0, timing


if __name__ == "__main__":
    import traceback

    try:
        test_prefill_first_orchestration_via_dynamo()
        print("PASS  test_prefill_first_orchestration_via_dynamo")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
