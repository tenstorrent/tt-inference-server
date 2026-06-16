#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Prefill/decode integration tests for the Dynamo + cpp_server DISAGGREGATED stack.

 ISL decides where prefill is expected to happen:

  - ISL <  MAX_TOKENS_TO_PREFILL_ON_DECODE -> prefilled locally on the decode
    server: the request completes, prefix-cache HIT/MISS shows up in the decode
    log, and no offload line appears in the prefill log.
  - ISL >= threshold -> offloaded: the prefill log shows
    "[InterServerService] Received prefill request: N (tokens: T)".

Offloaded requests complete on this MOCK stack (the prefill worker services its
own KV allocations). test_05 still uses a short client timeout since it only
asserts offload routing from the prefill log, not the response.

Bring the stack up first:

    benchmarks/run_stack.sh up

Then:

    pytest -v benchmarks/test_prefill_decode.py
    # or, without pytest:
    python3 benchmarks/test_prefill_decode.py

Config via env:
    TARGET       frontend base URL          (default http://127.0.0.1:8080)
    MODEL        model id from /v1/models   (default moonshotai/Kimi-K2.6)
    DECODE_URL   decode worker REST         (default http://127.0.0.1:8001)
    PREFILL_URL  prefill worker REST        (default http://127.0.0.1:8002)
    DECODE_LOG   decode server log          (default /tmp/tt_decode.log)
    PREFILL_LOG  prefill server log         (default /tmp/tt_prefill.log)
    THRESHOLD    MAX_TOKENS_TO_PREFILL_ON_DECODE the decode was started with (default 1000)
    API_KEY      bearer token, if required  (default none)
"""

import json
import os
import re
import time
import uuid
import urllib.error
import urllib.request

TARGET = os.environ.get("TARGET", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL", "moonshotai/Kimi-K2.6")
DECODE_URL = os.environ.get("DECODE_URL", "http://127.0.0.1:8001").rstrip("/")
PREFILL_URL = os.environ.get("PREFILL_URL", "http://127.0.0.1:8002").rstrip("/")
DECODE_LOG = os.environ.get("DECODE_LOG", "/tmp/tt_decode.log")
PREFILL_LOG = os.environ.get("PREFILL_LOG", "/tmp/tt_prefill.log")
RESULT_LOG = os.environ.get("RESULT_LOG", "/tmp/tt_test_results.log")
THRESHOLD = int(os.environ.get("THRESHOLD", "1000"))
API_KEY = os.environ.get("API_KEY", "")

KV_BLOCK = 32
FIRST_BLOCK = 128
# Cold-MISS TTFT >= this floor means a cache-aware mock delay is set (run_tests.sh
# sets MOCK_PREFILL_SLEEP_US_PER_TOKEN); the warm HIT must then drop to <= FRACTION of it.
TTFT_MEANINGFUL_S = float(os.environ.get("TTFT_MEANINGFUL_S", "0.15"))
TTFT_HIT_MAX_FRACTION = float(os.environ.get("TTFT_HIT_MAX_FRACTION", "0.6"))
LOG_SETTLE_S = 0.6

# test_07 large-prompt prefix cache: ~50k system + ~5k user. The ~55k prompt (>>
# THRESHOLD) offloads; the warm repeat hits the cached system prefix. Needs MAX_ISL
# raised (run_tests.sh) to accept a 55k prompt.
SYSTEM_TOKENS = int(os.environ.get("SYSTEM_TOKENS", "50000"))
USER_TOKENS = int(os.environ.get("USER_TOKENS", "5000"))
TOKENS_PER_WORD = float(os.environ.get("TOKENS_PER_WORD", "1.75"))
USER_VARIES = os.environ.get("USER_VARIES", "1") not in ("", "0", "false", "no")

_RECEIVED = re.compile(r"Received prefill request: \d+ \(tokens: (\d+)\)")
_HIT = re.compile(r"Prefix cache HIT .*matchedTokens=(\d+)")
_MISS = re.compile(r"Prefix cache MISS")


def _log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(RESULT_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango"
).split()


def _filler(n, start=0):
    return " ".join(_WORDS[(start + i) % len(_WORDS)] for i in range(n))


def _words_for(tokens):
    return max(1, round(tokens / TOKENS_PER_WORD))


def _unique():
    return "session-" + uuid.uuid4().hex + " "


def _req(method, path, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(TARGET + path, data=data, method=method)
    if payload is not None:
        r.add_header("Content-Type", "application/json")
    if API_KEY:
        r.add_header("Authorization", "Bearer " + API_KEY)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def _chat(text, max_tokens=8, timeout=120):
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
    choice = body["choices"][0]
    msg = choice["message"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    out = msg.get("content") or msg.get("reasoning_content")
    _log(
        "usage prompt=%s cached=%s completion=%s total=%s finish=%s | dynamo.usage=%s"
        % (
            u["prompt_tokens"],
            cached,
            u["completion_tokens"],
            u.get("total_tokens"),
            choice.get("finish_reason"),
            json.dumps(u),
        )
    )
    _log("  output[:120]=%r" % ((out or "")[:120],))
    return {
        "prompt_tokens": u["prompt_tokens"],
        "cached_tokens": cached,
        "completion_tokens": u["completion_tokens"],
        "total_tokens": u.get("total_tokens"),
        # reasoning models may emit only reasoning_content for short replies
        "output": out,
    }


def _chat_stream(text, max_tokens=16, timeout=120, system=None):
    # Stream a chat completion over SSE and time the first content delta (TTFT).
    # stream_options.include_usage asks the frontend for a trailing usage-only chunk.
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    r = urllib.request.Request(
        TARGET + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
    )
    r.add_header("Content-Type", "application/json")
    r.add_header("Accept", "text/event-stream")
    if API_KEY:
        r.add_header("Authorization", "Bearer " + API_KEY)
    t0 = time.monotonic()
    ttft = None
    t_first = None
    t_last = None
    chunks = 0
    pieces = []
    usage = None
    finish = None
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        assert resp.status == 200, resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            ev = json.loads(body)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    now = time.monotonic()
                    if ttft is None:
                        ttft = now - t0
                        t_first = now
                    t_last = now
                    pieces.append(piece)
                    chunks += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total_s = time.monotonic() - t0
    out = "".join(pieces)
    u = usage or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    # Decode TPS: tokens after the first over the first->last delta span.
    # completion_tokens is authoritative; fall back to chunk count if usage is absent.
    gen = u.get("completion_tokens") or chunks
    decode_span = (t_last - t_first) if (t_first and t_last) else 0.0
    tps = ((gen - 1) / decode_span) if (gen and gen > 1 and decode_span > 0) else None
    _log(
        "stream ttft=%s total=%.3fs tps=%s chunks=%s finish=%s prompt=%s cached=%s completion=%s"
        % (
            ("%.3fs" % ttft) if ttft is not None else None,
            total_s,
            ("%.1f" % tps) if tps is not None else None,
            chunks,
            finish,
            u.get("prompt_tokens"),
            cached,
            u.get("completion_tokens"),
        )
    )
    _log("  output[:120]=%r" % (out[:120],))
    return {
        "ttft": ttft,
        "total_s": total_s,
        "tps": tps,
        "chunks": chunks,
        "finish": finish,
        "output": out,
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": u.get("completion_tokens"),
        "usage": usage,
    }


def _fire(text, max_tokens=4, timeout=10):
    # POST a completion; return the usage dict if it completes, or None if the
    # client times out (an offloaded request never completes on the mock stack).
    try:
        return _chat(text, max_tokens=max_tokens, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        _log(
            "no completion (client timeout/abort, expected for offloaded reqs) — %r" % e
        )
        return None


def _offset(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _since(path, offset):
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# Assert a HIT or MISS in the decode log after `offset`. For HITs, return the
# matchedTokens count — the server-side measure of prefix reuse, which the usage
# block's cached_tokens now mirrors.
def _expect_cache_event(offset, kind):
    time.sleep(LOG_SETTLE_S)
    tail = _since(DECODE_LOG, offset)
    if kind == "HIT":
        m = _HIT.search(tail)
        _log(
            "cache event expected=HIT found=%s matchedTokens=%s (decode log %s)"
            % (bool(m), m.group(1) if m else None, DECODE_LOG)
        )
        assert m, "expected 'Prefix cache HIT' in %s; recent log:\n%s" % (
            DECODE_LOG,
            tail[-1500:],
        )
        return int(m.group(1))
    found = bool(_MISS.search(tail))
    _log("cache event expected=MISS found=%s (decode log %s)" % (found, DECODE_LOG))
    assert found, "expected 'Prefix cache MISS' in %s; recent log:\n%s" % (
        DECODE_LOG,
        tail[-1500:],
    )
    return None


def _expect_no_offload(offset, isl):
    time.sleep(LOG_SETTLE_S)
    m = _RECEIVED.search(_since(PREFILL_LOG, offset))
    assert not m, (
        "ISL %s tok (< threshold %d) was offloaded to prefill (tokens=%s) but should "
        "stay local on decode" % (isl, THRESHOLD, m.group(1))
    )
    _log("no offload in prefill log — prefilled locally on decode (isl=%s)" % isl)


def _received_since(offset, deadline_s=12):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        m = _RECEIVED.search(_since(PREFILL_LOG, offset))
        if m:
            _log(
                "prefill server received offload: tokens=%s (%s)"
                % (m.group(1), PREFILL_LOG)
            )
            return int(m.group(1))
        time.sleep(0.5)
    _log("no offload seen in prefill log within %ss" % deadline_s)
    return None


def _worker_health(base, timeout=5):
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except (urllib.error.URLError, OSError):
        return None, None


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
    raise AssertionError("server not healthy within %ds (%s)" % (timeout_s, last))


def _ensure_disaggregated():
    assert os.path.exists(PREFILL_LOG), (
        "prefill log %s missing — start the disaggregated stack: run_stack.sh up"
        % PREFILL_LOG
    )


# 1. Decode + prefill healthy, inter-server socket up, frontend round-trip works.
def test_01_health_and_frontend():
    _log(
        "--------- Test01 Health: decode + prefill /health + frontend round-trip --------"
    )
    _ensure_server()
    _ensure_disaggregated()
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
    _log(
        "prefill healthy at %s socket_status=%s"
        % (PREFILL_URL, body.get("socket_status"))
    )
    r = _chat("Hello, are you healthy?", max_tokens=16)
    assert r["prompt_tokens"] > 0 and r["completion_tokens"] > 0 and r["output"], r


# 2. Small ISL: fresh ~200-tok prompt is prefilled locally on decode —
#    completes, cold cache MISS, and no offload line in the prefill log.
def test_02_small_isl_local_cold_miss():
    _log(
        "--------- Test02 Small ISL < %d: local prefill on decode, cold MISS, no offload --------"
        % THRESHOLD
    )
    _ensure_server()
    _ensure_disaggregated()
    d_off, p_off = _offset(DECODE_LOG), _offset(PREFILL_LOG)
    r = _chat(_unique() + _filler(110), max_tokens=8)
    assert 120 <= r["prompt_tokens"] <= 450, r
    assert r["prompt_tokens"] < THRESHOLD, r
    assert r["cached_tokens"] == 0, r
    _expect_cache_event(d_off, "MISS")
    _expect_no_offload(p_off, r["prompt_tokens"])


# 3. ISL grows slowly with a STABLE prefix, all turns < threshold -> every turn is
#    prefilled locally on decode (no offload), turn 2+ hit the prefix cache, and the
#    cached span grows as the shared prefix grows. Streamed for per-turn TTFT/TPS:
#    turn 0 cold-prefills the whole base (high TTFT); turns 1+ hit the cache and
#    prefill only the new delta (TTFT drops, asserted when a delay is configured).
def test_03_prefix_cache_slow_growth():
    _log(
        "--------- Test03 Prefix cache slow growth: stable prefix, all local, cached span grows ---------"
    )
    _ensure_server()
    _ensure_disaggregated()
    base = _unique() + _filler(400)
    grow = [_filler(20, 100), _filler(20, 130), _filler(20, 160)]
    p_off = _offset(PREFILL_LOG)
    results = []
    for i in range(len(grow) + 1):
        prompt = base + ((" " + " ".join(grow[:i])) if i else "")
        _log("turn %d/%d" % (i, len(grow)))
        d_off = _offset(DECODE_LOG)
        r = _chat_stream(prompt, max_tokens=24)
        results.append(r)
        assert r["prompt_tokens"] is not None and r["prompt_tokens"] < THRESHOLD, r
        assert r["ttft"] is not None and r["chunks"] >= 1, ("no stream", r)
        if i == 0:
            assert r["cached_tokens"] == 0, ("turn 0 should be cold", r)
            _expect_cache_event(d_off, "MISS")
        else:
            matched = _expect_cache_event(d_off, "HIT")
            assert r["cached_tokens"] > 0, (
                "turn %d: decode log matchedTokens=%s but API cached_tokens=0 — "
                "usage should reflect local prefix-cache reuse" % (i, matched),
                r,
            )
    cached = [r["cached_tokens"] for r in results]
    assert all(cached[i] >= cached[i - 1] for i in range(1, len(cached))), cached
    for i in range(1, len(results)):
        prev_pt = results[i - 1]["prompt_tokens"]
        assert results[i]["cached_tokens"] >= prev_pt - FIRST_BLOCK, (cached, prev_pt)
    _expect_no_offload(p_off, results[-1]["prompt_tokens"])

    ttfts = [r["ttft"] for r in results]
    _log(
        "per-turn TTFT=%s TPS=%s cached=%s"
        % (
            ["%.3fs" % t for t in ttfts],
            [("%.1f" % r["tps"]) if r["tps"] else None for r in results],
            cached,
        )
    )
    # Decode TPS should be positive and ~stable across turns (decode work is constant).
    for i, r in enumerate(results):
        if r["tps"] is not None:
            assert r["tps"] > 0, ("turn %d non-positive TPS" % i, r)
    # With a cache-aware delay, every warm turn must drop TTFT well below turn 0.
    if ttfts[0] >= TTFT_MEANINGFUL_S:
        for i in range(1, len(ttfts)):
            assert ttfts[i] <= ttfts[0] * TTFT_HIT_MAX_FRACTION, (
                "turn %d warm TTFT %.3fs not <= %.0f%% of cold turn-0 %.3fs"
                % (i, ttfts[i], TTFT_HIT_MAX_FRACTION * 100, ttfts[0]),
                ttfts,
            )


# 4. Shared prefix below threshold: ~400 tok, then ~800 tok sharing the first ~400.
#    The shared prefix is a cache HIT and the tail is still prefilled locally on
#    decode — no offload.The >=1k variant would offload and cannot complete on
#    the mock stack; covered on hardware.
def test_04_shared_prefix_stays_local():
    _log(
        "--------- Test04 Shared prefix < %d: ~400 then ~800 tok, HIT, all local ---------"
        % THRESHOLD
    )
    _ensure_server()
    _ensure_disaggregated()
    p_off = _offset(PREFILL_LOG)
    base = _unique() + _filler(220)
    _log("first request (~400-tok prefix)")
    a = _chat(base, max_tokens=8)
    assert 300 <= a["prompt_tokens"] < THRESHOLD, a
    _log("second request (~800 tok, shares first ~400)")
    big = base + " " + _filler(220, 7)
    d_off = _offset(DECODE_LOG)
    b = _chat(big, max_tokens=8)
    assert b["prompt_tokens"] < THRESHOLD, b
    hit = _expect_cache_event(d_off, "HIT")
    assert b["prompt_tokens"] >= 1.5 * a["prompt_tokens"], (a, b)
    assert b["cached_tokens"] >= a["prompt_tokens"] - 2 * FIRST_BLOCK, (
        "decode log matchedTokens=%s but API cached_tokens=%s — usage should "
        "reflect local prefix-cache reuse" % (hit, b["cached_tokens"]),
        a,
        b,
    )
    assert b["cached_tokens"] < b["prompt_tokens"], (a, b)
    _expect_no_offload(p_off, b["prompt_tokens"])


# 5. Large ISL (>= threshold) is offloaded to the prefill server.
def test_05_large_isl_offloads_to_prefill():
    _log(
        "--------- Test05 Large ISL >= %d: offloaded to prefill server ---------"
        % THRESHOLD
    )
    _ensure_server()
    _ensure_disaggregated()
    p_off = _offset(PREFILL_LOG)
    # ~1400 tok (>= threshold) offloads to prefill. It completes on the mock; we only
    # assert offload routing from the prefill log, hence the short timeout.
    _fire(_unique() + _filler(750), timeout=20)
    offloaded = _received_since(p_off)
    assert offloaded is not None, "large ISL was NOT offloaded to the prefill server"
    assert offloaded >= THRESHOLD, "offloaded token count %d < threshold %d" % (
        offloaded,
        THRESHOLD,
    )


# 6. Streaming TTFT: same sub-threshold prefix streamed cold (MISS) then warm (HIT).
#    Hard asserts cover streaming correctness (first delta timed, token-by-token,
#    usage present) and the decode-log cache event. With a cache-aware delay
#    (MOCK_PREFILL_SLEEP_US_PER_TOKEN, set by run_tests.sh) the warm HIT skips the
#    shared prefix so TTFT drops sharply — asserted; otherwise a loose upper bound.
def test_06_streaming_ttft_hit_vs_miss():
    _log(
        "--------- Test06 Streaming TTFT: cold MISS then warm HIT on a shared prefix ---------"
    )
    _ensure_server()
    _ensure_disaggregated()
    prompt = _unique() + _filler(400)

    d_off = _offset(DECODE_LOG)
    cold = _chat_stream(prompt, max_tokens=16)
    assert cold["ttft"] is not None, ("no content delta streamed", cold)
    assert cold["chunks"] >= 2, ("expected token-by-token stream, not one blob", cold)
    assert cold["output"], cold
    assert cold["finish"], ("stream ended without a finish_reason", cold)
    if cold["prompt_tokens"] is not None:
        assert cold["prompt_tokens"] < THRESHOLD, cold
        assert cold["cached_tokens"] == 0, ("cold stream should be a cold cache", cold)
    _expect_cache_event(d_off, "MISS")

    d_off = _offset(DECODE_LOG)
    warm = _chat_stream(prompt, max_tokens=16)
    assert warm["ttft"] is not None, ("no content delta streamed", warm)
    assert warm["chunks"] >= 2, warm
    assert warm["output"], warm
    matched = _expect_cache_event(d_off, "HIT")
    if warm["prompt_tokens"] is not None:
        assert warm["cached_tokens"] > 0, (
            "streamed warm HIT: decode log matchedTokens=%s but API cached_tokens=0 — "
            "usage should reflect local prefix-cache reuse" % matched,
            warm,
        )
    speedup = cold["ttft"] / warm["ttft"] if warm["ttft"] else float("inf")
    _log(
        "TTFT cold(MISS)=%.3fs warm(HIT)=%.3fs speedup=%.2fx matchedTokens=%s"
        % (cold["ttft"], warm["ttft"], speedup, matched)
    )
    if cold["ttft"] >= TTFT_MEANINGFUL_S:
        # Warm HIT reprefills only the uncached tail, so TTFT must drop below cold MISS.
        assert warm["ttft"] <= cold["ttft"] * TTFT_HIT_MAX_FRACTION, (
            "warm-HIT TTFT %.3fs not <= %.0f%% of cold-MISS %.3fs — prefix cache did "
            "not save prefill work"
            % (warm["ttft"], TTFT_HIT_MAX_FRACTION * 100, cold["ttft"]),
            cold,
            warm,
        )
    else:
        # No delay configured: plain mock does no real prefill work, so just sanity-bound.
        assert warm["ttft"] <= cold["ttft"] * 3 + 0.5, (
            "warm-HIT TTFT unexpectedly high vs cold-MISS",
            cold,
            warm,
        )


# 7. Large-prompt prefix-cache TTFT (disaggregation): ~50k system + ~5k user. The
#    ~55k prompt (>> THRESHOLD) offloads; the warm repeat (different user) hits the
#    cached ~50k system prefix, prefilling only ~5k, so TTFT drops. Completes on the
#    mock. Needs MAX_ISL raised to accept 55k (run_tests.sh).
def test_07_large_prompt_prefix_cache_ttft():
    _log(
        "--------- Test07 Large-prompt prefix-cache TTFT: %d-tok system + %d-tok user, "
        "offloaded, cold vs warm ---------" % (SYSTEM_TOKENS, USER_TOKENS)
    )
    _ensure_server()
    _ensure_disaggregated()
    system = (
        _unique()
        + "You are a helpful assistant.\n"
        + _filler(_words_for(SYSTEM_TOKENS))
    )
    user1 = "Please answer question A.\n" + _filler(_words_for(USER_TOKENS), 7)
    user2 = (
        ("Please answer question B.\n" + _filler(_words_for(USER_TOKENS), 137))
        if USER_VARIES
        else user1
    )

    p_off = _offset(PREFILL_LOG)
    _log("request 1 (cold)")
    cold = _chat_stream(user1, system=system, max_tokens=16, timeout=120)
    _log(
        "request 2 (warm, %s user message)"
        % ("different" if USER_VARIES else "identical")
    )
    warm = _chat_stream(user2, system=system, max_tokens=16, timeout=120)

    assert cold["ttft"] is not None and warm["ttft"] is not None, (cold, warm)
    assert cold["prompt_tokens"] and cold["prompt_tokens"] >= 0.5 * (
        SYSTEM_TOKENS + USER_TOKENS
    ), (
        "prompt tokenized far below the ~%d-tok target" % (SYSTEM_TOKENS + USER_TOKENS),
        cold,
    )
    # Both requests (>> THRESHOLD) offloaded to the prefill server.
    offloaded = [int(t) for t in _RECEIVED.findall(_since(PREFILL_LOG, p_off))]
    big = [t for t in offloaded if t >= THRESHOLD]
    assert len(big) >= 2, (
        "expected both ~55k requests offloaded to the prefill server (>=2 'Received "
        "prefill request' with tokens>=%d); saw %s" % (THRESHOLD, offloaded)
    )
    # Cold is a cold cache; warm hits the shared ~50k system prefix.
    assert cold["cached_tokens"] == 0, ("first request should be a cold cache", cold)
    assert warm["cached_tokens"] >= 0.5 * SYSTEM_TOKENS, (
        "warm request should hit the cached system prefix; cached_tokens=%s"
        % warm["cached_tokens"],
        warm,
    )
    if USER_VARIES:
        assert warm["cached_tokens"] < warm["prompt_tokens"], (
            "warm cached_tokens should not cover the whole prompt when the user differs",
            warm,
        )
    for r in (cold, warm):
        if r["tps"] is not None:
            assert r["tps"] > 0, ("non-positive decode TPS", r)
    speedup = cold["ttft"] / warm["ttft"] if warm["ttft"] else float("inf")
    _log(
        "TTFT cold=%.3fs warm=%.3fs speedup=%.2fx | TPS cold=%s warm=%s (cached cold=%s warm=%s)"
        % (
            cold["ttft"],
            warm["ttft"],
            speedup,
            ("%.1f" % cold["tps"]) if cold["tps"] else None,
            ("%.1f" % warm["tps"]) if warm["tps"] else None,
            cold["cached_tokens"],
            warm["cached_tokens"],
        )
    )
    # With a cache-aware delay, the warm HIT (only ~5k new) must drop TTFT below cold.
    if cold["ttft"] >= TTFT_MEANINGFUL_S:
        assert warm["ttft"] <= cold["ttft"] * TTFT_HIT_MAX_FRACTION, (
            "warm TTFT %.3fs not <= %.0f%% of cold %.3fs — prefix cache did not save "
            "prefill work" % (warm["ttft"], TTFT_HIT_MAX_FRACTION * 100, cold["ttft"]),
            cold,
            warm,
        )


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  " + t.__name__)
        except Exception as e:
            failed += 1
            print("FAIL  " + t.__name__ + ": " + repr(e))
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
