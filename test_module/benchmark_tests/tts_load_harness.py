# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Standalone TTS load harness for a running server's ``/v1/audio/speech``.

Not part of the workflow engine: ``benchmark_tests/__init__.py`` exposes only the
``run_*_benchmark`` entry points, so nothing here is imported by ``run.py``. Run
it by hand against any deployed TTS server.

    sweep    concurrency sweep, one of three arrival models -> results JSON
    report   that JSON -> summary table, FC/SC/TC box table, optional charts
    dump     one concurrency, per-chunk detail: occupancy, bubbles, per-user table
    ladder   dump repeated across a concurrency ladder, with server-side spans

    python tts_load_harness.py sweep --host d08u08 --concurrency 1,2,4,8,16 \\
        --duration 60 --warmup 20 --text-tokens 1024 --out results.json
    python tts_load_harness.py report --results results.json --target all \\
        --chart curve.png
    python tts_load_harness.py dump --users 64 --duration 90 --skip 25 --outdir dump_u64
    python tts_load_harness.py ladder --users 16,32,64,128 --container tt-cpp-worker

Requires Python 3.10+. Only matplotlib is non-stdlib, and only for ``--chart`` --
every table works without it.

ARRIVAL MODELS -- they answer different questions; say which one produced a number.
  closed  fixed worker pool, each worker sends its next request on completion. Self
          throttling: cannot overload the server. Concurrency is the input, request
          rate the output. Answers "with N live sessions, what do they experience".
  open    Poisson arrivals, as tts_sim models them:
              lambda = target_concurrency / mean_service_time   (simulation.py:92)
              inter_arrival ~ Exponential(lambda)               (simulation.py:266)
          Does NOT self-throttle: past capacity the backlog grows without bound and
          latency climbs with elapsed time rather than settling. Answers "what rate
          can we serve at target latency". ``--rate`` pins lambda instead.
  burst   every request at a level released at the same instant -- a thundering herd.
          Worst-case simultaneous prefill demand. One shot per level, no steady state.

MEASUREMENT NOTES
  * TTFB/FC is send -> first AUDIO chunk. The server emits the 44-byte WAV header
    BEFORE synthesis starts, so it is consumed and timed separately (``hdr_s``) and
    never counted as audio. Timing the first body byte instead measures the HTTP
    round trip (~2 ms) and is wrong.
  * Chunk framing is parsed (transfer-encoding: chunked), never byte thresholds. The
    server emits one HTTP chunk per audio chunk, so SC/TC live in the chunk
    boundaries; http.client dechunks transparently and destroys them, hence the raw
    socket. A byte-threshold mark silently measures the wrong chunk when header size
    or token count drifts, and blocks across chunk boundaries inside recv().
  * The response must be chunked. Sending "Connection: close" makes drogon delimit
    the body by closing the socket instead, after which chunk boundaries (and so
    SC/TC) are unrecoverable; requests are sent keep-alive and an unchunked response
    is reported loudly rather than as silently empty audio.
  * Latency is measured on a monotonic clock; occupancy integrals use wall clock,
    because they must line up across threads and with server logs.
  * Requests run to COMPLETION unless ``--max-chunks`` caps them. Do not cap the
    output if the figure is meant to be characters/hour: a truncated request did not
    process the characters you are claiming credit for.
  * Thin levels lie. Fewer than ~20 completions makes p90/p99 a single outlier; the
    box table says so. Cold IO threads pay a ~1.3 s tokenizer cache miss, which at
    low arrival rates never warms out.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import socket
import statistics as st
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_PATH = "/v1/audio/speech"
DEFAULT_KEY = "your-secret-key"
SAMPLE_RATE = 48000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44
MAX_TC_GAPS = 200  # cap the per-request gap list so a sweep's JSON stays manageable

# Tokens per audio chunk must match the server's TTS_CHUNK_TOKENS: the underrun
# threshold and the TC deadline both derive from the audio one chunk carries.
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "30"))
CHUNK_AUDIO_S = CHUNK_TOKENS * 960 / SAMPLE_RATE
CHUNK_AUDIO_MS = CHUNK_AUDIO_S * 1000.0
# SC deadline is the 0.3 s playback buffer; TC deadline is the audio a chunk
# carries, less the 30 ms safety margin (tts_sim/config.py tc_deadline_ms).
FC_P50_MS, FC_P95_MS, SC_DEADLINE_MS = 150.0, 350.0, 270.0
TC_DEADLINE_MS = CHUNK_AUDIO_MS - 30.0

TARGETS = {
    # key: (label, axis name, per-request field, p50 target ms, p90 target ms)
    "ttfb": ("TTFB", "TTFB — time to first audio chunk (ms)", "ttfb_s", 150.0, 350.0),
    "ttfs": ("TTFS", "TTFS — first->second chunk gap (ms)", "sc_s", 270.0, 270.0),
    "ttft": (
        "TTFT",
        "TTFT — steady-state inter-chunk gap (ms)",
        "tc_gaps",
        570.0,
        570.0,
    ),
}


@dataclass
class Target:
    """Everything one request needs: where to send it and what to send."""

    host: str = "localhost"
    port: int = 8000
    path: str = DEFAULT_PATH
    key: str = DEFAULT_KEY
    timeout: float = 300.0
    text: str = ""
    # OSL cap in audio chunks; 0 = uncapped. 34 -> OSL 1020 output tokens, keeping
    # ISL+OSL inside CACHE_MAX_SEQ_LEN=2048. The client closes the socket at the cap.
    max_chunks: int = 0
    request: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        body = json.dumps({"text": self.text}).encode()
        self.request = (
            b"POST " + self.path.encode() + b" HTTP/1.1\r\n"
            b"Host: " + self.host.encode() + b"\r\n"
            b"Authorization: Bearer " + self.key.encode() + b"\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: keep-alive\r\n\r\n" + body
        )


def build_text(nchars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    return (base * (nchars // len(base) + 1))[:nchars]


# --------------------------------------------------------------------------- client


def request(tgt: Target, uid: int = -1, rec: dict | None = None) -> dict:
    """Issue one request over a raw socket, timing every audio chunk. Never raises.

    A connect or protocol error must not make the request vanish: it still held a
    server slot for real time, so dropping it biases the occupancy integral low and
    hides the failure. Everything is recorded here and classified downstream.

    Times are seconds since send, on a monotonic clock:
      admit_s  response headers arrived        (admission proof)
      hdr_s    44-byte WAV header consumed     (never counted as audio)
      ttfb_s   first AUDIO chunk               (FC -- the listener's wait)
      sc_s     chunk2 - chunk1                 (SC)
      gaps     chunk[k+1] - chunk[k], k >= 2   (TC, steady-state gaps)
    ``t_send``/``t_end`` are wall clock, for occupancy integrals.
    """
    # A caller may register the record BEFORE the send (see run_closed/dump), so that
    # a request still in flight when the join deadline expires is still in the record
    # set. In that case t_send is already stamped and must not be reset here.
    if rec is None:
        rec = {"uid": uid, "t_send": time.time()}
    rec.update({
        "t_end": None,
        "admit_s": None,
        "hdr_s": None,
        "ttfb_s": None,
        "sc_s": None,
        "gaps": [],
        "marks": [],
        "nchunks": 0,
        "gen_s": None,
        "bytes": 0,
        "audio_s": 0.0,
        "chars": len(tgt.text),
        "ok": False,
        "capped": False,
        "status": None,
        "error": None,
    })
    t0 = time.perf_counter()
    sock = None
    marks: list[float] = rec["marks"]
    audio = {"bytes": 0, "hdr_left": WAV_HEADER_BYTES}
    try:
        sock = socket.create_connection((tgt.host, tgt.port), timeout=tgt.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(tgt.request)
        buf = b""

        def fill() -> bytes:
            chunk = sock.recv(65536)
            if not chunk:
                raise EOFError("connection closed")
            return chunk

        while b"\r\n\r\n" not in buf:
            buf += fill()
        rec["admit_s"] = time.perf_counter() - t0
        head, buf = buf.split(b"\r\n\r\n", 1)
        rec["status"] = int(head.split(b"\r\n", 1)[0].split()[1])

        if rec["status"] != 200:
            # Record the server's OWN explanation, and do it BEFORE the chunked test:
            # error responses are never chunked, so that test would otherwise report a
            # misleading parse error and lose the one piece of evidence that says
            # WHICH admission limit rejected this request.
            err_body = buf
            match = re.search(rb"content-length:\s*(\d+)", head, re.I)
            if match:
                want = min(int(match.group(1)), 2048)
                try:
                    while len(err_body) < want:
                        err_body += fill()
                except Exception:
                    pass
            rec["error"] = (
                f"http {rec['status']}: "
                f"{err_body[:512].decode('utf-8', 'replace').strip()}"
            )
            return rec
        if b"chunked" not in head.lower():
            rec["error"] = (
                "response not chunked -- cannot recover chunk boundaries (SC/TC). "
                "Check that no Connection: close is sent."
            )
            return rec

        def note(payload: bytes) -> None:
            """Account one body payload; the first WAV_HEADER_BYTES are not audio."""
            n = len(payload)
            if audio["hdr_left"]:
                take = min(audio["hdr_left"], n)
                audio["hdr_left"] -= take
                n -= take
                if rec["hdr_s"] is None and audio["hdr_left"] == 0:
                    rec["hdr_s"] = time.perf_counter() - t0
            if n > 0:
                audio["bytes"] += n
                marks.append(time.perf_counter() - t0)

        while True:
            while b"\r\n" not in buf:
                buf += fill()
            line, buf = buf.split(b"\r\n", 1)
            size = int(line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                break
            while len(buf) < size + 2:
                buf += fill()
            note(buf[:size])
            buf = buf[size + 2 :]
            if tgt.max_chunks and len(marks) >= tgt.max_chunks:
                rec["capped"] = True
                break

        # gen_s covers the WHOLE request, terminating chunk included: that is the time
        # the slot was held. ``marks[-1]`` is where the last audio landed, which is
        # what W should be when the tail is accounted separately (see ``dump``).
        rec["gen_s"] = time.perf_counter() - t0
        rec["ok"] = audio["bytes"] > 0
        if not rec["ok"]:
            rec["error"] = "empty audio (header only)"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["gen_s"] = time.perf_counter() - t0
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        # Derive in the finally, so a stream that died mid-flight keeps the chunks it
        # did receive: it held a server slot for real time, and dropping it biases an
        # occupancy integral low and hides the failure. ``ok`` stays False for those.
        # t_end likewise MUST be stamped here -- the early returns above (non-200,
        # non-chunked) would otherwise leave it None, and those are exactly the records
        # needed to account for rejected requests.
        rec["t_end"] = time.time()
        rec["nchunks"] = len(marks)
        rec["bytes"] = audio["bytes"] + (WAV_HEADER_BYTES - audio["hdr_left"])
        rec["audio_s"] = audio["bytes"] / BYTES_PER_SAMPLE / SAMPLE_RATE
        if marks:
            rec["ttfb_s"] = marks[0]
            if len(marks) > 1:
                rec["sc_s"] = marks[1] - marks[0]
            # TC = steady-state inter-chunk gaps, from the 3rd chunk on
            rec["gaps"] = [marks[i + 1] - marks[i] for i in range(1, len(marks) - 1)]
    return rec


def _serializable(rec: dict, conc: int) -> dict:
    """Trim one record for the sweep JSON: full marks are dump-only detail."""
    out = {k: v for k, v in rec.items() if k not in ("marks", "gaps", "uid")}
    out["conc"] = conc
    out["tc_gaps"] = [round(g, 6) for g in rec["gaps"][:MAX_TC_GAPS]]
    return out


# ---------------------------------------------------------------------- arrival loops


def run_closed(
    tgt: Target, conc: int, duration: float, warmup: float
) -> tuple[list, float, float]:
    """Closed loop: each worker sends its next request as soon as one completes."""
    stop = threading.Event()
    out, lock = [], threading.Lock()

    def worker(uid: int) -> None:
        while not stop.is_set():
            rec = {"uid": uid, "t_send": time.time(), "t_end": None}
            with lock:
                out.append(rec)          # registered BEFORE the send, never vanishes
            request(tgt, uid, rec)

    logger.info("  warmup %.0fs @ conc=%d ...", warmup, conc)
    threads = [
        threading.Thread(target=worker, args=(u,), daemon=True) for u in range(conc)
    ]
    t_start = time.time()
    for t in threads:
        t.start()
    time.sleep(warmup + duration)  # steady-state window begins after warmup
    stop.set()
    # Close the window the instant arrivals stop, BEFORE the join -- run_open already
    # does this and explains why; run_closed was still stamping it post-join, so the
    # drain inflated the denominator. Measured on the closed loop: RPS read 18.8% low
    # (u=256: 12.90 vs a true 15.88) and the occupancy integral ~14% low with it.
    w_end = time.time()
    deadline = time.time() + tgt.timeout + 5
    for t in threads:
        t.join(max(0, deadline - time.time()))
    cut = t_start + warmup
    with lock:
        # Return EVERY record. Filtering by arrival here also discarded the requests
        # that were already in flight at `cut` -- at steady state that is ~C of them,
        # so the occupancy integral lost ~C*W/2 request-seconds and read several
        # percent low. Latency/throughput filters belong downstream, per metric.
        return list(out), cut, w_end


def run_open(
    tgt: Target,
    conc: int,
    duration: float,
    warmup: float,
    mean_service_s: float,
    rate: float | None = None,
    max_inflight: int = 0,
    on_full: str = "shed",
    max_inflight_mult: int = 4,
) -> tuple[list, float, float, int, float]:
    """Open loop: Poisson arrivals, rate derived the way tts_sim derives it.

    Unlike the closed loop this does NOT self-throttle, so two things are handled:

      * in-flight is capped (``max_inflight``, else ``max_inflight_mult * conc``).
        Past that, arrivals are SHED and counted -- an unbounded thread spawn would
        measure the client's collapse, not the server's.
      * the caller must check whether the level ever reached steady state; a
        percentile from a non-converged window is an artifact of window length.

    ``max_inflight`` turns this hybrid: paced arrivals with a hard ceiling, where
    ``on_full="shed"`` drops the arrival (load-balancer reject) and ``"block"`` makes
    it wait for a free slot (connection pool). Block degenerates to a closed loop as
    lambda grows, and to a pure open loop as the ceiling grows.
    """
    lam = rate if rate else (conc / mean_service_s if mean_service_s > 0 else 1.0)
    logger.info(
        "  open loop: lambda=%.2f req/s (C_target=%d / W=%.1fs), warmup %.0fs",
        lam,
        conc,
        mean_service_s,
        warmup,
    )
    rng = random.Random(42 + conc)  # tts_sim rng_seed default is 42
    stop = threading.Event()
    out, lock = [], threading.Lock()
    inflight = [0]
    shed = [0]
    blocked, blocked_s = [0], [0.0]
    cap = (
        max(1, int(max_inflight))
        if max_inflight
        else max(1, int(max_inflight_mult * conc))
    )
    if max_inflight:
        logger.info("  in-flight ceiling: %d (%s when full)", cap, on_full)

    def fire() -> None:
        # Register before sending so an arrival that is still in flight when the run
        # ends is in the record set; otherwise unfinished requests vanish and the
        # occupancy integral cannot see the work they represent.
        rec = {"uid": -1, "t_send": time.time(), "t_end": None}
        with lock:
            out.append(rec)
        request(tgt, -1, rec)
        with lock:
            inflight[0] -= 1

    t_start = time.time()
    while not stop.is_set():
        if time.time() - t_start >= warmup + duration:
            break
        with lock:
            busy = inflight[0]
        if busy >= cap and on_full == "block":
            t_blocked = time.time()
            while not stop.is_set():
                with lock:
                    if inflight[0] < cap:
                        break
                time.sleep(0.002)
            blocked[0] += 1
            blocked_s[0] += time.time() - t_blocked
            with lock:
                inflight[0] += 1
            threading.Thread(target=fire, daemon=True).start()
        elif busy >= cap:
            shed[0] += 1
        else:
            with lock:
                inflight[0] += 1
            threading.Thread(target=fire, daemon=True).start()
        time.sleep(rng.expovariate(lam))

    if max_inflight and on_full == "block" and blocked[0]:
        logger.info(
            "  blocked %d arrivals, %.0f ms mean wait",
            blocked[0],
            blocked_s[0] / max(1, blocked[0]) * 1000,
        )
    cut = t_start + warmup
    # The measurement window ENDS when arrivals stop. The drain that follows must not
    # extend it: completions would then be divided by a longer window and both RPS and
    # C would read low -- an accounting artifact, not server behaviour.
    w_end = time.time()
    # drain: stop arriving, let in-flight finish (bounded), so the server is not left
    # with abandoned streams
    deadline = time.time() + tgt.timeout + 30
    while time.time() < deadline:
        with lock:
            if inflight[0] == 0:
                break
        time.sleep(0.5)
    with lock:
        return list(out), cut, w_end, shed[0], lam   # every record; filter downstream


def run_burst(tgt: Target, conc: int) -> tuple[list, float, float]:
    """Burst: fire all ``conc`` requests at the SAME instant, then wait for every one.

    No pacing, no self-throttling: every request is outstanding from t=0, so the
    server sees its worst-case simultaneous prefill demand. One shot per level -- there
    is no steady state, so the window is simply first-send to last-completion.
    """
    recs, lock = [], threading.Lock()
    start_gate = threading.Event()

    def worker(uid: int) -> None:
        start_gate.wait()  # release every thread together
        rec = request(tgt, uid)
        with lock:
            recs.append(rec)

    threads = [
        threading.Thread(target=worker, args=(u,), daemon=True) for u in range(conc)
    ]
    for t in threads:
        t.start()
    time.sleep(0.5)  # let every thread reach the gate
    logger.info("  burst: releasing %d simultaneous requests", conc)
    w0 = time.time()
    start_gate.set()
    for t in threads:
        t.join(tgt.timeout + 60)
    return recs, w0, time.time()


# ----------------------------------------------------------------------------- sweep


def _preflight(tgt: Target) -> dict:
    """Fail fast rather than burning a whole sweep on a dead server.

    Retried: this stack can intermittently return an empty WAV (HTTP 200, header only)
    at a low rate, and a single unlucky probe must not abort a sweep -- but a server
    that is genuinely down must still fail fast, so the tries are capped.
    """
    probe = {}
    for attempt in range(1, 4):
        probe = request(tgt)
        if probe["ok"]:
            logger.info(
                "preflight ok (ttfb %.0f ms, %d chunks, %d B)",
                probe["ttfb_s"] * 1000,
                probe["nchunks"],
                probe["bytes"],
            )
            return probe
        logger.warning("preflight attempt %d/3 failed: %s", attempt, probe["error"])
    sys.exit(f"preflight FAILED 3x: {probe['error']} -- is the server up and healthy?")


def sweep(args: argparse.Namespace, tgt: Target) -> dict:
    """Run every concurrency level and return the results document."""
    levels = sorted({int(x) for x in args.concurrency.split(",") if x.strip()})
    logger.info(
        "target http://%s:%d%s  chars/req=%d  levels=%s",
        tgt.host,
        tgt.port,
        tgt.path,
        len(tgt.text),
        levels,
    )
    if tgt.max_chunks:
        logger.info(
            "OSL cap: %d chunks = %d output tokens",
            tgt.max_chunks,
            tgt.max_chunks * CHUNK_TOKENS,
        )
    probe = _preflight(tgt)

    meta = {
        "host": tgt.host,
        "port": tgt.port,
        "chars_per_request": len(tgt.text),
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "arrival": args.arrival,
        "closed_loop": args.arrival == "closed",
        "started": time.time(),
    }
    # Open loop needs a service-time estimate to derive lambda. Seed it from the
    # preflight -- a single unloaded request, so it UNDERestimates W under load and
    # therefore overestimates lambda -- then refine it after every level from the
    # measured mean, which is what keeps later levels honest.
    mean_service_s = probe["gen_s"] or 1.0
    all_recs, levels_meta = [], []
    doc = {"meta": meta, "levels": levels_meta, "records": all_recs}

    for conc in levels:
        logger.info(
            "[conc=%d] running %.0fs (%s loop) ...", conc, args.duration, args.arrival
        )
        shed, lam = 0, None
        if args.arrival == "open":
            recs, w0, w1, shed, lam = run_open(
                tgt,
                conc,
                args.duration,
                args.warmup,
                mean_service_s,
                rate=args.rate,
                max_inflight=args.max_inflight,
                on_full=args.on_full,
            )
        elif args.arrival == "burst":
            recs, w0, w1 = run_burst(tgt, conc)
        else:
            recs, w0, w1 = run_closed(tgt, conc, args.duration, args.warmup)

        ok = [r for r in recs if r["ok"]]
        rps = len(ok) / (w1 - w0) if w1 > w0 else 0.0
        ttfbs = sorted(r["ttfb_s"] for r in ok if r["ttfb_s"] is not None)
        p50 = ttfbs[len(ttfbs) // 2] * 1000 if ttfbs else float("nan")
        gens = [r["gen_s"] for r in ok if r.get("gen_s")]
        if gens:
            mean_service_s = st.mean(gens)
        # Steady state check: an open-loop level past capacity never converges -- TTFB
        # drifts upward for the whole window, so a percentile over it is a function of
        # how long we ran, not of the server. Compare the window's halves.
        unstable = False
        if ok:
            mid = (w0 + w1) / 2
            first = [r["ttfb_s"] for r in ok if r["t_send"] < mid and r["ttfb_s"]]
            second = [r["ttfb_s"] for r in ok if r["t_send"] >= mid and r["ttfb_s"]]
            if len(first) >= 5 and len(second) >= 5:
                unstable = st.median(second) > 1.5 * st.median(first)
        extra = f"  shed={shed}" if args.arrival == "open" else ""
        if unstable:
            extra += (
                "  UNSTABLE (TTFB still climbing -- level did not reach steady state)"
            )
        logger.info(
            "[conc=%d] %d/%d ok  rps=%.2f  ttfb_p50=%.0fms  errors=%d%s",
            conc,
            len(ok),
            len(recs),
            rps,
            p50,
            len(recs) - len(ok),
            extra,
        )

        all_recs += [_serializable(r, conc) for r in recs]
        levels_meta.append(
            {
                "conc": conc,
                "window_start": w0,
                "window_end": w1,
                "n_total": len(recs),
                "n_ok": len(ok),
                "shed": shed,
                "lambda_rps": lam,
                "unstable": unstable,
                "mean_service_s": mean_service_s,
            }
        )
        with open(args.out, "w") as f:  # checkpoint after every level
            json.dump(doc, f)
        if conc != levels[-1]:
            time.sleep(args.settle)

    meta["finished"] = time.time()
    with open(args.out, "w") as f:
        json.dump(doc, f)
    logger.info("wrote %s  (%d records)", args.out, len(all_recs))
    return doc


# ----------------------------------------------------------------- aggregate + report


def pct(vals, p: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), min(int(math.floor(k)) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _metric_samples(ok_recs: list[dict], key: str) -> list[float]:
    """Milliseconds for one metric. ``tc_gaps`` is a LIST per request -- every gap is a
    sample, matching how the TC deadline was always evaluated (per gap, not per
    request), so one long request cannot hide many late chunks."""
    field_name = TARGETS[key][2]
    out = []
    for r in ok_recs:
        v = r.get(field_name)
        if v is None:
            continue
        if isinstance(v, list):
            out.extend(x * 1000.0 for x in v)
        else:
            out.append(v * 1000.0)
    return out


def _occupancy(recs: list[dict], w0: float, w1: float) -> float:
    """Time-average in-flight over [w0, w1], from arrival/departure events.

    NOT assumed equal to the worker count: requests that start before the window or
    run past it are clipped, so this is what the server actually carried. Cross-check
    with Little's law, C = RPS * W.
    """
    if not (w0 and w1 and w1 > w0):
        return float("nan")
    events = []
    for r in recs:
        start = max(r["t_send"], w0)
        end = min(r.get("t_end") or w1, w1)
        if end > start:
            events.append((start, 1))
            events.append((end, -1))
    events.sort()
    area, cur, prev = 0.0, 0, w0
    for t, delta in events:
        area += cur * (t - prev)
        prev = t
        cur += delta
    area += cur * (w1 - prev)
    return area / (w1 - w0)


def aggregate(doc: dict) -> tuple[list[dict], int]:
    """One row per concurrency level.

    Only completed requests count toward RPS. A request that errored, returned
    non-200, or came back as a bare WAV header processed no characters and is
    reported in the errors column instead -- a level that "went fast" by failing
    cannot masquerade as throughput. This matters: a broken build can return HTTP 200
    with an empty WAV in ~50 ms, which naive accounting scores as enormous throughput.
    """
    chars = doc["meta"]["chars_per_request"]
    by_conc: dict[int, list[dict]] = {}
    for r in doc["records"]:
        by_conc.setdefault(r["conc"], []).append(r)
    level_meta = {lvl["conc"]: lvl for lvl in doc.get("levels", [])}

    rows = []
    for conc in sorted(by_conc):
        recs = by_conc[conc]
        meta = level_meta.get(conc, {})
        w0, w1 = meta.get("window_start"), meta.get("window_end")
        # TWO COHORTS, deliberately different:
        #  * `recs` (everything, incl. warmup carryover and unfinished) -> occupancy.
        #    Those requests really were occupying the server inside the window.
        #  * `ok` (clean completions that ARRIVED in the window) -> latency + RPS,
        #    which must not be inflated by warmup arrivals now that run_closed and
        #    run_open return the full record set instead of pre-filtering.
        ok = [r for r in recs if r.get("ok")
              and (w0 is None or w1 is None or w0 <= r["t_send"] < w1)]
        window = (meta.get("window_end", 0) - meta.get("window_start", 0)) or float(
            "nan"
        )
        rps = len(ok) / window if window and window == window else float("nan")
        ttfb = [r["ttfb_s"] * 1000 for r in ok if r.get("ttfb_s") is not None]
        gen = [r["gen_s"] for r in ok if r.get("gen_s")]
        audio = [r["audio_s"] for r in ok if r.get("audio_s")]
        rtf = [
            r["gen_s"] / r["audio_s"] for r in ok if r.get("audio_s") and r.get("gen_s")
        ]
        nchunks = [r["nchunks"] for r in ok if r.get("nchunks")]
        errors: dict[str, int] = {}
        for r in recs:
            if not r.get("ok"):
                key = r.get("error") or "?"
                errors[key] = errors.get(key, 0) + 1

        row = {
            "conc": conc,
            "n_ok": len(ok),
            "n_err": len(recs) - len(ok),
            "rps": rps,
            "ttfb_p50": pct(ttfb, 50),
            "ttfb_p90": pct(ttfb, 90),
            "ttfb_p99": pct(ttfb, 99),
            "gen_p50": pct(gen, 50) if gen else float("nan"),
            "audio_p50": pct(audio, 50) if audio else float("nan"),
            "rtf_p50": pct(rtf, 50) if rtf else float("nan"),
            "mchar_h": chars * rps * 3600.0 / 1e6 if rps == rps else float("nan"),
            "C": _occupancy(recs, w0, w1),
            "chunks": (sum(nchunks) / len(nchunks)) if nchunks else float("nan"),
            "errors": errors,
        }
        for key in TARGETS:
            samples = _metric_samples(ok, key)
            row[f"{key}_p50"] = pct(samples, 50)
            row[f"{key}_p90"] = pct(samples, 90)
            row[f"{key}_p99"] = pct(samples, 99)
            row[f"{key}_n"] = len(samples)
        rows.append(row)
    return rows, chars


SUMMARY_COLS = [
    ("conc", "users", 5),
    ("rps", "RPS", 7),
    ("mchar_h", "Mchar/h", 8),
    ("ttfb_p50", "TTFB p50", 9),
    ("ttfb_p90", "TTFB p90", 9),
    ("ttfs_p50", "TTFS p50", 9),
    ("ttfs_p90", "TTFS p90", 9),
    ("ttft_p50", "TTFT p50", 9),
    ("ttft_p90", "TTFT p90", 9),
    ("rtf_p50", "RTF", 6),
    ("n_ok", "ok", 6),
    ("n_err", "err", 5),
]


def summary_table(rows: list[dict], chars: int, closed_loop) -> str:
    """Throughput is Mchar/h = chars_per_request * RPS * 3600 / 1e6."""
    out = [f"\nchars/request = {chars}   closed-loop = {closed_loop}"]
    out.append("".join(h.rjust(w) for _, h, w in SUMMARY_COLS))
    out.append("-" * sum(w for _, _, w in SUMMARY_COLS))
    for r in rows:
        line = ""
        for key, _, w in SUMMARY_COLS:
            v = r[key]
            line += (f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(w)
        out.append(line)
    return "\n".join(out)


def box_table(rows: list[dict]) -> str:
    """The FC/SC/TC target table: one box per concurrency level, with the agreed
    deadlines and each metric's ratio to its target. A metric is ✓ only when it is at
    or under target -- ratios are value/target, so lower is better."""
    spec = [
        ("FC p50", "ttfb_p50", 150.0),
        # P95, not P90: the acceptance target is FC p95 <= 350 ms. Passing p90 says
        # nothing about p95, and a p90 check can report PASS on a level that misses.
        ("FC p95", "ttfb_p95", 350.0),
        ("SC p50", "ttfs_p50", 270.0),
        ("SC p95", "ttfs_p95", 270.0),
        ("TC p50", "ttft_p50", 570.0),
        ("TC p95", "ttft_p95", 570.0),
    ]
    heads = ["users", "chunks", "C"] + [n for n, _, _ in spec]
    widths = [7, 7, 7] + [8] * len(spec)

    def rule(left, mid, right):
        return left + mid.join("─" * w for w in widths) + right

    def row(cells):
        return "│" + "│".join(c.center(w) for c, w in zip(cells, widths)) + "│"

    out = []
    for r in rows:
        values = [str(r["conc"]), f"{r['chunks']:.1f}", f"{r['C']:.1f}"]
        values += [f"{r[k]:.0f}ms" if r[k] == r[k] else "n/a" for _, k, _ in spec]
        marks = []
        for _, k, target in spec:
            v = r[k]
            marks.append(
                "n/a"
                if v != v
                else ("✓ " if v <= target else "✗ ") + f"{v / target:.2f}×"
            )
        out.append(rule("┌", "┬", "┐"))
        out.append(row(heads))
        out.append(rule("├", "┼", "┤"))
        out.append(row(values))
        out.append(rule("├", "┼", "┤"))
        out.append(row(["target", "", ""] + [f"≤{t:.0f}ms" for _, _, t in spec]))
        out.append(rule("├", "┼", "┤"))
        out.append(row(["", "", ""] + marks))
        out.append(rule("└", "┴", "┘"))
        if r["n_ok"] < 20:
            out.append(
                f"  ! only {r['n_ok']} completed requests at conc={r['conc']}: "
                f"p90/p99 are "
                f"dominated by single outliers (cold IO threads pay a ~1.3 s tokenizer "
                f"cache miss). Treat p50 only."
            )
        out.append("")
    return "\n".join(out)


def print_report(doc: dict, args: argparse.Namespace) -> list[dict]:
    rows, chars = aggregate(doc)
    if not rows:
        sys.exit("no records")

    if getattr(args, "format", "table") == "csv":
        print(",".join(k for k, _, _ in SUMMARY_COLS))
        for r in rows:
            print(
                ",".join(
                    f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])
                    for k, _, _ in SUMMARY_COLS
                )
            )
    else:
        print(summary_table(rows, chars, doc["meta"].get("closed_loop")))
        print()
        print(box_table(rows), end="")
        bad = [r for r in rows if r["n_err"]]
        if bad:
            print("\nERRORS (excluded from RPS):")
            for r in bad:
                for err, n in sorted(r["errors"].items(), key=lambda x: -x[1]):
                    print(f"  conc={r['conc']:<5} {n:>5} x {err}")

    if getattr(args, "chart", None):
        _render_charts(rows, chars, args)
    return rows


def _render_charts(rows: list[dict], chars: int, args: argparse.Namespace) -> None:
    """TTFB/TTFS/TTFT (x) vs millions of characters per hour (y)."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, key, show_targets=True, colors=None):
        label, xaxis, _field, t50, t90 = TARGETS[key]
        c50, c90 = colors or ("tab:blue", "tab:orange")
        ax.plot(
            [r[f"{key}_p50"] for r in rows],
            [r["mchar_h"] for r in rows],
            "-o",
            color=c50,
            label=f"{label} p50",
            zorder=3,
        )
        ax.plot(
            [r[f"{key}_p90"] for r in rows],
            [r["mchar_h"] for r in rows],
            "--s",
            color=c90,
            label=f"{label} p90",
            zorder=3,
        )
        for r in rows:
            ax.annotate(
                str(r["conc"]),
                (r[f"{key}_p50"], r["mchar_h"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
        if show_targets:
            ax.axvline(
                t50,
                color="tab:green",
                ls=":",
                lw=1,
                label=f"{label} p50 target {t50:.0f} ms",
            )
            if t90 != t50:
                ax.axvline(
                    t90,
                    color="tab:orange",
                    ls=":",
                    lw=1,
                    label=f"{label} p90 target {t90:.0f} ms",
                )
        if args.mark_errors:
            err = [r for r in rows if r["n_err"]]
            if err:
                ax.scatter(
                    [r[f"{key}_p50"] for r in err],
                    [r["mchar_h"] for r in err],
                    s=140,
                    facecolors="none",
                    edgecolors="red",
                    zorder=4,
                    label="level had failed requests",
                )
        return xaxis

    if args.no_subtitle:
        sub = ""
    else:
        sub = args.subtitle or f"{chars} chars/request, closed-loop"
        sub = "\n".join(textwrap.wrap(sub + "; point label = concurrent users", 88))
    base, ext = os.path.splitext(args.chart)
    ext = ext or ".png"

    keys = ["ttfb", "ttfs", "ttft"] if args.target == "all" else [args.target]
    written = []
    for key in keys:
        if not any(r[f"{key}_n"] for r in rows):
            logger.warning(
                "skipping %s: no samples (needs >= %d chunks/request)",
                key,
                2 if key == "ttfs" else 3,
            )
            continue
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        xaxis = draw(ax, key)
        ax.set_xlabel(xaxis)
        ax.set_ylabel("Throughput (millions of characters / hour)")
        ax.set_title(f"{args.title}\n{sub}" if sub else args.title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        out = args.chart if len(keys) == 1 else f"{base}_{key}{ext}"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    if args.target == "all":
        # combined overlay: all three metrics share the latency axis
        fig, ax = plt.subplots(figsize=(10, 6.8))
        palette = {
            "ttfb": ("tab:blue", "tab:cyan"),
            "ttfs": ("tab:green", "yellowgreen"),
            "ttft": ("tab:red", "salmon"),
        }
        for key in keys:
            if any(r[f"{key}_n"] for r in rows):
                draw(ax, key, show_targets=False, colors=palette[key])
        for key in keys:
            t50 = TARGETS[key][3]
            ax.axvline(
                t50,
                ls=":",
                lw=1,
                color=palette[key][0],
                label=f"{TARGETS[key][0]} target {t50:.0f} ms",
            )
        ax.set_xlabel("latency (ms) — TTFB / TTFS / TTFT")
        ax.set_ylabel("Throughput (millions of characters / hour)")
        title = (
            args.title if args.no_subtitle else f"{args.title} — all latency targets"
        )
        ax.set_title(f"{title}\n{sub}" if sub else title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        out = f"{base}_combined{ext}"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    for path in written:
        logger.info("wrote %s", path)


# ------------------------------------------------------------------- pipeline dump


def dump(
    tgt: Target,
    users: int,
    duration: float,
    skip: float,
    outdir: str,
    warm_requests: int = 24,
) -> dict:
    """Closed-loop run at one concurrency, recording every chunk of every stream.

    Answers "where do the bubbles live?": for every virtual user, the arrival time of
    every HTTP chunk and the cadence derived from it.

    A BUBBLE is a real-time underrun, not just a slow chunk. Each audio chunk carries
    CHUNK_TOKENS tokens = 0.600 s of audio, and once playback has started the player
    drains that much buffer between chunks, so an inter-chunk gap > 600 ms means the
    pipeline produced audio slower than real time and the listener hears a stall.
    gap / 600 ms is the local RTF.
    """
    streams: list[dict] = []
    lock, stop = threading.Lock(), threading.Event()

    def worker(uid: int) -> None:
        while not stop.is_set():
            rec = {"uid": uid, "t_send": time.time(), "t_end": None}
            with lock:
                streams.append(rec)  # registered BEFORE the send, classified later
            request(tgt, uid, rec)

    logger.info("warming (%d users, %d chars)...", users, len(tgt.text))
    warmers = [
        threading.Thread(target=request, args=(tgt,)) for _ in range(warm_requests)
    ]
    for t in warmers:
        t.start()
    for t in warmers:
        t.join(tgt.timeout)

    t_start = time.time()
    threads = [
        threading.Thread(target=worker, args=(u,), daemon=True) for u in range(users)
    ]
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    # Measurement ends the instant arrivals stop -- BEFORE the join. A post-join stamp
    # includes the drain, during which workers finish without being replaced and
    # occupancy falls to zero; integrating to it dragged C ~14% low (u=256 read 218.8
    # against a Little's-law 254.9).
    measurement_end = time.time()
    for t in threads:
        t.join(120)
    window_end = time.time()

    cut = t_start + skip
    with lock:
        allrec = list(streams)
    # SCORED = completed cleanly. Accepting any >=3-chunk response let an errored
    # stream count toward latency percentiles and throughput.
    ok = [r for r in allrec
          if r.get("t_end") and not r["error"] and r["t_send"] >= cut
          and len(r["marks"]) >= 3]
    # ANY errored request is a failure, whatever it managed to deliver first. The
    # old `and len(marks) < 3` meant an errored >=3-chunk stream was excluded from
    # `ok` (correctly) but counted in no failure bucket either -- it just vanished.
    n_fail = sum(1 for r in allrec if r["error"])
    n_short = sum(1 for r in allrec if not r["error"] and 0 < len(r["marks"]) < 3)
    n_unfin = sum(1 for r in allrec if not r.get("t_end"))
    n_empty = sum(1 for r in allrec if not r["marks"])
    if not ok:
        raise SystemExit("no streams scored -- server too slow or refusing")

    for r in ok:
        r["admit_ms"] = r["admit_s"] * 1000.0
        r["fc_ms"] = r["ttfb_s"] * 1000.0
        r["sc_ms"] = r["sc_s"] * 1000.0 if r["sc_s"] is not None else float("nan")
        r["gaps_ms"] = [g * 1000.0 for g in r["gaps"]]
        # TIME IN SYSTEM -- what Little's law needs. audio_end_s stops at the last
        # AUDIO chunk; the request still holds a slot until the chunked terminator is
        # read and the socket closed, so using it for W understates C. Both are
        # reported, with the tail between them, rather than picking one silently.
        r["audio_end_s"] = r["marks"][-1]
        r["wall_s"] = r["t_end"] - r["t_send"]
        r["tail_s"] = r["wall_s"] - r["audio_end_s"]
        r["rtf"] = r["audio_end_s"] / r["audio_s"] if r["audio_s"] else float("nan")
        r["bubbles"] = [g for g in r["gaps_ms"] if g > CHUNK_AUDIO_MS]  # real underruns
        r["stall_ms"] = sum(g - CHUNK_AUDIO_MS for g in r["bubbles"])

    # C = total request-seconds overlapping the window / window seconds. The window is
    # [cut, measurement_end]: after warmup, before drain. EVERY request that overlaps
    # it counts -- failures, short responses, and requests that started during warmup
    # and were still running at `cut` (clipped by the max/min below). Retaining those
    # removes the ramp-in that would otherwise need a service-time offset. Validates
    # against Little's law to <1%.
    area = 0.0
    for r in allrec:
        lo = max(r["t_send"], cut)
        # unfinished at measurement_end -> still occupying the server, clip it there
        hi = min(r["t_end"] or measurement_end, measurement_end)
        if hi > lo:
            area += hi - lo
    c_meas = area / (measurement_end - cut) if measurement_end > cut else 0.0

    os.makedirs(outdir, exist_ok=True)
    rows = _dump_user_rows(ok, users)
    tsv = _write_dump_tsvs(ok, rows, outdir, users, t_start)

    # RPS MUST divide arrivals by the ARRIVAL window, not by a post-join window: the
    # latter includes the drain (~W seconds), which understated RPS -- and chunks/s and
    # Mchar/h with it -- by ~24% at duration=90/skip=25/W=16 (u=256 read 12.90 RPS
    # against a true 15.95, and Little's law then gave C=207 where the occupancy trace
    # correctly said ~256).
    elapsed = duration - skip
    summary = {
        "users": users,
        "scored": len(ok),
        "recorded": len(allrec),
        "failed": n_fail,
        "short": n_short,
        "empty": n_empty,
        "chunks": sum(r["nchunks"] for r in ok),
        "elapsed_s": elapsed,
        "rps": len(ok) / elapsed,
        "C": c_meas,
        "window_s": measurement_end - cut,
        "drain_window_s": window_end - cut,
        "fc": [r["fc_ms"] for r in ok],
        "sc": [r["sc_ms"] for r in ok if r["sc_ms"] == r["sc_ms"]],
        "tc": [g for r in ok for g in r["gaps_ms"]],
        "admit": [r["admit_ms"] for r in ok],
        "bubbles": [g for r in ok for g in r["bubbles"]],
        "rtf": [r["rtf"] for r in ok],
        "mean_gen_s": st.mean([r["audio_end_s"] for r in ok]),
        "mean_wall_s": st.mean([r["wall_s"] for r in ok]),
        "mean_tail_s": st.mean([r["tail_s"] for r in ok]),
        "users_with_bubbles": sum(1 for r in rows if r["bubbles"] > 0),
        "n_users": len(rows),
        "chars": len(tgt.text),
        "tsv": tsv,
        "outdir": outdir,
    }
    print(_dump_text(summary))
    return summary


def _dump_user_rows(ok: list[dict], users: int) -> list[dict]:
    rows = []
    for uid in range(users):
        mine = [r for r in ok if r["uid"] == uid]
        if not mine:
            continue
        gaps = [g for r in mine for g in r["gaps_ms"]]
        bubbles = [g for r in mine for g in r["bubbles"]]
        rows.append(
            {
                "u": uid,
                "streams": len(mine),
                "chunks": sum(r["nchunks"] for r in mine),
                "admit_p50": pct([r["admit_ms"] for r in mine], 50),
                "fc_p50": pct([r["fc_ms"] for r in mine], 50),
                "fc_p90": pct([r["fc_ms"] for r in mine], 90),
                "sc_p50": pct([r["sc_ms"] for r in mine], 50),
                "tc_p50": pct(gaps, 50),
                "tc_p90": pct(gaps, 90),
                "tc_max": max(gaps) if gaps else float("nan"),
                "bubbles": len(bubbles),
                "bub_pct": 100.0 * len(bubbles) / len(gaps) if gaps else 0.0,
                "stall_ms": sum(r["stall_ms"] for r in mine),
                "rtf_p50": pct([r["rtf"] for r in mine], 50),
            }
        )
    return rows


def _write_dump_tsvs(
    ok: list[dict], rows: list[dict], outdir: str, users: int, t_start: float
) -> str:
    header = [
        "u",
        "streams",
        "chunks",
        "admit_p50",
        "fc_p50",
        "fc_p90",
        "sc_p50",
        "tc_p50",
        "tc_p90",
        "tc_max",
        "bubbles",
        "bub_pct",
        "stall_ms",
        "rtf_p50",
    ]
    tsv = os.path.join(outdir, f"pipeline_dump_u0_u{users - 1}.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h])
                    for h in header
                )
                + "\n"
            )

    with open(os.path.join(outdir, "timings.tsv"), "w") as f:
        f.write("t_send\tt_end\tgen_s\twall_s\ttail_s\tnchunks\n")
        for r in ok:
            f.write(
                f"{r['t_send']:.4f}\t{r['t_end']:.4f}\t{r['audio_end_s']:.4f}\t"
                f"{r['wall_s']:.4f}\t{r['tail_s']:.4f}\t{r['nchunks']}\n"
            )

    # occupancy trace over the scored set (for the tsv only; C uses every request)
    events = []
    for r in ok:
        events.append((r["t_send"], +1))
        events.append((r["t_end"], -1))
    events.sort()
    with open(os.path.join(outdir, "occupancy.tsv"), "w") as f:
        f.write("t_rel_s\tin_flight\n")
        cur = 0
        for t, delta in events:
            f.write(f"{t - t_start:.3f}\t{cur}\n")
            cur += delta
    return tsv


def _pct_over(vals: list[float], deadline: float) -> float:
    return 100.0 * sum(1 for v in vals if v > deadline) / max(1, len(vals))


def _latency_row(label: str, vals: list[float], deadline: str) -> str:
    """One percentile row of the dump's latency table."""
    cells = "".join(f"{pct(vals, p):10.1f}" for p in (50, 90, 95, 99))
    return f"{label:<11}{cells}{max(vals):11.1f}   {deadline}"


def _dump_text(s: dict) -> str:
    fc, sc, tc, admit = s["fc"], s["sc"], s["tc"], s["admit"]
    stall_s = sum(g - CHUNK_AUDIO_MS for g in s["bubbles"]) / 1000.0
    latency = "\n".join(
        [
            _latency_row("admission", admit, "--"),
            _latency_row("FC", fc, f"{FC_P50_MS:.0f}/{FC_P95_MS:.0f}"),
            _latency_row("SC", sc, f"{SC_DEADLINE_MS:.0f}"),
            _latency_row("TC", tc, f"{TC_DEADLINE_MS:.0f}"),
        ]
    )
    header = f"pipeline dump: {s['users']} users, {s['elapsed_s']:.0f}s scored"
    fc_p50 = "MISS" if pct(fc, 50) > FC_P50_MS else "ok"
    fc_p95 = "MISS" if pct(fc, 95) > FC_P95_MS else "ok"
    return f"""
================ {header} ================
streams scored {s["scored"]}   chunks {s["chunks"]}   \
chunks/s {s["chunks"] / s["elapsed_s"]:.1f}   chars/request {s["chars"]}
requests recorded {s["recorded"]}   scored {s["scored"]}   failed {s["failed"]}   \
short(1-2ch) {s["short"]}   empty {s["empty"]}
arrival window {s["elapsed_s"]:.1f}s   C window {s["window_s"]:.1f}s   \
(drain excluded; with drain it would be {s["drain_window_s"]:.1f}s)
C (time-avg in-flight)  {s["C"]:.1f}          RPS {s["rps"]:.2f}
  Little's law  C = RPS x W
     W = gen_s  (to last audio chunk) {s["mean_gen_s"]:6.2f}s -> C \
{s["rps"] * s["mean_gen_s"]:7.1f}
     W = wall_s (to socket close)     {s["mean_wall_s"]:6.2f}s -> C \
{s["rps"] * s["mean_wall_s"]:7.1f}
     post-audio tail                  {s["mean_tail_s"]:6.2f}s

                  p50       p90       p95       p99        max      deadline
{latency}

BUBBLES (inter-chunk gap > {CHUNK_AUDIO_MS:.0f} ms = audio underrun)
  count {len(s["bubbles"])} / {len(tc)} gaps = \
{100.0 * len(s["bubbles"]) / max(1, len(tc)):.1f}%
  total stall {stall_s:.1f} s across {s["scored"]} streams
  worst gap {max(tc):.0f} ms = {max(tc) / CHUNK_AUDIO_MS:.2f}x real time
  users with >=1 bubble: {s["users_with_bubbles"]} / {s["n_users"]}
  RTF p50 {pct(s["rtf"], 50):.3f}  p90 {pct(s["rtf"], 90):.3f}  (must stay < 1.0)

DEADLINE VIOLATIONS
  FC p50 {fc_p50}   FC p95 {fc_p95}
  SC over {SC_DEADLINE_MS:.0f}ms: {_pct_over(sc, SC_DEADLINE_MS):.1f}%
  TC over {TC_DEADLINE_MS:.0f}ms: {_pct_over(tc, TC_DEADLINE_MS):.1f}%

wrote {s["tsv"]}
      {os.path.join(s["outdir"], "occupancy.tsv")}
"""


# ------------------------------------------------------------------------- ladder


def _docker_log_lines(container: str) -> int:
    """Current log length, used to mark where this level's telemetry starts."""
    try:
        out = subprocess.run(
            ["docker", "logs", container], capture_output=True, text=True
        )
        return len((out.stdout + out.stderr).splitlines())
    except FileNotFoundError:
        logger.warning("docker not found -- skipping server-side span capture")
        return -1


def _capture_spans(container: str, mark: int, path: str) -> None:
    """Write this level's [tts-spans]/[tts-writer] lines and report their medians."""
    if mark < 0:
        return
    out = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
    lines = [
        line
        for line in (out.stdout + out.stderr).splitlines()[mark:]
        if "tts-spans" in line or "tts-writer" in line
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    device_us = [
        float(m) for line in lines for m in re.findall(r"device_us=([\d.]+)", line)
    ]
    rows_avg = [
        float(m) for line in lines for m in re.findall(r"rows_avg=([\d.]+)", line)
    ]
    if device_us:
        logger.info(
            "  spans: device_us_p50=%.0f rows_avg_p50=%.2f samples=%d",
            pct(device_us, 50),
            pct(rows_avg, 50) if rows_avg else float("nan"),
            len(device_us),
        )


LADDER_COLS = [
    ("users", 6),
    ("scored", 8),
    ("RPS", 8),
    ("C", 8),
    ("FC p50", 9),
    ("FC p90", 9),
    ("SC p50", 9),
    ("TC p50", 9),
    ("TC p90", 9),
    ("bubble%", 9),
    ("RTF p50", 9),
    ("failed", 8),
]


def ladder_table(results: list[dict]) -> str:
    out = [
        "",
        "".join(h.rjust(w) for h, w in LADDER_COLS),
        "-" * sum(w for _, w in LADDER_COLS),
    ]
    for s in results:
        tc = s["tc"]
        cells = [
            str(s["users"]),
            str(s["scored"]),
            f"{s['rps']:.2f}",
            f"{s['C']:.1f}",
            f"{pct(s['fc'], 50):.0f}",
            f"{pct(s['fc'], 90):.0f}",
            f"{pct(s['sc'], 50):.0f}",
            f"{pct(tc, 50):.0f}",
            f"{pct(tc, 90):.0f}",
            f"{100.0 * len(s['bubbles']) / max(1, len(tc)):.1f}",
            f"{pct(s['rtf'], 50):.3f}",
            str(s["failed"]),
        ]
        out.append("".join(c.rjust(w) for c, (_, w) in zip(cells, LADDER_COLS)))
    return "\n".join(out)


def ladder(args: argparse.Namespace, tgt: Target) -> list[dict]:
    """Run ``dump`` at each user count, capturing the server spans for that window."""
    levels = sorted({int(x) for x in args.users.split(",") if x.strip()})
    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for users in levels:
        logger.info("=== %d users %s ===", users, time.strftime("%H:%M:%S"))
        mark = _docker_log_lines(args.container) if args.container else -1
        summary = dump(
            tgt,
            users,
            args.duration,
            args.skip,
            os.path.join(args.outdir, f"dump_u{users}"),
            args.warm_requests,
        )
        if args.container:
            _capture_spans(
                args.container, mark, os.path.join(args.outdir, f"u{users}.server.log")
            )
        results.append(summary)
        if users != levels[-1]:
            time.sleep(args.settle)
    print(ladder_table(results))
    return results


# ---------------------------------------------------------------------------- CLI


def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=os.environ.get("TTS_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", 8000)))
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--key", default=os.environ.get("TTS_KEY", DEFAULT_KEY))
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--text-chars", type=int, default=2000)
    p.add_argument(
        "--text-tokens",
        type=int,
        default=None,
        help="input length in TOKENS (ISL) instead of chars, converted at "
        "--chars-per-token. Overrides --text-chars.",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=3.46,
        help="measured for this tokenizer/corpus: 3.46",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="OSL cap in audio chunks (30 speech tokens each); 0 = uncapped. "
        "34 -> OSL 1020, keeping ISL+OSL under CACHE_MAX_SEQ_LEN=2048.",
    )


def _add_report_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=["table", "csv"], default="table")
    p.add_argument(
        "--chart", default=None, help="write latency/throughput chart(s) here"
    )
    p.add_argument(
        "--target",
        default="ttfb",
        choices=["ttfb", "ttfs", "ttft", "all"],
        help="which latency metric on the chart's x axis. ttfb=first chunk "
        "(FC), ttfs=first->second gap (SC), ttft=steady-state gap (TC), "
        "all=one chart each plus a combined overlay",
    )
    p.add_argument("--title", default="TTS — latency vs throughput")
    p.add_argument("--subtitle", default="")
    p.add_argument(
        "--no-subtitle",
        action="store_true",
        help="render the title alone, with no second line",
    )
    p.add_argument(
        "--mark-errors",
        action="store_true",
        help="ring levels that had failed requests (off by default; failures "
        "are always listed in the table regardless)",
    )


def _build_target(args: argparse.Namespace) -> Target:
    nchars = args.text_chars
    if getattr(args, "text_tokens", None):
        nchars = int(round(args.text_tokens * args.chars_per_token))
    return Target(
        host=args.host,
        port=args.port,
        path=args.path,
        key=args.key,
        timeout=args.timeout,
        text=build_text(nchars),
        max_chunks=args.max_chunks,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep", help="concurrency sweep -> results JSON + tables")
    _add_target_args(p)
    _add_report_args(p)
    p.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32,64,128,256",
        help="comma list, swept in increasing order",
    )
    p.add_argument(
        "--duration", type=float, default=60, help="steady-state seconds per level"
    )
    p.add_argument(
        "--warmup", type=float, default=20, help="seconds discarded per level"
    )
    p.add_argument(
        "--settle", type=float, default=5, help="idle seconds between levels"
    )
    p.add_argument("--out", default="tts_results.json")
    p.add_argument(
        "--arrival",
        choices=["closed", "open", "burst"],
        default="closed",
        help="see the module docstring: closed self-throttles, open is a "
        "Poisson process that can overload, burst is a thundering herd",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=None,
        help="open loop: FIXED arrival rate in req/s, overriding the conc/W "
        "derived lambda. Use for 'serve at RPS N' questions.",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="hybrid: hard ceiling on concurrent requests (0 = 4x conc)",
    )
    p.add_argument(
        "--on-full",
        choices=["shed", "block"],
        default="shed",
        help="at the ceiling: drop the arrival (shed) or make it wait (block)",
    )
    p.add_argument(
        "--no-table", action="store_true", help="skip the tables after the sweep"
    )

    p = sub.add_parser("report", help="tables and charts from a sweep's JSON")
    _add_report_args(p)
    p.add_argument("--results", required=True)

    p = sub.add_parser("dump", help="per-chunk detail at one concurrency")
    _add_target_args(p)
    p.add_argument("--users", type=int, default=64)
    p.add_argument("--duration", type=float, default=120)
    p.add_argument(
        "--skip", type=float, default=30, help="warmup seconds excluded from scoring"
    )
    p.add_argument("--outdir", default="tts_dump")
    p.add_argument("--warm-requests", type=int, default=24)

    p = sub.add_parser("ladder", help="dump repeated across a concurrency ladder")
    _add_target_args(p)
    p.add_argument(
        "--users", default="16,32,64,128,256", help="comma list of user counts"
    )
    p.add_argument("--duration", type=float, default=90)
    p.add_argument("--skip", type=float, default=25)
    p.add_argument(
        "--settle", type=float, default=5, help="idle seconds between levels"
    )
    p.add_argument("--outdir", default="tts_ladder")
    p.add_argument("--warm-requests", type=int, default=24)
    p.add_argument(
        "--container",
        default=os.environ.get("TTS_CONTAINER", ""),
        help="docker container whose [tts-spans]/[tts-writer] telemetry is "
        "captured per level; empty disables capture",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.command == "report":
        with open(args.results) as f:
            print_report(json.load(f), args)
        return

    tgt = _build_target(args)
    if args.command == "sweep":
        doc = sweep(args, tgt)
        if not args.no_table:
            print_report(doc, args)
    elif args.command == "dump":
        dump(tgt, args.users, args.duration, args.skip, args.outdir, args.warm_requests)
    elif args.command == "ladder":
        ladder(args, tgt)


if __name__ == "__main__":
    main()


__all__ = [
    "Target",
    "build_text",
    "request",
    "run_closed",
    "run_open",
    "run_burst",
    "sweep",
    "aggregate",
    "print_report",
    "summary_table",
    "box_table",
    "dump",
    "ladder",
    "ladder_table",
    "pct",
    "main",
]
