# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Standalone TTS load harness for a running server's ``/v1/audio/speech``.

The workflow engine imports its load-generation and aggregation functions through
``tts_benchmark_tests.py``. It can also be run directly against any deployed TTS
server with the CLI below.

    sweep    concurrency sweep, one of three arrival models -> results JSON
    report   that JSON -> summary table, FC/SC/TC box table, optional charts
    dump     one concurrency, per-chunk detail: occupancy, bubbles, per-user table
    ladder   dump repeated across a concurrency ladder

    python tts_load_harness.py sweep --host d08u08 --concurrency 1,2,4,8,16 \\
        --duration 60 --warmup 20 --text-tokens 1024 --out results.json
    python tts_load_harness.py report --results results.json --target all \\
        --chart curve.png
    python tts_load_harness.py dump --users 64 --duration 90 --skip 25 --outdir dump_u64
    python tts_load_harness.py ladder --users 16,32,64,128

Requires Python 3.10+. Only matplotlib is non-stdlib, and only for ``--chart`` --
every table works without it.

Measurement definitions:
  * Closed loop maintains a fixed worker pool. Open loop uses exponential sleeps
    between launches; thread-launch overhead adds to the requested intervals.
    A full in-flight cap either sheds arrivals or blocks the generator. Burst mode
    releases one request per worker together.
  * FC measures request start to the first complete audio chunk. SC is the second
    chunk gap; TC covers later gaps. The client assumes one HTTP chunk per audio
    chunk, a 44-byte WAV header, and mono 48 kHz, 16-bit PCM audio.
  * Latencies use a monotonic clock. Occupancy uses wall-clock request intervals,
    clipped to the measurement window, including failures and unfinished requests.
    Wall-clock changes can affect C; it measures client requests, not device slots.
  * Throughput is an arrival-cohort estimate: successful window arrivals divided
    by window duration, following them through drain. Capped requests count toward
    request rate but receive no character credit. It is not in-window completions.
  * Sweep JSON retains only the first 200 TC gaps per request. Passing SC/TC P95
    does not establish zero violations. Dump reports per-gap deadline violations.
  * Legacy "bubbles" and "stall_ms" fields count long gaps and their excess over
    one chunk's audio duration. They do not simulate buffered playback stalls.
  * Short runs have weak tail-percentile estimates. Unfinished requests contribute
    to occupancy but not successful-request latency; drain deadlines are bounded.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import socket
import statistics as st
import sys
import threading
import time
from dataclasses import dataclass, field


# Support both direct execution and package imports.
if __package__:
    from . import tts_load_report as reporting
else:
    import tts_load_report as reporting

SAMPLE_RATE = reporting.SAMPLE_RATE
CHUNK_TOKENS = reporting.CHUNK_TOKENS
CHUNK_AUDIO_S = reporting.CHUNK_AUDIO_S
CHUNK_AUDIO_MS = reporting.CHUNK_AUDIO_MS
FC_P50_MS = reporting.FC_P50_MS
FC_P95_MS = reporting.FC_P95_MS
SC_DEADLINE_MS = reporting.SC_DEADLINE_MS
TC_DEADLINE_MS = reporting.TC_DEADLINE_MS
TARGETS = reporting.TARGETS
pct = reporting.pct
_metric_samples = reporting._metric_samples
_arrival_cohort = reporting._arrival_cohort
_overlaps = reporting._overlaps
_occupancy = reporting._occupancy
aggregate = reporting.aggregate
SUMMARY_COLS = reporting.SUMMARY_COLS
summary_table = reporting.summary_table
box_table = reporting.box_table
print_report = reporting.print_report
_render_charts = reporting._render_charts
_dump_user_rows = reporting._dump_user_rows
_write_dump_tsvs = reporting._write_dump_tsvs
_pct_over = reporting._pct_over
_latency_row = reporting._latency_row
_dump_text = reporting._dump_text
LADDER_COLS = reporting.LADDER_COLS
ladder_table = reporting.ladder_table

logger = logging.getLogger(__name__)

DEFAULT_PATH = "/v1/audio/speech"
DEFAULT_KEY = "your-secret-key"
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44
MAX_TC_GAPS = 200  # cap the per-request gap list so a sweep's JSON stays manageable


class PreflightError(RuntimeError):
    """The target server did not produce a valid response during preflight."""


@dataclass
class Target:
    """Everything one request needs: where to send it and what to send."""

    host: str = "localhost"
    port: int = 8000
    path: str = DEFAULT_PATH
    key: str = DEFAULT_KEY
    timeout: float = 300.0
    text: str = ""
    # Client-side audio chunk cap; 0 = uncapped. This does not limit server decoding.
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
    """Read a chunked WAV response and retain timings even on failure.

    FC/SC/TC use monotonic offsets from request start. Wall-clock t_send/t_end
    delimit client occupancy. A supplied record is registered before execution.
    """
    # A caller may register the record BEFORE the send (see run_closed/dump), so that
    # a request still in flight when the join deadline expires is still in the record
    # set. In that case t_send is already stamped and must not be reset here.
    if rec is None:
        rec = {"uid": uid, "t_send": time.time()}
    rec.update(
        {
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
        }
    )
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
        # Retain partial audio and stamp completion on every exit, including errors.
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
                out.append(rec)  # registered BEFORE the send, never vanishes
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
    # Exclude drain from the measurement window.
    w_end = time.time()
    deadline = time.time() + tgt.timeout + 5
    for t in threads:
        t.join(max(0, deadline - time.time()))
    cut = t_start + warmup
    with lock:
        # Keep warmup carryover for occupancy; filter cohorts downstream.
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
    """Launch requests with exponential interarrival sleeps and an in-flight cap.

    Rate is explicit or calibrated from target concurrency / mean request duration.
    At the cap, shed arrivals or block until a slot opens. Blocking introduces
    completion feedback; neither mode proves that the server reached steady state.
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
        rec = {"uid": -1, "t_send": time.time(), "t_end": None}
        with lock:
            out.append(rec)
        request(tgt, -1, rec)
        with lock:
            inflight[0] -= 1

    t_start = time.time()
    while True:
        if time.time() - t_start >= warmup + duration:
            break
        with lock:
            busy = inflight[0]
        if busy >= cap and on_full == "block":
            t_blocked = time.time()
            while True:
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
        return list(out), cut, w_end, shed[0], lam  # every record; filter downstream


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
    raise PreflightError(
        f"preflight FAILED 3x: {probe['error']} -- is the server up and healthy?"
    )


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

        in_win = _arrival_cohort(recs, w0, w1)
        ok = [r for r in in_win if r["ok"]]
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
            len(in_win),
            rps,
            p50,
            len(in_win) - len(ok),  # same cohort as the report; `recs` now carries
            # warmup, so len(recs)-len(ok) called healthy
            # warmup requests errors
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


# ------------------------------------------------------------------- pipeline dump


def dump(
    tgt: Target,
    users: int,
    duration: float,
    skip: float,
    outdir: str,
    warm_requests: int = 24,
) -> dict:
    """Closed-loop chunk diagnostics; legacy bubble fields measure long gaps."""
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
    # Exclude drain from the measurement window.
    measurement_end = time.time()
    for t in threads:
        t.join(120)
    window_end = time.time()

    cut = t_start + skip
    with lock:
        allrec = list(streams)
    # Chunk diagnostics require clean responses with at least three chunks.
    ok = [
        r
        for r in allrec
        if r.get("t_end")
        and not r["error"]
        and r["t_send"] >= cut
        and len(r["marks"]) >= 3
    ]
    n_fail = sum(1 for r in allrec if r["error"])
    n_short = sum(1 for r in allrec if not r["error"] and 0 < len(r["marks"]) < 3)
    n_empty = sum(1 for r in allrec if not r["marks"])
    if not ok:
        raise SystemExit("no streams scored -- server too slow or refusing")

    for r in ok:
        r["admit_ms"] = r["admit_s"] * 1000.0
        r["fc_ms"] = r["ttfb_s"] * 1000.0
        r["sc_ms"] = r["sc_s"] * 1000.0 if r["sc_s"] is not None else float("nan")
        r["gaps_ms"] = [g * 1000.0 for g in r["gaps"]]
        # Keep last-audio time and full client lifetime separate for tail diagnostics.
        r["audio_end_s"] = r["marks"][-1]
        r["wall_s"] = r["t_end"] - r["t_send"]
        r["tail_s"] = r["wall_s"] - r["audio_end_s"]
        r["rtf"] = r["audio_end_s"] / r["audio_s"] if r["audio_s"] else float("nan")
        r["bubbles"] = [
            g for g in r["gaps_ms"] if g > CHUNK_AUDIO_MS
        ]  # gaps longer than one chunk of audio
        r["stall_ms"] = sum(g - CHUNK_AUDIO_MS for g in r["bubbles"])

    # All overlapping client requests contribute to C, including unfinished ones.
    area = sum(end - start for start, end in _overlaps(allrec, cut, measurement_end))
    c_meas = area / (measurement_end - cut) if measurement_end > cut else 0.0

    os.makedirs(outdir, exist_ok=True)
    rows = _dump_user_rows(ok, users)
    tsv = _write_dump_tsvs(
        ok, rows, outdir, users, t_start, allrec, cut, measurement_end
    )

    # Arrival-cohort estimate using the configured arrival-window duration.
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


# ------------------------------------------------------------------------- ladder


def ladder(args: argparse.Namespace, tgt: Target) -> list[dict]:
    """Run ``dump`` at each user count and summarize the results."""
    levels = sorted({int(x) for x in args.users.split(",") if x.strip()})
    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for users in levels:
        logger.info("=== %d users %s ===", users, time.strftime("%H:%M:%S"))
        summary = dump(
            tgt,
            users,
            args.duration,
            args.skip,
            os.path.join(args.outdir, f"dump_u{users}"),
            args.warm_requests,
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
        help="Approximate input tokens; overrides --text-chars.",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=3.46,
        help="Characters per estimated input token.",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Client audio-chunk cap; 0 runs to completion.",
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
        help="Latency metric: FC, SC, TC, or all charts.",
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
        help="Mark chart points with failed requests.",
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
        help="Load model: closed workers, open arrivals, or one burst.",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Open-loop requests/s; overrides concurrency/service-time calibration.",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="Open-loop cap; 0 uses 4x configured concurrency.",
    )
    p.add_argument(
        "--on-full",
        choices=["shed", "block"],
        default="shed",
        help="At the open-loop cap: shed arrivals or wait.",
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
        try:
            doc = sweep(args, tgt)
        except PreflightError as exc:
            raise SystemExit(str(exc)) from None
        if not args.no_table:
            print_report(doc, args)
    elif args.command == "dump":
        dump(tgt, args.users, args.duration, args.skip, args.outdir, args.warm_requests)
    elif args.command == "ladder":
        ladder(args, tgt)


if __name__ == "__main__":
    main()


__all__ = [
    "PreflightError",
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
