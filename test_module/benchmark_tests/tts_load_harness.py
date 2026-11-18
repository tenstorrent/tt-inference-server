# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Chunk-aware closed-loop load generator for a streaming TTS server.

``tts_benchmark_tests.run_tts_load_benchmark`` drives it. Per concurrency level,
``conc`` workers each send their next request as soon as the previous one ends,
for ``warmup + duration`` seconds; only requests sent after the warmup count.

Measurement definitions:
  * FC is request start to the first audio chunk, SC the gap to the second, TC
    the gap to the third, and 4C+ every later gap. One HTTP chunk is one audio
    chunk (see ``tts_client.stream_speech``).
  * C is the average number of requests in flight over the window, from request
    lifetimes clipped to it, warmup carryover and unfinished requests included.
  * Throughput counts the successful requests SENT in the window (the arrival
    cohort) and follows them through the drain; capped responses get no
    character credit.
  * The sweep JSON keeps the first ``MAX_TC_GAPS`` gaps per request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics as st
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

from ..eval_tests.tts_client import auth_headers, stream_speech, tt_tts_payload

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
SPEECH_PATH = "/v1/audio/speech"
MAX_TC_GAPS = 200  # cap the per-request gap list so a sweep's JSON stays manageable
PREFLIGHT_TRIES = 3

# metric key -> (per-request field, slice of it). TC and 4C+ share the gap list:
# TC is the 3rd-chunk gap and 4C+ every later one, so the 3rd chunk's settle cost
# is scored apart from the sustained cadence. Every gap is one sample.
METRICS = {
    "ttfb": ("ttfb_s", None),
    "ttfs": ("sc_s", None),
    "ttft": ("tc_gaps", slice(0, 1)),
    "ttf4": ("tc_gaps", slice(1, None)),
}


class PreflightError(RuntimeError):
    """The target server did not produce a valid response during preflight."""


@dataclass(frozen=True)
class Target:
    """Where to send a request and what to send."""

    url: str
    text: str = ""
    speech_ids: list = field(default_factory=list, repr=False)
    timeout: float = 300.0

    def with_prompt(self, text: str, speech_ids: list) -> "Target":
        return Target(self.url, text, speech_ids, self.timeout)


@dataclass(frozen=True)
class SweepConfig:
    concurrency: tuple[int, ...]
    duration: float
    warmup: float
    settle: float = 5.0
    # Spread each level's worker starts uniformly over [0, ramp) s.
    ramp: float = 0.0
    # Discarded burst before the sweep: prewarm_concurrency workers for prewarm s.
    prewarm: float = 0.0
    prewarm_concurrency: int = 0
    out: Path | None = None


Draw = Callable[[], Target]


def build_text(nchars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    return (base * (nchars // len(base) + 1))[:nchars]


def _session() -> aiohttp.ClientSession:
    # limit=0: the default 100-connection pool would cap concurrency silently.
    return aiohttp.ClientSession(
        headers=auth_headers(), connector=aiohttp.TCPConnector(limit=0)
    )


async def request(
    session: aiohttp.ClientSession,
    tgt: Target,
    rec: dict | None = None,
    body: bytearray | None = None,
) -> dict:
    rec = rec if rec is not None else {"t_send": time.time()}
    rec["chars"] = len(tgt.text)
    rec["voice_speech_ids"] = len(tgt.speech_ids)
    return await stream_speech(
        session,
        tgt.url,
        tt_tts_payload(tgt.text, tgt.speech_ids),
        sample_rate=SAMPLE_RATE,
        timeout_s=tgt.timeout,
        rec=rec,
        body=body,
    )


async def _workers(
    session: aiohttp.ClientSession,
    tgt: Target,
    conc: int,
    run_s: float,
    draw: Draw | None,
    offsets: list[float],
) -> tuple[list[dict], float, float]:
    """Run ``conc`` closed-loop workers for ``run_s`` s, then drain (bounded).

    Every record is registered before its send, so a request still in flight at
    the drain deadline keeps its place (and its lifetime) in the results.
    """
    out: list[dict] = []
    stop = asyncio.Event()

    async def worker(offset: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), offset)
        except asyncio.TimeoutError:
            pass
        while not stop.is_set():
            rec = {"t_send": time.time()}
            out.append(rec)
            await request(session, draw() if draw else tgt, rec)

    t_start = time.time()
    tasks = [asyncio.create_task(worker(o)) for o in offsets[:conc]]
    await asyncio.sleep(run_s)
    stop.set()
    t_stop = time.time()
    _, pending = await asyncio.wait(tasks, timeout=tgt.timeout + 5)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return out, t_start, t_stop


async def run_closed(
    session: aiohttp.ClientSession,
    tgt: Target,
    conc: int,
    duration: float,
    warmup: float,
    draw: Draw | None = None,
    ramp: float = 0.0,
) -> tuple[list[dict], float, float]:
    """One closed-loop level; returns (records, window start, window end)."""
    rng = random.Random(f"ramp-{conc}")
    offsets = [rng.uniform(0, ramp) if ramp > 0 else 0.0 for _ in range(conc)]
    logger.info("  warmup %.0fs @ conc=%d ...", warmup, conc)
    recs, t_start, t_stop = await _workers(
        session, tgt, conc, warmup + duration, draw, offsets
    )
    return recs, t_start + warmup, t_stop


async def _prewarm(
    session: aiohttp.ClientSession,
    tgt: Target,
    conc: int,
    duration: float,
    draw: Draw | None,
) -> int:
    if conc <= 0 or duration <= 0:
        return 0
    logger.info("prewarm: %d concurrent requests for %.0fs (discarded)", conc, duration)
    recs, _, _ = await _workers(session, tgt, conc, duration, draw, [0.0] * conc)
    done = sum(1 for r in recs if r.get("t_end") is not None)
    logger.info("prewarm: done, %d requests completed", done)
    return done


def prewarm(tgt: Target, conc: int, duration: float, draw: Draw | None = None) -> int:
    """Burn off TTS cold start (the first requests can take >10 s to first audio):
    run ``conc`` closed-loop workers for ``duration`` s and discard the results."""

    async def main() -> int:
        async with _session() as session:
            return await _prewarm(session, tgt, conc, duration, draw)

    return asyncio.run(main())


async def _preflight(session: aiohttp.ClientSession, tgt: Target) -> dict:
    """Fail fast on a dead server; retried, since a healthy one can rarely return
    an empty (header-only) WAV."""
    probe: dict = {}
    for attempt in range(1, PREFLIGHT_TRIES + 1):
        probe = await request(session, tgt)
        if probe["ok"]:
            logger.info(
                "preflight ok (ttfb %.0f ms, %d chunks)",
                probe["ttfb_s"] * 1000,
                probe["nchunks"],
            )
            return probe
        logger.warning(
            "preflight attempt %d/%d failed: %s",
            attempt,
            PREFLIGHT_TRIES,
            probe["error"],
        )
    raise PreflightError(
        f"preflight failed {PREFLIGHT_TRIES}x: {probe['error']} "
        "-- is the server up and healthy?"
    )


def _serializable(rec: dict, conc: int) -> dict:
    out = {k: v for k, v in rec.items() if k not in ("marks", "gaps")}
    out["conc"] = conc
    out["tc_gaps"] = [round(g, 6) for g in rec.get("gaps", [])[:MAX_TC_GAPS]]
    return out


def _write(doc: dict, out: Path | None) -> None:
    if out is not None:
        with open(out, "w") as f:
            json.dump(doc, f)


def sweep(cfg: SweepConfig, tgt: Target, draw: Draw | None = None) -> dict:
    """Run every level (in the order given) and return the results document.

    ``draw`` gives each prewarm and measured request its own prompt; ``tgt`` also
    serves the preflight. The document is checkpointed to ``cfg.out`` per level.
    """
    return asyncio.run(_sweep(cfg, tgt, draw))


async def _sweep(cfg: SweepConfig, tgt: Target, draw: Draw | None) -> dict:
    levels = list(dict.fromkeys(cfg.concurrency))
    logger.info("target %s  levels=%s", tgt.url, levels)
    async with _session() as session:
        # Before the preflight too, so nothing measured sees a cold server.
        prewarmed = await _prewarm(
            session, tgt, cfg.prewarm_concurrency, cfg.prewarm, draw
        )
        await _preflight(session, tgt)
        meta = {
            "url": tgt.url,
            "chars_per_request": len(tgt.text),
            "duration_s": cfg.duration,
            "warmup_s": cfg.warmup,
            "prewarm_requests": prewarmed,
            "started": time.time(),
        }
        all_recs: list[dict] = []
        levels_meta: list[dict] = []
        doc = {"meta": meta, "levels": levels_meta, "records": all_recs}
        for conc in levels:
            logger.info("[conc=%d] running %.0fs ...", conc, cfg.duration)
            recs, w0, w1 = await run_closed(
                session, tgt, conc, cfg.duration, cfg.warmup, draw, cfg.ramp
            )
            in_win = _arrival_cohort(recs, w0, w1)
            ok = [r for r in in_win if r["ok"]]
            ttfbs = [r["ttfb_s"] * 1000 for r in ok if r["ttfb_s"] is not None]
            logger.info(
                "[conc=%d] %d/%d ok  rps=%.2f  ttfb_p50=%.0fms",
                conc,
                len(ok),
                len(in_win),
                len(ok) / (w1 - w0) if w1 > w0 else 0.0,
                pct(ttfbs, 50),
            )
            all_recs += [_serializable(r, conc) for r in recs]
            levels_meta.append(
                {"conc": conc, "window_start": w0, "window_end": w1, "n_ok": len(ok)}
            )
            _write(doc, cfg.out)
            if conc != levels[-1]:
                await asyncio.sleep(cfg.settle)
    if draw and all_recs:
        meta["chars_per_request"] = round(st.mean(r["chars"] for r in all_recs))
    meta["finished"] = time.time()
    _write(doc, cfg.out)
    logger.info("wrote %s (%d records)", cfg.out, len(all_recs))
    return doc


# ------------------------------------------------------------------ aggregation


def pct(vals, p: float) -> float:
    """Linear-interpolated percentile; NaN for no samples."""
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = (len(v) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _metric_samples(ok_recs: list[dict], key: str) -> list[float]:
    """Milliseconds for one metric; every gap in the slice is its own sample, so
    one long request cannot hide many late chunks."""
    name, gap_slice = METRICS[key]
    out: list[float] = []
    for r in ok_recs:
        v = r.get(name)
        if v is None:
            continue
        if isinstance(v, list):
            out.extend(x * 1000.0 for x in (v[gap_slice] if gap_slice else v))
        else:
            out.append(v * 1000.0)
    return out


def _arrival_cohort(recs: list[dict], w0: float, w1: float) -> list[dict]:
    return [r for r in recs if w0 <= r["t_send"] < w1]


def _occupancy(recs: list[dict], w0: float, w1: float) -> float:
    """Average requests in flight, with every lifetime clipped to the window."""
    if not w1 > w0:
        return float("nan")
    events = []
    for r in recs:
        start, end = max(r["t_send"], w0), min(r.get("t_end") or w1, w1)
        if end > start:
            events += [(start, 1), (end, -1)]
    events.sort()
    area, cur, prev = 0.0, 0, w0
    for t, delta in events:
        area += cur * (t - prev)
        prev, cur = t, cur + delta
    area += cur * (w1 - prev)
    return area / (w1 - w0)


def aggregate(doc: dict) -> tuple[list[dict], int]:
    """Per level: occupancy, arrival-cohort throughput and latency percentiles."""
    chars = doc["meta"]["chars_per_request"]
    by_conc: dict[int, list[dict]] = {}
    for r in doc["records"]:
        by_conc.setdefault(r["conc"], []).append(r)
    rows = []
    for lvl in doc["levels"]:
        conc, w0, w1 = lvl["conc"], lvl["window_start"], lvl["window_end"]
        recs = by_conc.get(conc, [])
        in_win = _arrival_cohort(recs, w0, w1)
        ok = [r for r in in_win if r.get("ok")]
        window = w1 - w0
        nchunks = [r["nchunks"] for r in ok if r.get("nchunks")]
        row = {
            "conc": conc,
            "n_ok": len(ok),
            "n_err": len(in_win) - len(ok),
            "rps": len(ok) / window if window > 0 else float("nan"),
            # Credit only responses that spoke the whole text (not client-capped).
            "mchar_h": (
                sum(r.get("chars") or chars for r in ok if not r.get("capped"))
                / window
                * 3600.0
                / 1e6
                if window > 0
                else float("nan")
            ),
            "C": _occupancy(recs, w0, w1),
            "chunks": sum(nchunks) / len(nchunks) if nchunks else float("nan"),
        }
        for key in METRICS:
            samples = _metric_samples(ok, key)
            for p in (50, 95):
                row[f"{key}_p{p}"] = pct(samples, p)
            row[f"{key}_max"] = max(samples) if samples else float("nan")
        rows.append(row)
    return rows, chars


__all__ = [
    "PreflightError",
    "SweepConfig",
    "Target",
    "aggregate",
    "build_text",
    "prewarm",
    "request",
    "sweep",
]
