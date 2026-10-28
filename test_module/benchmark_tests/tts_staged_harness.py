# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Staged TTS load test that reproduces the input mix of a captured run.

Recreates the staged run captured on tts2-110a-q3, 2026-10-27 21:15-21:47 UTC:

    warmup  rounds of simultaneous requests; each round drains before the next
    ladder  closed loop at 1,2,4,...,256 workers for 120 s per level; each level
            drains completely before the next one starts
    probe   one request at a time across every voice size, cancelled after two
            chunks, to measure first-chunk latency against prompt length

Each ladder request draws a text and a voice reference independently at random
(seeded) with the counts observed in the capture, so the ISL distribution
(text tokens + voice speech tokens) matches it. The capture showed no correlation
between the two, so independent draws reproduce the joint distribution too.

    python tts_staged_harness.py build-dataset --capture tts2-110a-q3-....jsonl \\
        --start 2026-10-27T21:15:00 --end 2026-10-27T21:48:00
    python tts_staged_harness.py run --host d08u08 --out staged.json
    python tts_staged_harness.py report --results staged.json

Voice references:
  The capture records only each voice prompt's length in speech tokens, not its
  audio. ``run`` therefore needs one WAV per voice size, ``<voice-dir>/voice_<N>.wav``.
  Missing files are cut from the recordings in ``--voice-source`` or, without that
  option, from speech the server under test generates with no voice prompt. Clips
  are written to ``--voice-dir``, so later runs reuse identical audio. Clip length
  is N / ``--speech-ids-per-s`` seconds. The default 50/s assumes the voice encoder
  runs at the output codec's rate (960 samples at 48 kHz per token); check the
  server's prompt token counts if that matters.
"""

from __future__ import annotations

import argparse
import array
import collections
import datetime as dt
import http.client
import io
import json
import logging
import math
import os
import random
import sys
import threading
import time
import wave

# Support both direct execution and package imports.
if __package__:
    from . import tts_load_harness as harness
else:
    import tts_load_harness as harness

pct = harness.pct

logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datasets",
    "tts_staged_2026-10-27.json",
)
DEFAULT_LEVELS = "1,2,4,8,16,32,64,128,256"
LEVEL_DURATION_S = 120.0
VOICE_RATE_HZ = 16000
# Speech prompt tokens wrap in <|audio_prompt_start|> ... <|audio_prompt_end|>.
VOICE_WRAP_TOKENS = 2
# Output audio cap observed on the server: 6000 tokens = 120 s.
OUTPUT_CAP_S = 119.9
# Runaway heuristic: audio well past what the text needs (~0.07 s per character).
RUNAWAY_S_PER_CHAR = 0.07
RUNAWAY_FACTOR = 3.0
RUNAWAY_SLACK_S = 5.0
MAX_GAPS = harness.MAX_TC_GAPS


def isl_tokens(text_tokens: int, voice_ids: int) -> int:
    return text_tokens + voice_ids + (VOICE_WRAP_TOKENS if voice_ids else 0)


def is_runaway(audio_s: float, chars: int) -> bool:
    expected = RUNAWAY_S_PER_CHAR * chars
    return audio_s > RUNAWAY_FACTOR * expected + RUNAWAY_SLACK_S


# ---------------------------------------------------------------- dataset build


def _parse_utc(value: str) -> int:
    t = dt.datetime.fromisoformat(value)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1e6)


def _segments(rows: list, min_idle_s: float) -> list:
    """Split rows wherever nothing was in flight for at least ``min_idle_s``."""
    out, cur, busy_until = [], [], None
    for r in rows:
        start = r["arrival_wall_us"]
        if cur and start - busy_until >= min_idle_s * 1e6:
            out.append(cur)
            cur = []
        cur.append(r)
        end = start + r["total_us"]
        busy_until = end if busy_until is None else max(busy_until, end)
    if cur:
        out.append(cur)
    return out


def _text_tokens(r: dict) -> int:
    ids = r["prompt_speech_ids"]
    return r["prompt_tokens"] - ids - (VOICE_WRAP_TOKENS if ids else 0)


def _stage_stats(rows: list) -> dict:
    span = (rows[-1]["arrival_wall_us"] - rows[0]["arrival_wall_us"]) / 1e6
    audio = [r["audio_samples"] / r["sample_rate_hz"] for r in rows]
    fc = [r["ttfc_us"] / 1e3 for r in rows if r["chunks"]]
    gaps = [
        (o[i + 1] - o[i]) / 1e3
        for r in rows
        for o in [r["chunk_offsets_us"]]
        for i in range(1, len(o) - 1)
    ]
    isl = [r["prompt_tokens"] for r in rows]
    return {
        "n": len(rows),
        "cancelled": sum(r["outcome"] != "completed" for r in rows),
        "max_inflight": max(r["inflight_on_arrival"] for r in rows) + 1,
        "arrival_span_s": round(span, 1),
        "req_per_s": round(len(rows) / span, 2) if span > 0 else None,
        "isl_p50": round(pct(isl, 50)),
        "isl_p90": round(pct(isl, 90)),
        "isl_max": max(isl),
        "fc_p50_ms": round(pct(fc, 50), 1),
        "fc_p95_ms": round(pct(fc, 95), 1),
        "fc_p99_ms": round(pct(fc, 99), 1),
        "tc_p50_ms": round(pct(gaps, 50), 1),
        "audio_p50_s": round(pct(audio, 50), 2),
        "capped_pct": round(100 * sum(a >= OUTPUT_CAP_S for a in audio) / len(rows), 1),
        "runaway_pct": round(
            100
            * sum(is_runaway(a, r["text_chars"]) for a, r in zip(audio, rows))
            / len(rows),
            1,
        ),
    }


def _ladder_stages(rows: list, levels: list, duration_s: float) -> list:
    """Replay the closed-loop schedule over the capture to attribute rows to levels.

    A level opens at its first arrival and accepts arrivals for ``duration_s``; the
    next level opens at the first arrival after every earlier request finished.
    """
    stages, i = [], 0
    for level in levels:
        if i >= len(rows):
            raise ValueError(f"capture ran out of requests before level {level}")
        start = rows[i]["arrival_wall_us"]
        j = i
        while j < len(rows) and rows[j]["arrival_wall_us"] < start + duration_s * 1e6:
            j += 1
        stage = rows[i:j]
        drained = max(r["arrival_wall_us"] + r["total_us"] for r in stage)
        # Arrivals that slipped in before the drain finished still belong here.
        while j < len(rows) and rows[j]["arrival_wall_us"] < drained - 500_000:
            stage.append(rows[j])
            j += 1
        seen = max(r["inflight_on_arrival"] for r in stage) + 1
        if seen > level:
            raise ValueError(
                f"level {level}: {seen} requests in flight -- schedule does not fit"
            )
        stages.append({"concurrency": level, "captured": _stage_stats(stage)})
        i = j
    if i != len(rows):
        raise ValueError(f"{len(rows) - i} ladder requests left after the last level")
    return stages


def build_dataset(
    capture: str,
    start: str,
    end: str,
    levels: list,
    duration_s: float = LEVEL_DURATION_S,
    min_idle_s: float = 10.0,
) -> dict:
    lo, hi = _parse_utc(start), _parse_utc(end)
    with open(capture) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = sorted(
        (r for r in rows if lo <= r["arrival_wall_us"] < hi),
        key=lambda r: r["arrival_wall_us"],
    )
    phases = _segments(rows, min_idle_s)
    if len(phases) != 3:
        raise ValueError(
            f"expected warmup, ladder and probe separated by idle gaps, "
            f"found {len(phases)} segments"
        )
    warm, ladder, probe = phases

    tokens: dict = {}
    for r in rows:
        tokens.setdefault(r["text"], collections.Counter())[_text_tokens(r)] += 1
    text_tokens = {t: c.most_common(1)[0][0] for t, c in tokens.items()}

    texts = collections.Counter(r["text"] for r in ladder)
    voices = collections.Counter(r["prompt_speech_ids"] for r in ladder)
    cancel_chunks = collections.Counter(
        r["chunks"] for r in probe if r["outcome"] != "completed"
    )

    def spec(r: dict) -> dict:
        return {
            "text": r["text"],
            "text_tokens": text_tokens[r["text"]],
            "voice_speech_ids": r["prompt_speech_ids"],
        }

    return {
        "source": {
            "capture": os.path.basename(capture),
            "start": start,
            "end": end,
            "requests": len(rows),
        },
        "warmup": [
            {"requests": [spec(r) for r in rnd], "captured": _stage_stats(rnd)}
            for rnd in _segments(warm, 0.0)
        ],
        "ladder": {
            "duration_s": duration_s,
            "stages": _ladder_stages(ladder, levels, duration_s),
            "texts": [
                {"text": t, "text_tokens": text_tokens[t], "count": n}
                for t, n in sorted(texts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "voices": [
                {"voice_speech_ids": v, "count": n} for v, n in sorted(voices.items())
            ],
        },
        "probe": {
            "cancel_after_chunks": cancel_chunks.most_common(1)[0][0]
            if cancel_chunks
            else 0,
            "requests": [spec(r) for r in probe],
            "captured": _stage_stats(probe),
        },
    }


# ------------------------------------------------------------------ voice clips


def _wav_bytes(pcm: array.array) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(VOICE_RATE_HZ)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _to_voice_rate(pcm: array.array, rate: int, channels: int) -> array.array:
    """Downmix to mono and linearly resample to VOICE_RATE_HZ."""
    if channels > 1:
        pcm = array.array(
            "h",
            (
                sum(pcm[i : i + channels]) // channels
                for i in range(0, len(pcm) - channels + 1, channels)
            ),
        )
    if rate == VOICE_RATE_HZ or not pcm:
        return pcm
    n = int(len(pcm) * VOICE_RATE_HZ / rate)
    out = array.array("h", bytes(2 * n))
    step = rate / VOICE_RATE_HZ
    last = len(pcm) - 1
    for i in range(n):
        pos = i * step
        k = min(int(pos), last)
        frac = pos - k
        out[i] = int(pcm[k] + (pcm[min(k + 1, last)] - pcm[k]) * frac)
    return out


def _read_wav(path: str) -> array.array:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        pcm = array.array("h", w.readframes(w.getnframes()))
        return _to_voice_rate(pcm, w.getframerate(), w.getnchannels())


def _speak(tgt: harness.Target, text: str) -> array.array:
    """Generate speech with no voice prompt and return it at VOICE_RATE_HZ."""
    conn = http.client.HTTPConnection(tgt.host, tgt.port, timeout=tgt.timeout)
    try:
        conn.request(
            "POST",
            tgt.path,
            body=json.dumps({"text": text}),
            headers={
                "Authorization": f"Bearer {tgt.key}",
                "Content-Type": "application/json",
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"voice bootstrap: http {resp.status}: {body[:200]!r}")
    finally:
        conn.close()
    audio = body[harness.WAV_HEADER_BYTES :]
    pcm = array.array("h", audio[: len(audio) // 2 * 2])
    return _to_voice_rate(pcm, harness.SAMPLE_RATE, 1)


class VoiceBank:
    """One reference WAV per voice size, created on first use and kept on disk."""

    def __init__(self, voice_dir: str, ids_per_s: float) -> None:
        self.voice_dir = voice_dir
        self.ids_per_s = ids_per_s
        self.clips: dict = {}

    def path(self, ids: int) -> str:
        return os.path.join(self.voice_dir, f"voice_{ids}.wav")

    def ensure(
        self,
        sizes: list,
        source_dir: str | None = None,
        tgt: harness.Target | None = None,
        bootstrap_text: str = "",
    ) -> None:
        sizes = sorted({s for s in sizes if s > 0})
        missing = [s for s in sizes if not os.path.exists(self.path(s))]
        if missing:
            os.makedirs(self.voice_dir, exist_ok=True)
            longest = int(max(missing) / self.ids_per_s * VOICE_RATE_HZ)
            source = self._source(longest, source_dir, tgt, bootstrap_text)
            for i, ids in enumerate(missing):
                need = int(ids / self.ids_per_s * VOICE_RATE_HZ)
                # Different offsets keep the clips distinct beyond their length.
                room = len(source) - need
                off = (i * 7919 * 37) % room if room > 0 else 0
                with open(self.path(ids), "wb") as f:
                    f.write(_wav_bytes(source[off : off + need]))
            logger.info("voice clips: wrote %d to %s", len(missing), self.voice_dir)
        for s in sizes:
            with open(self.path(s), "rb") as f:
                self.clips[s] = f.read()

    def _source(
        self,
        need: int,
        source_dir: str | None,
        tgt: harness.Target | None,
        text: str,
    ) -> array.array:
        pcm = array.array("h")
        if source_dir:
            for name in sorted(os.listdir(source_dir)):
                if name.lower().endswith(".wav"):
                    pcm.extend(_read_wav(os.path.join(source_dir, name)))
            if not pcm:
                raise SystemExit(f"no 16-bit WAV files in {source_dir}")
        else:
            if tgt is None:
                raise SystemExit("voice clips missing and no server to bootstrap from")
            logger.info(
                "voice clips: generating %.0f s of reference speech",
                need / VOICE_RATE_HZ,
            )
            for _ in range(8):
                pcm.extend(_speak(tgt, text))
                if len(pcm) >= need:
                    break
        while len(pcm) < need:  # loop short sources rather than fail
            pcm.extend(pcm[: need - len(pcm)])
        return pcm

    def get(self, ids: int) -> bytes:
        return self.clips[ids] if ids else b""


# ------------------------------------------------------------------- execution


class Sampler:
    """Seeded independent draws of a text and a voice size, weighted by count."""

    def __init__(self, ladder: dict, seed: int) -> None:
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.texts = ladder["texts"]
        self.text_w = [t["count"] for t in self.texts]
        self.voices = [v["voice_speech_ids"] for v in ladder["voices"]]
        self.voice_w = [v["count"] for v in ladder["voices"]]

    def draw(self) -> dict:
        with self.lock:
            t = self.rng.choices(self.texts, self.text_w)[0]
            v = self.rng.choices(self.voices, self.voice_w)[0]
        return {
            "text": t["text"],
            "text_tokens": t["text_tokens"],
            "voice_speech_ids": v,
        }


class Runner:
    def __init__(self, base: dict, bank: VoiceBank) -> None:
        self.base = base
        self.bank = bank
        self.records: list = []
        self.lock = threading.Lock()

    def _send(self, spec: dict, stage: str, conc: int, max_chunks: int = 0) -> None:
        tgt = harness.Target(
            text=spec["text"],
            voice_wav=self.bank.get(spec["voice_speech_ids"]),
            max_chunks=max_chunks,
            **self.base,
        )
        rec = {
            "stage": stage,
            "conc": conc,
            "voice_speech_ids": spec["voice_speech_ids"],
            "isl": isl_tokens(spec["text_tokens"], spec["voice_speech_ids"]),
            "t_send": time.time(),
            "t_end": None,
        }
        with self.lock:
            self.records.append(rec)  # registered BEFORE the send, never vanishes
        harness.request(tgt, -1, rec)
        rec["gaps"] = [round(g, 6) for g in rec["gaps"][:MAX_GAPS]]
        rec.pop("marks", None)
        rec.pop("uid", None)

    def burst(self, specs: list, stage: str) -> None:
        gate = threading.Event()

        def fire(spec: dict) -> None:
            gate.wait()
            self._send(spec, stage, len(specs))

        threads = [threading.Thread(target=fire, args=(s,), daemon=True) for s in specs]
        for t in threads:
            t.start()
        time.sleep(0.2)
        gate.set()
        self._join(threads)

    def closed(self, sampler: Sampler, conc: int, duration: float, stage: str) -> None:
        stop = threading.Event()

        def worker() -> None:
            while not stop.is_set():
                self._send(sampler.draw(), stage, conc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
        for t in threads:
            t.start()
        time.sleep(duration)
        stop.set()
        self._join(threads)  # drain: the next level starts only once this one is idle

    def sequential(self, specs: list, stage: str, max_chunks: int) -> None:
        for spec in specs:
            self._send(spec, stage, 1, max_chunks)

    def _join(self, threads: list) -> None:
        deadline = time.time() + self.base["timeout"] + 5
        for t in threads:
            t.join(max(0, deadline - time.time()))


def run(args: argparse.Namespace) -> dict:
    with open(args.dataset) as f:
        ds = json.load(f)
    ladder = ds["ladder"]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    duration = args.duration if args.duration else ladder["duration_s"]
    phases = set(args.phases.split(","))
    base = {
        "host": args.host,
        "port": args.port,
        "path": args.path,
        "key": args.key,
        "timeout": args.timeout,
    }

    sizes = [v["voice_speech_ids"] for v in ladder["voices"]]
    sizes += [s["voice_speech_ids"] for rnd in ds["warmup"] for s in rnd["requests"]]
    sizes += [s["voice_speech_ids"] for s in ds["probe"]["requests"]]
    bank = VoiceBank(args.voice_dir, args.speech_ids_per_s)
    bootstrap = " ".join(t["text"] for t in ladder["texts"] if t["text_tokens"] > 40)
    bank.ensure(
        sizes,
        args.voice_source,
        harness.Target(**base),
        bootstrap[: args.bootstrap_chars],
    )

    runner = Runner(base, bank)
    stages: list = []
    doc = {
        "meta": {
            "dataset": os.path.basename(args.dataset),
            "source": ds["source"],
            "host": args.host,
            "port": args.port,
            "seed": args.seed,
            "levels": levels,
            "duration_s": duration,
            "speech_ids_per_s": args.speech_ids_per_s,
            "voice_dir": os.path.abspath(args.voice_dir),
            "started": time.time(),
        },
        "stages": stages,
        "records": runner.records,
    }

    def checkpoint(
        name: str, kind: str, conc: int, t0: float, captured: dict | None
    ) -> None:
        stages.append(
            {
                "stage": name,
                "kind": kind,
                "conc": conc,
                "start": t0,
                "end": time.time(),
                "captured": captured,
            }
        )
        n = sum(r["stage"] == name for r in runner.records)
        logger.info("[%s] done: %d requests in %.0f s", name, n, time.time() - t0)
        with open(args.out, "w") as f:
            json.dump(doc, f)

    if "warmup" in phases:
        for i, rnd in enumerate(ds["warmup"]):
            name = f"warmup{i + 1}"
            logger.info("[%s] %d simultaneous requests", name, len(rnd["requests"]))
            t0 = time.time()
            runner.burst(rnd["requests"], name)
            checkpoint(name, "burst", len(rnd["requests"]), t0, rnd["captured"])

    if "ladder" in phases:
        captured = {s["concurrency"]: s["captured"] for s in ladder["stages"]}
        sampler = Sampler(ladder, args.seed)
        for conc in levels:
            name = f"c{conc}"
            logger.info("[%s] closed loop, %d workers, %.0f s", name, conc, duration)
            t0 = time.time()
            runner.closed(sampler, conc, duration, name)
            checkpoint(name, "closed", conc, t0, captured.get(conc))

    if "probe" in phases:
        probe = ds["probe"]
        logger.info(
            "[probe] %d sequential requests, cancelled after %d chunks",
            len(probe["requests"]),
            probe["cancel_after_chunks"],
        )
        t0 = time.time()
        runner.sequential(probe["requests"], "probe", probe["cancel_after_chunks"])
        checkpoint("probe", "sequential", 1, t0, probe["captured"])

    doc["meta"]["finished"] = time.time()
    with open(args.out, "w") as f:
        json.dump(doc, f)
    logger.info("wrote %s (%d requests)", args.out, len(runner.records))
    return doc


# ---------------------------------------------------------------------- report


def stage_stats(recs: list) -> dict:
    ok = [r for r in recs if r["ok"]]
    t0 = min(r["t_send"] for r in recs)
    span = max(r["t_send"] for r in recs) - t0
    fc = [r["ttfb_s"] * 1e3 for r in ok if r["ttfb_s"] is not None]
    gaps = [g * 1e3 for r in ok for g in r["gaps"]]
    audio = [r["audio_s"] for r in ok]
    isl = [r["isl"] for r in recs]
    full = [r for r in ok if not r["capped"]]
    return {
        "n": len(recs),
        "errors": len(recs) - len(ok),
        "arrival_span_s": round(span, 1),
        "req_per_s": round(len(recs) / span, 2) if span > 0 else None,
        "isl_p50": round(pct(isl, 50)),
        "isl_p90": round(pct(isl, 90)),
        "isl_max": max(isl),
        "fc_p50_ms": round(pct(fc, 50), 1),
        "fc_p95_ms": round(pct(fc, 95), 1),
        "fc_p99_ms": round(pct(fc, 99), 1),
        "tc_p50_ms": round(pct(gaps, 50), 1),
        "audio_p50_s": round(pct(audio, 50), 2),
        "capped_pct": round(
            100 * sum(r["audio_s"] >= OUTPUT_CAP_S for r in full) / max(1, len(full)), 1
        ),
        "runaway_pct": round(
            100
            * sum(is_runaway(r["audio_s"], r["chars"]) for r in full)
            / max(1, len(full)),
            1,
        ),
    }


COLS = [
    ("n", "n", "{:d}"),
    ("errors", "err", "{:d}"),
    ("req_per_s", "req/s", "{:.2f}"),
    ("isl_p50", "ISL p50", "{:.0f}"),
    ("isl_p90", "ISL p90", "{:.0f}"),
    ("isl_max", "ISL max", "{:.0f}"),
    ("fc_p50_ms", "FC p50", "{:.0f}"),
    ("fc_p95_ms", "FC p95", "{:.0f}"),
    ("fc_p99_ms", "FC p99", "{:.0f}"),
    ("tc_p50_ms", "TC p50", "{:.0f}"),
    ("audio_p50_s", "audio p50", "{:.1f}"),
    ("capped_pct", "120s %", "{:.1f}"),
    ("runaway_pct", "runaway %", "{:.1f}"),
]


def _fmt(value, spec: str) -> str:
    if value is None or math.isnan(value):
        return "-"
    return spec.format(value)


def report_table(doc: dict) -> str:
    by_stage = collections.defaultdict(list)
    for r in doc["records"]:
        by_stage[r["stage"]].append(r)
    head = ["stage", "conc", "src"] + [c[1] for c in COLS]
    rows = [head]
    for s in doc["stages"]:
        recs = by_stage.get(s["stage"])
        if not recs:
            continue
        got = stage_stats(recs)
        rows.append(
            [s["stage"], str(s["conc"]), "run"]
            + [_fmt(got.get(k), f) for k, _, f in COLS]
        )
        cap = s.get("captured")
        if cap:
            cap = dict(cap, errors=0)
            rows.append(["", "", "capture"] + [_fmt(cap.get(k), f) for k, _, f in COLS])
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    notes = (
        "FC = first chunk, TC = later chunk gaps (ms). ISL = prompt tokens "
        "(text + voice + 2). 120s % = hit the output cap; runaway % = audio > "
        f"{RUNAWAY_FACTOR:g} x {RUNAWAY_S_PER_CHAR} s/char + {RUNAWAY_SLACK_S:g} s. "
        "Capture req/s is over its arrival span; capture 'err' is not recorded."
    )
    return "\n".join(lines) + "\n\n" + notes


def probe_table(doc: dict) -> str:
    groups = collections.defaultdict(list)
    for r in doc["records"]:
        if r["stage"] == "probe" and r["ttfb_s"] is not None:
            groups[r["isl"]].append(r["ttfb_s"] * 1e3)
    if not groups:
        return ""
    lines = ["probe: first chunk vs ISL", "     ISL   n   FC mean ms"]
    for isl in sorted(groups):
        v = groups[isl]
        lines.append(f"  {isl:6d} {len(v):3d}   {sum(v) / len(v):8.1f}")
    return "\n".join(lines)


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-dataset", help="capture JSONL -> dataset JSON")
    p.add_argument("--capture", required=True, help="server capture JSONL")
    p.add_argument("--start", required=True, help="UTC ISO time, e.g. 2026-10-27T21:15")
    p.add_argument("--end", required=True, help="UTC ISO time, exclusive")
    p.add_argument("--levels", default=DEFAULT_LEVELS)
    p.add_argument("--duration", type=float, default=LEVEL_DURATION_S)
    p.add_argument("--out", default=DEFAULT_DATASET)

    p = sub.add_parser("run", help="run the staged test -> results JSON")
    p.add_argument("--host", default=os.environ.get("TTS_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8000")))
    p.add_argument("--path", default=harness.DEFAULT_PATH)
    p.add_argument("--key", default=os.environ.get("TTS_KEY", harness.DEFAULT_KEY))
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--levels", default=DEFAULT_LEVELS)
    p.add_argument(
        "--duration", type=float, default=0, help="seconds per level; 0 = dataset's"
    )
    p.add_argument(
        "--phases", default="warmup,ladder,probe", help="comma list of phases to run"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--voice-dir", default="tts_voices")
    p.add_argument(
        "--voice-source",
        default=None,
        help="directory of 16-bit WAV recordings to cut voice clips from "
        "(default: generate them on the server under test)",
    )
    p.add_argument("--speech-ids-per-s", type=float, default=50.0)
    p.add_argument("--bootstrap-chars", type=int, default=1600)
    p.add_argument("--out", default="tts_staged_results.json")
    p.add_argument("--no-table", action="store_true")

    p = sub.add_parser("report", help="tables from a run's results JSON")
    p.add_argument("--results", required=True)
    return ap


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.command == "build-dataset":
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        ds = build_dataset(args.capture, args.start, args.end, levels, args.duration)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(ds, f, indent=1)
            f.write("\n")
        lad = ds["ladder"]
        logger.info(
            "wrote %s: %d texts, %d voices, %d warmup rounds, %d probe requests",
            args.out,
            len(lad["texts"]),
            len(lad["voices"]),
            len(ds["warmup"]),
            len(ds["probe"]["requests"]),
        )
        return

    if args.command == "report":
        with open(args.results) as f:
            doc = json.load(f)
    else:
        doc = run(args)
        if args.no_table:
            return
    print(report_table(doc))
    probe = probe_table(doc)
    if probe:
        print("\n" + probe)


if __name__ == "__main__":
    main()
