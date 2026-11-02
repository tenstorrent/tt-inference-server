# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Correctness check for a running TTS server: one prompt, sent many times.

    run    REFS sequential reference requests, then REQUESTS copies of the same
           request at CONCURRENCY; saves every WAV, scores it, exit 1 on failure
    score  re-score an existing run directory (e.g. with another ASR model)

    python tts_correctness.py run --host d08u08 --requests 100 --concurrency 100 \\
        --speech-ids voice.json --outdir tts_correctness
    python tts_correctness.py score --outdir tts_correctness --asr-model distil-large-v3

Each response, reference or load, fails on either of:
  * no audio: HTTP error, transport error, or a header-only WAV.
  * transcript: ASR (faster-whisper) WER against the input text above
    ``--max-wer``. Substitutions point at garbled words, deletions at dropped or
    truncated audio, insertions at runaway output.

Duration (vs the reference median) and whether the PCM matches the references are
recorded per response for information only: the server samples, so slow but
intelligible speech is not a failure. References are sent with nothing else in
flight, so a failing reference means the server misbehaves without load too.
Only faster-whisper is non-stdlib, and only for the transcript check;
``--no-asr`` skips it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

if __package__:
    from . import tts_load_harness as harness
else:
    import tts_load_harness as harness

logger = logging.getLogger(__name__)

DEFAULT_TEXT = (
    "Hello, and thank you for calling. Your order has shipped and should arrive "
    "within three business days. If you have any questions about your delivery, "
    "please reply to this message and a member of our team will be happy to help."
)
RESULTS_FILE = "correctness.json"


# ------------------------------------------------------------------------- scoring


def normalize(text: str) -> list[str]:
    """Lowercase words with punctuation dropped ("It's done." -> ["its", "done"])."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", "")).split()


def word_errors(ref: list[str], hyp: list[str]) -> dict:
    """Levenshtein alignment of word lists -> substitution/deletion/insertion counts."""
    # cost[i][j] = (edits, subs, dels, ins) aligning ref[:i] with hyp[:j]
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            s, d, n = prev[j - 1], prev[j], cur[j - 1]
            cur.append(
                min(
                    (s[0] + 1, s[1] + 1, s[2], s[3]),
                    (d[0] + 1, d[1], d[2] + 1, d[3]),
                    (n[0] + 1, n[1], n[2], n[3] + 1),
                )
            )
        prev = cur
    edits, subs, dels, ins = prev[-1]
    return {
        "wer": edits / max(len(ref), 1),
        "sub": subs,
        "del": dels,
        "ins": ins,
    }


def transcribe(paths: list[str], model_name: str, threads: int) -> dict[str, str]:
    """ASR every WAV with faster-whisper on CPU: English, beam 5, temperature 0."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name, device="cpu", compute_type="int8", cpu_threads=threads
    )
    out = {}
    for n, path in enumerate(paths, 1):
        segments, _ = model.transcribe(
            path,
            language="en",
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        out[path] = " ".join(seg.text.strip() for seg in segments)
        if n % 20 == 0:
            logger.info("  transcribed %d/%d", n, len(paths))
    return out


def score(records: list[dict], text: str, args: argparse.Namespace) -> dict:
    """Mark every record pass/fail on its WER against ``text``; return a summary."""
    ref_words = normalize(text)
    if not args.no_asr:
        paths = [r["wav"] for r in records if r["ok"]]
        logger.info("Transcribing %d responses with %s", len(paths), args.asr_model)
        texts = transcribe(paths, args.asr_model, args.asr_threads)
        for r in records:
            r["transcript"] = texts.get(r["wav"]) if r["ok"] else None
            if r["transcript"] is not None:
                r.update(word_errors(ref_words, normalize(r["transcript"])))

    refs = [r for r in records if r["tag"] == "ref" and r["ok"]]
    if not refs:
        raise SystemExit("no successful reference request; nothing to compare with")
    ref_hashes = {r["sha256"] for r in refs}
    deterministic = len(refs) > 1 and len(ref_hashes) == 1
    ref_dur = st.median(r["audio_s"] for r in refs)
    wer_limit = None if args.no_asr else args.max_wer

    for r in records:
        reasons = []
        if not r["ok"]:
            reasons.append(r["error"] or "no audio")
        else:
            r["duration_ratio"] = r["audio_s"] / ref_dur if ref_dur else None
            r["pcm_matches_ref"] = r["sha256"] in ref_hashes
            if wer_limit is not None and r.get("wer", 0) > wer_limit:
                reasons.append(f"wer {r['wer']:.1%} > {wer_limit:.1%}")
        r["reasons"] = reasons
        r["pass"] = not reasons

    load = [r for r in records if r["tag"] == "load"]
    return {
        "references": len(refs),
        "reference_failures": sum(
            1 for r in records if r["tag"] == "ref" and not r["pass"]
        ),
        "deterministic": deterministic,
        "reference_duration_s": ref_dur,
        "wer_limit": wer_limit,
        "load_requests": len(load),
        "load_failures": sum(1 for r in load if not r["pass"]),
        "distinct_pcm": len({r["sha256"] for r in load if r["ok"]}),
    }


def report(summary: dict, records: list[dict], worst: int = 10) -> str:
    def pct_list(values: list[float]) -> str:
        if len(values) < 2:
            return f"max {values[0]:.1%}" if values else "-"
        q = st.quantiles(values, n=20, method="inclusive")
        return f"p50 {q[9]:.1%}  p95 {q[18]:.1%}  max {max(values):.1%}"

    lines = [
        f"References: {summary['references']}, "
        f"{'deterministic (byte-identical)' if summary['deterministic'] else 'not byte-identical'}, "
        f"duration {summary['reference_duration_s']:.2f} s, "
        f"{summary['reference_failures']} failed",
    ]
    for tag in ("ref", "load"):
        wers = [
            r["wer"] for r in records if r["tag"] == tag and r.get("wer") is not None
        ]
        lines.append(f"{tag:>4} WER: {pct_list(wers)}")
    if summary["wer_limit"] is not None:
        lines.append(f"WER limit: {summary['wer_limit']:.1%}")
    lines.append(
        f"Load: {summary['load_requests'] - summary['load_failures']}/"
        f"{summary['load_requests']} pass, {summary['distinct_pcm']} distinct PCM"
    )
    failed = [r for r in records if not r["pass"]]
    reasons: dict[str, int] = {}
    for r in failed:
        for reason in r["reasons"]:
            key = re.sub(r"[\d.]+%?s?", "N", reason)
            reasons[key] = reasons.get(key, 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:4d} x {reason}")
    failed.sort(key=lambda r: -(r.get("wer") or 0))
    for r in failed[:worst]:
        dur = f" ({r['audio_s']:.1f} s)" if r["ok"] else ""
        lines.append(f"  {r['tag']} #{r['i']:03d}{dur} {'; '.join(r['reasons'])}")
        if r.get("transcript") is not None:
            lines.append(f'       "{r["transcript"][:160]}"')
    return "\n".join(lines)


# ------------------------------------------------------------------------- sending


def send(
    tgt: harness.Target, tag: str, count: int, concurrency: int, outdir: str
) -> list[dict]:
    """Send ``count`` copies of one request over ``concurrency`` workers; save WAVs."""

    def one(i: int) -> dict:
        body = bytearray()
        rec = harness.request(tgt, i, body=body)
        wav = os.path.join(outdir, "wav", f"{tag}_{i:03d}.wav")
        with open(wav, "wb") as f:
            f.write(body)
        return {
            "tag": tag,
            "i": i,
            "ok": rec["ok"],
            "status": rec["status"],
            "error": rec["error"],
            "ttfb_s": rec["ttfb_s"],
            "audio_s": rec["audio_s"],
            "sha256": hashlib.sha256(body[harness.WAV_HEADER_BYTES :]).hexdigest(),
            "wav": wav,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, range(count)))


def _speech_ids(args: argparse.Namespace) -> list[int]:
    if args.speech_ids:
        with open(args.speech_ids) as f:
            ids = json.load(f)
        return ids["speech_ids"] if isinstance(ids, dict) else ids
    return []


def _run(args: argparse.Namespace) -> int:
    text = args.text
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read().strip()
    os.makedirs(os.path.join(args.outdir, "wav"), exist_ok=True)
    tgt = harness.Target(
        host=args.host,
        port=args.port,
        path=args.path,
        key=args.key,
        timeout=args.timeout,
        text=text,
        speech_ids=_speech_ids(args),
    )
    logger.info(
        "Text %d chars, voice prompt %d speech IDs", len(text), len(tgt.speech_ids)
    )
    t0 = time.time()
    logger.info("[ref] %d sequential requests", args.refs)
    records = send(tgt, "ref", args.refs, 1, args.outdir)
    logger.info("[load] %d requests at concurrency %d", args.requests, args.concurrency)
    records += send(tgt, "load", args.requests, args.concurrency, args.outdir)
    logger.info("Sent in %.0f s", time.time() - t0)
    doc = {
        "meta": {
            "host": args.host,
            "port": args.port,
            "text": text,
            "speech_ids": tgt.speech_ids,
            "refs": args.refs,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "started": t0,
        },
        "records": records,
    }
    return _score_and_save(doc, args)


def _score_and_save(doc: dict, args: argparse.Namespace) -> int:
    summary = score(doc["records"], doc["meta"]["text"], args)
    doc["summary"] = summary
    with open(os.path.join(args.outdir, RESULTS_FILE), "w") as f:
        json.dump(doc, f, indent=1)
    print(report(summary, doc["records"]))
    return 1 if summary["load_failures"] or summary["reference_failures"] else 0


# ----------------------------------------------------------------------------- CLI


def _add_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--outdir", default="tts_correctness")
    p.add_argument("--no-asr", action="store_true", help="skip the transcript check")
    p.add_argument(
        "--asr-model",
        default="large-v3",
        help="faster-whisper model name or local path (e.g. distil-large-v3)",
    )
    p.add_argument("--asr-threads", type=int, default=os.cpu_count() or 8)
    p.add_argument(
        "--max-wer",
        type=float,
        default=0.05,
        help="fail a response whose WER against the input text exceeds this",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="send the requests, then score them")
    p.add_argument("--host", default=os.environ.get("TTS_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", 8000)))
    p.add_argument("--path", default=harness.DEFAULT_PATH)
    p.add_argument("--key", default=os.environ.get("TTS_KEY", harness.DEFAULT_KEY))
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--text-file", default=None, help="read the text from a file")
    p.add_argument(
        "--speech-ids",
        default=None,
        help='JSON file: voice prompt speech IDs, a list or {"speech_ids": [...]}',
    )
    p.add_argument("--refs", type=int, default=5, help="sequential reference requests")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=100)
    _add_score_args(p)

    p = sub.add_parser("score", help="re-score a run directory")
    _add_score_args(p)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if args.command == "run":
        sys.exit(_run(args))
    with open(os.path.join(args.outdir, RESULTS_FILE)) as f:
        doc = json.load(f)
    sys.exit(_score_and_save(doc, args))


if __name__ == "__main__":
    main()
