# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Same-prompt correctness eval for the tt-tts server (``tts_api=tt_tts``).

``refs`` sequential reference requests, then ``requests`` copies of the same
request at ``concurrency``. Each response, reference or load, fails on either:
  * no audio: HTTP error, transport error, or a header-only WAV;
  * transcript: Whisper WER against the input text above ``max_wer``.
    Substitutions point at garbled words, deletions at dropped or truncated
    audio, insertions at runaway output.
Any failure fails the eval. Duration against the reference median and whether
the PCM matches a reference are recorded for information only: the server
samples, so slow but intelligible speech is not a failure. References run with
nothing else in flight, so a failing reference means the server misbehaves
without load too. WAVs and per-response results land in ``outdir``.

ASR is faster-whisper (CTranslate2, int8, greedy) on CPU. The model loads while
the reference requests run, and each response is transcribed as soon as it
arrives, on ``asr_workers`` parallel workers that split the CPU budget.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import statistics as st
import time
from math import gcd
from pathlib import Path

import aiohttp

from .._test_common import BaseTest
from .tts_client import (
    STREAM_WAV_HEADER_BYTES,
    auth_headers,
    decode_streamed_wav,
    stream_speech,
    tt_tts_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_TEXT = (
    "Hello, and thank you for calling. Your order has shipped and should arrive "
    "within three business days. If you have any questions about your delivery, "
    "please reply to this message and a member of our team will be happy to help."
)
DEFAULT_REFS = 5
DEFAULT_REQUESTS = 100
DEFAULT_CONCURRENCY = 100
DEFAULT_MAX_WER = 0.05
# faster-whisper (CTranslate2) model name or local path.
DEFAULT_ASR_MODEL = "distil-large-v3"
DEFAULT_TIMEOUT_SECONDS = 300.0
SAMPLE_RATE = 48000
ASR_SAMPLE_RATE = 16000
# Parallel ASR workers cap; each gets an equal share of the CPU budget.
MAX_ASR_WORKERS = 4
RESULTS_FILE = "correctness.json"


def normalize(text: str) -> list:
    """Lowercase words with punctuation dropped ("It's done." -> ["its", "done"])."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", "")).split()


def word_errors(ref: list, hyp: list) -> dict:
    """Word-level Levenshtein alignment -> WER and substitution/deletion/insertion counts."""
    # Each cell: (edits, subs, dels, ins) aligning ref[:i] with hyp[:j].
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
    return {"wer": edits / max(len(ref), 1), "sub": subs, "del": dels, "ins": ins}


def grade(records: list, max_wer: float) -> dict:
    """Mark every record pass/fail and return the summary."""
    refs = [r for r in records if r["tag"] == "ref" and r["ok"]]
    ref_hashes = {r["sha256"] for r in refs}
    ref_dur = st.median(r["audio_s"] for r in refs) if refs else None
    for r in records:
        reasons = []
        if not r["ok"]:
            reasons.append(r["error"] or "no audio")
        else:
            r["duration_ratio"] = r["audio_s"] / ref_dur if ref_dur else None
            r["pcm_matches_ref"] = r["sha256"] in ref_hashes if refs else None
            if r.get("wer") is None:
                reasons.append("no transcript")
            elif r["wer"] > max_wer:
                reasons.append(f"wer {r['wer']:.1%} > {max_wer:.1%}")
        r["reasons"] = reasons
        r["pass"] = not reasons
    load = [r for r in records if r["tag"] == "load"]
    return {
        "reference_requests": sum(1 for r in records if r["tag"] == "ref"),
        "reference_failures": sum(
            1 for r in records if r["tag"] == "ref" and not r["pass"]
        ),
        "deterministic": len(refs) > 1 and len(ref_hashes) == 1,
        "reference_duration_s": ref_dur,
        "load_requests": len(load),
        "load_failures": sum(1 for r in load if not r["pass"]),
        "distinct_pcm": len({r["sha256"] for r in load if r["ok"]}),
    }


def cpu_budget() -> int:
    """CPUs this process may use: affinity, capped by a cgroup CPU quota.

    ``os.cpu_count()`` reports the host's cores inside a CPU-limited container;
    sizing threads from it oversubscribes the pod's share.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1
    quota = None
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            quota = int(q) / int(p)
    except (OSError, ValueError):
        try:  # cgroup v1: quota -1 means unlimited
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    if quota is not None:
        cpus = min(cpus, max(1, int(quota)))
    return max(1, cpus)


def asr_threads(cpus: int, workers: int) -> int:
    """CTranslate2 threads per worker; one CPU stays free for the HTTP client,
    which is still reading responses while the first ones are transcribed."""
    return max(1, (cpus - 1) // workers)


def default_asr_workers(cpus: int) -> int:
    """About four CPU threads per worker, at most ``MAX_ASR_WORKERS`` workers."""
    return max(1, min(MAX_ASR_WORKERS, cpus // 4))


class TTSCorrectnessTest(BaseTest):
    """One prompt sent many times; every response transcribed and graded."""

    KIND = "tts_correctness_evals"
    TASK_TYPE = "audio"

    async def _run_specific_test_async(self):
        start = time.time()
        t = self.targets
        text = t.get("text", DEFAULT_TEXT)
        refs = int(t.get("refs", DEFAULT_REFS))
        requests = int(t.get("requests", DEFAULT_REQUESTS))
        concurrency = int(t.get("concurrency", DEFAULT_CONCURRENCY))
        max_wer = float(t.get("max_wer", DEFAULT_MAX_WER))
        asr_model = t.get("asr_model", DEFAULT_ASR_MODEL)
        cpus = cpu_budget()
        workers = max(1, int(t.get("asr_workers", default_asr_workers(cpus))))
        threads = asr_threads(cpus, workers)
        outdir = Path(t.get("outdir", "tts_correctness"))
        (outdir / "wav").mkdir(parents=True, exist_ok=True)
        payload = tt_tts_payload(text, t.get("speech_ids"))
        url = f"{self.base_url}/v1/audio/speech"

        logger.info(
            f"Loading ASR {asr_model}: {workers} workers x "
            f"{threads} threads ({cpus} CPUs)"
        )
        # Loads alongside the reference requests; transcription waits on it.
        loader = asyncio.ensure_future(
            asyncio.to_thread(self._load_asr, asr_model, workers, threads)
        )
        slots = asyncio.Semaphore(workers)
        pending: dict = {}
        done = [0]

        async def transcribe(wav: str) -> str:
            asr = await loader
            async with slots:
                out = await asyncio.to_thread(asr, wav)
            done[0] += 1
            if done[0] % 20 == 0:
                logger.info(f"  transcribed {done[0]}/{len(pending)}")
            return out

        def on_audio(wav: str) -> None:
            pending[wav] = asyncio.ensure_future(transcribe(wav))

        try:
            async with aiohttp.ClientSession(
                headers=auth_headers(), connector=aiohttp.TCPConnector(limit=0)
            ) as session:
                logger.info(f"[ref] {refs} sequential requests")
                records = await self._send(
                    session, url, payload, "ref", refs, 1, outdir, on_audio
                )
                if loader.done() and loader.exception() is not None:
                    raise loader.exception()
                logger.info(f"[load] {requests} requests at concurrency {concurrency}")
                records += await self._send(
                    session,
                    url,
                    payload,
                    "load",
                    requests,
                    concurrency,
                    outdir,
                    on_audio,
                )
            logger.info(
                f"Requests done after {time.time() - start:.0f}s; waiting on "
                f"{len(pending) - done[0]} of {len(pending)} transcripts"
            )
            results = await asyncio.gather(*pending.values(), return_exceptions=True)
        finally:
            for task in (loader, *pending.values()):
                task.cancel()
        for res in results:
            if isinstance(res, BaseException):
                raise res
        transcripts = dict(zip(pending, results))

        ref_words = normalize(text)
        for r in records:
            if r["ok"]:
                r["transcript"] = transcripts[r["wav"]]
                r.update(word_errors(ref_words, normalize(r["transcript"])))
        summary = grade(records, max_wer)

        graded = records
        wers = [r["wer"] for r in records if r.get("wer") is not None]
        failed = [f"{r['tag']}_{r['i']:03d}" for r in records if not r["pass"]]
        result = {
            "success": not failed,
            "pass_rate": 100.0 * (len(graded) - len(failed)) / len(graded)
            if graded
            else None,
            "mean_wer": sum(wers) / len(wers) if wers else None,
            "max_wer": max_wer,
            "concurrency": concurrency,
            "failed_responses": failed,
            "results_file": str(outdir / RESULTS_FILE),
            **summary,
        }
        with open(outdir / RESULTS_FILE, "w") as f:
            json.dump(
                {"text": text, "summary": result, "records": records}, f, indent=1
            )
        for r in sorted(
            (r for r in records if not r["pass"]), key=lambda r: -(r.get("wer") or 0)
        )[:10]:
            logger.warning(
                f"{r['tag']} #{r['i']:03d}: {'; '.join(r['reasons'])}"
                + (f' "{r["transcript"][:160]}"' if r.get("transcript") else "")
            )
        logger.info(
            f"TTS correctness: {len(graded) - len(failed)}/{len(graded)} pass, "
            f"reference failures={summary['reference_failures']}, "
            f"load failures={summary['load_failures']}, "
            f"distinct load PCM={summary['distinct_pcm']}, "
            f"duration={time.time() - start:.0f}s"
        )
        return result

    async def _send(
        self, session, url, payload, tag, count, concurrency, outdir, on_audio
    ):
        """``count`` copies of one request over ``concurrency`` slots; save WAVs
        and hand each one with audio to ``on_audio`` as soon as it lands."""
        slots = asyncio.Semaphore(concurrency)

        async def one(i: int) -> dict:
            body = bytearray()
            async with slots:
                rec = await stream_speech(
                    session,
                    url,
                    payload,
                    sample_rate=SAMPLE_RATE,
                    timeout_s=DEFAULT_TIMEOUT_SECONDS,
                    body=body,
                )
            wav = outdir / "wav" / f"{tag}_{i:03d}.wav"
            wav.write_bytes(body)
            if rec["ok"]:
                on_audio(str(wav))
            return {
                "tag": tag,
                "i": i,
                "ok": rec["ok"],
                "status": rec["status"],
                "error": rec["error"],
                "ttfb_s": rec["ttfb_s"],
                "audio_s": rec["audio_s"],
                "sha256": hashlib.sha256(body[STREAM_WAV_HEADER_BYTES:]).hexdigest(),
                "wav": str(wav),
            }

        return list(await asyncio.gather(*(one(i) for i in range(count))))

    @staticmethod
    def _load_asr(model_name: str, workers: int, threads: int):
        """Load faster-whisper once; return a thread-safe ``wav path -> text``.

        ``num_workers`` lets ``workers`` threads transcribe in parallel, each
        with ``threads`` CTranslate2 threads. Greedy, no timestamps and
        no carry-over between 30 s windows; no VAD, so a runaway response's
        extra speech still shows up as insertions.
        """
        import numpy as np
        from faster_whisper import WhisperModel
        from scipy.signal import resample_poly

        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=workers,
        )

        def transcribe(path: str) -> str:
            audio = decode_streamed_wav(Path(path).read_bytes())
            samples = audio.samples
            if audio.sample_rate != ASR_SAMPLE_RATE:
                g = gcd(audio.sample_rate, ASR_SAMPLE_RATE)
                samples = resample_poly(
                    samples, ASR_SAMPLE_RATE // g, audio.sample_rate // g
                )
            segments, _ = model.transcribe(
                np.asarray(samples, dtype=np.float32),
                language="en",
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()

        return transcribe


__all__ = ["TTSCorrectnessTest", "cpu_budget", "grade", "normalize", "word_errors"]
