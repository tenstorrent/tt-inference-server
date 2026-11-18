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
"""

import asyncio
import hashlib
import json
import logging
import re
import statistics as st
import time
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
DEFAULT_ASR_MODEL = "distil-whisper/distil-large-v3"
DEFAULT_TIMEOUT_SECONDS = 300.0
SAMPLE_RATE = 48000
ASR_SAMPLE_RATE = 16000
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
        outdir = Path(t.get("outdir", "tts_correctness"))
        (outdir / "wav").mkdir(parents=True, exist_ok=True)
        payload = tt_tts_payload(text, t.get("speech_ids"))
        url = f"{self.base_url}/v1/audio/speech"

        async with aiohttp.ClientSession(
            headers=auth_headers(), connector=aiohttp.TCPConnector(limit=0)
        ) as session:
            logger.info(f"[ref] {refs} sequential requests")
            records = await self._send(session, url, payload, "ref", refs, 1, outdir)
            logger.info(f"[load] {requests} requests at concurrency {concurrency}")
            records += await self._send(
                session, url, payload, "load", requests, concurrency, outdir
            )

        ok = [r for r in records if r["ok"]]
        logger.info(f"Transcribing {len(ok)} responses with {asr_model}")
        transcripts = await asyncio.to_thread(
            self._transcribe, [r["wav"] for r in ok], asr_model
        )
        ref_words = normalize(text)
        for r in ok:
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

    async def _send(self, session, url, payload, tag, count, concurrency, outdir):
        """``count`` copies of one request over ``concurrency`` slots; save WAVs."""
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
    def _transcribe(paths: list, model_name: str) -> dict:
        """English Whisper transcripts; long-form audio is chunked, so a runaway
        response's extra speech still shows up as insertions."""
        import librosa
        from transformers import pipeline

        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device="cpu",
            chunk_length_s=30,
        )
        out = {}
        for n, path in enumerate(paths, 1):
            audio = decode_streamed_wav(Path(path).read_bytes())
            samples = librosa.resample(
                audio.samples, orig_sr=audio.sample_rate, target_sr=ASR_SAMPLE_RATE
            )
            out[path] = asr(
                {"raw": samples, "sampling_rate": ASR_SAMPLE_RATE},
                generate_kwargs={"language": "en", "task": "transcribe"},
            )["text"].strip()
            if n % 20 == 0:
                logger.info(f"  transcribed {n}/{len(paths)}")
        return out


__all__ = ["TTSCorrectnessTest", "grade", "normalize", "word_errors"]
