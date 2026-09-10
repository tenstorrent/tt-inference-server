# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Speaker-similarity (SECS) eval for voice-cloning TTS.

For each of ``sample_count`` distinct LibriTTS-R speakers: send the speaker's
clip as the request's ``reference_audio``, synthesize a text the clip does NOT
contain (each speaker reads the next speaker's sentence, so content overlap
cannot inflate the score), embed both the reference clip and the synthesized
audio with a pinned speaker-verification model, and score their cosine
similarity. The gate is margin-based — cloned audio must sit measurably closer
to the reference than the default voice does — see the calibration comment at
the threshold constants below.

WER (tts_quality_test) is blind to identity: cloning could silently fall back
to the default voice and still transcribe perfectly. SECS is the metric that
catches that.
"""

import asyncio
import base64
import io
import logging
import time

import aiohttp

from .._libritts import iter_speaker_clips
from .._test_common import BaseTest

# Pinned speaker-verification encoder. transformers + torch are already eval
# dependencies, so this adds no new packages; the checkpoint is cached in
# HF_HOME after the first run.
SPEAKER_ENCODER_REPO = "microsoft/unispeech-sat-base-plus-sv"
ENCODER_SAMPLE_RATE = 16000

DEFAULT_SAMPLE_COUNT = 8

# Gate calibration (p300x2, 2026-09-07, 8 LibriTTS-R test.clean speakers):
# x-vector cosines are NOT calibrated to an absolute scale — the default
# (uncloned) voice already scores mean 0.760 (max 0.879) against arbitrary
# reference clips, while cloned outputs score mean 0.877. An absolute
# threshold can therefore barely separate working from broken cloning. The
# primary gate is the MARGIN: for every speaker the eval synthesizes the same
# text twice (with and without reference_audio, same seed) and requires
# cloned audio to sit closer to the reference than the default voice does.
# Measured mean margin +0.117; a cloning regression (output ignores the
# reference) collapses it to ~0. Ground-truth check: the same clips/texts/seeds
# through the vendored CPU coqui-reference blocks score mean SECS 0.888 /
# margin +0.130 — TT is at coqui parity.
#
# The absolute floor catches degenerate (non-speech) output. Measured:
# garbage audio scores mean 0.30-0.55 against the reference clips (erratic,
# not near zero — single pairs reach 0.91), so 0.65 sits above the worst
# garbage mean and well below the working mean (0.877).
DEFAULT_SECS_FLOOR = 0.65
DEFAULT_MARGIN_THRESHOLD = 0.05
DEFAULT_DATASET_SPLIT = "test.clean"

logger = logging.getLogger(__name__)

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class TTSSpeakerSimilarityTest(BaseTest):
    """SECS eval: cosine(embed(reference clip), embed(synthesized audio))."""

    KIND = "tts_speaker_similarity_evals"
    TASK_TYPE = "audio"

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets, **kwargs)
        self._encoder = None
        self._feature_extractor = None

    async def _run_specific_test_async(self):
        start = time.time()
        self.url = f"{self.base_url}/v1/audio/speech"
        sample_count = self.targets.get("sample_count", DEFAULT_SAMPLE_COUNT)
        secs_floor = self.targets.get("secs_floor", DEFAULT_SECS_FLOOR)
        margin_threshold = self.targets.get(
            "margin_threshold", DEFAULT_MARGIN_THRESHOLD
        )
        split = self.targets.get("dataset_split", DEFAULT_DATASET_SPLIT)

        clips = list(iter_speaker_clips(split, sample_count))
        if len(clips) < 2:
            return {"success": False, "error": "could not load speaker clips"}

        self._load_encoder()
        try:
            results = []
            for i, (speaker, wav_bytes, _own_text) in enumerate(clips):
                # Speak the NEXT speaker's sentence: same text domain, but never
                # the text of the reference clip itself.
                text = clips[(i + 1) % len(clips)][2]
                cloned = await self._sample_secs(i, wav_bytes, text, clone=True)
                baseline = await self._sample_secs(i, wav_bytes, text, clone=False)
                margin = (
                    cloned - baseline
                    if cloned is not None and baseline is not None
                    else None
                )
                results.append(
                    {
                        "speaker": speaker,
                        "secs": cloned,
                        "baseline_secs": baseline,
                        "margin": margin,
                    }
                )
                logger.info(
                    f"Sample {speaker}: SECS="
                    + (f"{cloned:.3f}" if cloned is not None else "N/A")
                    + " baseline="
                    + (f"{baseline:.3f}" if baseline is not None else "N/A")
                    + " margin="
                    + (f"{margin:+.3f}" if margin is not None else "N/A")
                )
        finally:
            self._unload_encoder()

        valid = [r for r in results if r["margin"] is not None]
        mean_secs = sum(r["secs"] for r in valid) / len(valid) if valid else None
        mean_margin = sum(r["margin"] for r in valid) / len(valid) if valid else None
        success = (
            mean_secs is not None
            and mean_secs >= secs_floor
            and mean_margin >= margin_threshold
        )
        logger.info(
            "TTS Speaker Similarity completed: mean_secs="
            + (f"{mean_secs:.3f}" if mean_secs is not None else "N/A")
            + ", mean_margin="
            + (f"{mean_margin:+.3f}" if mean_margin is not None else "N/A")
            + f", valid={len(valid)}/{len(results)}, floor={secs_floor}, "
            f"margin_threshold={margin_threshold}, success={success}, "
            f"duration={time.time() - start:.1f}s"
        )
        return {
            "success": success,
            "mean_secs": mean_secs,
            "mean_margin": mean_margin,
            "secs_floor": secs_floor,
            "margin_threshold": margin_threshold,
            "valid_samples": len(valid),
            "sample_count": len(results),
            "per_sample": results,
        }

    async def _sample_secs(self, index, ref_wav_bytes, text, *, clone):
        """Synthesize ``text`` (cloned voice or default) and score it against
        the reference clip. Same seed either way, so the cloned/baseline pair
        differs only in conditioning."""
        payload = {
            "text": text,
            "response_format": "json",
            # Deterministic per sample position so reruns reproduce bit-identical
            # audio (and therefore identical SECS) on the same deployment.
            "seed": index,
        }
        if clone:
            payload["reference_audio"] = base64.b64encode(ref_wav_bytes).decode("utf-8")
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            async with session.post(self.url, json=payload) as response:
                if response.status != 200:
                    logger.error(
                        f"Synthesis failed for sample {index} (clone={clone}): "
                        f"HTTP {response.status}"
                    )
                    return None
                result = await response.json()
        out_wav_bytes = base64.b64decode(result["audio"])
        # Embedding is CPU-bound; keep the event loop responsive.
        return await asyncio.to_thread(self._secs, ref_wav_bytes, out_wav_bytes)

    # ------------------------------------------------------------- encoder
    def _load_encoder(self):
        from transformers import AutoFeatureExtractor, UniSpeechSatForXVector

        logger.info(f"Loading speaker encoder {SPEAKER_ENCODER_REPO} ...")
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            SPEAKER_ENCODER_REPO
        )
        self._encoder = UniSpeechSatForXVector.from_pretrained(SPEAKER_ENCODER_REPO)
        self._encoder.eval()

    def _unload_encoder(self):
        self._encoder = None
        self._feature_extractor = None

    def _embed(self, wav_bytes):
        import librosa
        import soundfile as sf
        import torch

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != ENCODER_SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=ENCODER_SAMPLE_RATE)
        inputs = self._feature_extractor(
            data, sampling_rate=ENCODER_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            return self._encoder(**inputs).embeddings[0]

    def _secs(self, ref_wav_bytes, out_wav_bytes):
        import torch

        ref = self._embed(ref_wav_bytes)
        out = self._embed(out_wav_bytes)
        return float(
            torch.nn.functional.cosine_similarity(ref, out, dim=-1).clamp(-1.0, 1.0)
        )


__all__ = ["TTSSpeakerSimilarityTest"]
