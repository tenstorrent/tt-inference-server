# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Speaker-similarity (SECS) eval for voice-cloning TTS."""

import asyncio
import logging
import time

from .._libritts import iter_speaker_clips
from .._test_common import BaseTest
from .tts_client import SynthesizedAudio, VoiceClip, decode_wav_file, make_tts_client

# Pinned speaker-verification encoder; transformers + torch are already eval deps.
SPEAKER_ENCODER_REPO = "microsoft/unispeech-sat-base-plus-sv"
ENCODER_SAMPLE_RATE = 16000

DEFAULT_SAMPLE_COUNT = 8

# cross_speaker: clone the same text from another speaker's clip (no default voice).
BASELINE_DEFAULT_VOICE = "default_voice"
BASELINE_CROSS_SPEAKER = "cross_speaker"
SECS_BASELINES = (BASELINE_DEFAULT_VOICE, BASELINE_CROSS_SPEAKER)

# default_voice calibration (p300x2, 8 speakers): SECS 0.877, margin +0.117.
DEFAULT_SECS_FLOOR = 0.65
DEFAULT_MARGIN_THRESHOLD = 0.05
DEFAULT_DATASET_SPLIT = "test.clean"

logger = logging.getLogger(__name__)


def cross_speaker_partner(index: int, count: int) -> int:
    """Speaker whose clone is the cross-speaker baseline for speaker ``index``."""
    if count < 2:
        raise ValueError("cross-speaker baseline needs at least two speakers")
    return (index - 1) % count


def summarize_secs(results, secs_floor: float, margin_threshold: float) -> dict:
    """Aggregate per-sample ``secs``/``baseline_secs``/``margin`` into the gate."""
    valid = [r for r in results if r.get("margin") is not None]
    mean_secs = sum(r["secs"] for r in valid) / len(valid) if valid else None
    mean_baseline = (
        sum(r["baseline_secs"] for r in valid) / len(valid) if valid else None
    )
    mean_margin = sum(r["margin"] for r in valid) / len(valid) if valid else None
    success = (
        mean_secs is not None
        and mean_secs >= secs_floor
        and mean_margin >= margin_threshold
    )
    return {
        "success": success,
        "mean_secs": mean_secs,
        "mean_baseline_secs": mean_baseline,
        "mean_margin": mean_margin,
        "valid_samples": len(valid),
        "sample_count": len(results),
    }


def _fmt(value, spec=".3f"):
    return "N/A" if value is None else format(value, spec)


class TTSSpeakerSimilarityTest(BaseTest):
    """SECS eval: cosine(embed(reference clip), embed(synthesized audio))."""

    KIND = "tts_speaker_similarity_evals"
    TASK_TYPE = "audio"

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets or {}, **kwargs)
        self._encoder = None
        self._feature_extractor = None
        self._client = None

    def _resolve_baseline(self, client) -> str:
        baseline = self.targets.get("secs_baseline") or (
            BASELINE_DEFAULT_VOICE
            if client.has_default_voice
            else BASELINE_CROSS_SPEAKER
        )
        if baseline not in SECS_BASELINES:
            raise ValueError(
                f"secs_baseline must be one of {SECS_BASELINES}, got {baseline!r}"
            )
        if baseline == BASELINE_DEFAULT_VOICE and not client.has_default_voice:
            raise ValueError(
                f"{client.describe()} has no default voice; "
                f"use secs_baseline={BASELINE_CROSS_SPEAKER!r}"
            )
        return baseline

    async def _run_specific_test_async(self):
        start = time.time()
        self.url = f"{self.base_url}/v1/audio/speech"
        sample_count = self.targets.get("sample_count", DEFAULT_SAMPLE_COUNT)
        secs_floor = self.targets.get("secs_floor", DEFAULT_SECS_FLOOR)
        margin_threshold = self.targets.get(
            "margin_threshold", DEFAULT_MARGIN_THRESHOLD
        )
        split = self.targets.get("dataset_split", DEFAULT_DATASET_SPLIT)
        client = self._client = make_tts_client(self.targets)
        if not client.can_clone:
            return {
                "success": False,
                "error": f"{client.describe()} cannot clone a voice",
            }
        baseline_mode = self._resolve_baseline(client)
        logger.info(f"SECS: client={client.describe()}, baseline={baseline_mode}")

        clips = list(
            iter_speaker_clips(
                split,
                sample_count,
                min_seconds=self.targets.get("min_reference_seconds"),
                max_seconds=self.targets.get("max_reference_seconds"),
            )
        )
        if len(clips) < 2:
            return {"success": False, "error": "could not load speaker clips"}
        voices = [VoiceClip(wav, transcript) for _, wav, transcript in clips]

        self._load_encoder()
        try:
            results = []
            for i, (speaker, _wav, _own_text) in enumerate(clips):
                # The next speaker's sentence, never the reference clip's own text.
                text = clips[(i + 1) % len(clips)][2]
                ref_embedding = await asyncio.to_thread(
                    self._embed, decode_wav_file(voices[i].wav_bytes)
                )
                cloned = await self._sample_secs(i, ref_embedding, text, voices[i])
                if baseline_mode == BASELINE_CROSS_SPEAKER:
                    partner = cross_speaker_partner(i, len(clips))
                    baseline_voice = voices[partner]
                    baseline_speaker = clips[partner][0]
                else:
                    baseline_voice = None
                    baseline_speaker = None
                baseline = await self._sample_secs(
                    i, ref_embedding, text, baseline_voice
                )
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
                        "baseline_speaker": baseline_speaker,
                        "margin": margin,
                    }
                )
                logger.info(
                    f"Sample {speaker}: SECS={_fmt(cloned)} "
                    f"baseline={_fmt(baseline)}"
                    + (f" (clone of {baseline_speaker})" if baseline_speaker else "")
                    + f" margin={_fmt(margin, '+.3f')}"
                )
        finally:
            self._unload_encoder()

        summary = summarize_secs(results, secs_floor, margin_threshold)
        logger.info(
            f"TTS Speaker Similarity completed: mean_secs={_fmt(summary['mean_secs'])}, "
            f"mean_baseline={_fmt(summary['mean_baseline_secs'])}, "
            f"mean_margin={_fmt(summary['mean_margin'], '+.3f')}, "
            f"valid={summary['valid_samples']}/{summary['sample_count']}, "
            f"floor={secs_floor}, margin_threshold={margin_threshold}, "
            f"success={summary['success']}, duration={time.time() - start:.1f}s"
        )
        return {
            **summary,
            "baseline": baseline_mode,
            "client": client.describe(),
            "secs_floor": secs_floor,
            "margin_threshold": margin_threshold,
            "per_sample": results,
        }

    async def _sample_secs(self, index, ref_embedding, text, voice):
        """Synthesize ``text`` in ``voice`` (default voice when None) and score it."""
        audio = await self._client.synthesize(self.url, text, seed=index, voice=voice)
        if audio is None:
            logger.error(
                f"Synthesis failed for sample {index} (clone={voice is not None})"
            )
            return None
        # Embedding is CPU-bound; keep the event loop responsive.
        return await asyncio.to_thread(self._secs, ref_embedding, audio)

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

    def _embed(self, audio: SynthesizedAudio):
        import librosa
        import torch

        data = audio.samples
        if audio.sample_rate != ENCODER_SAMPLE_RATE:
            data = librosa.resample(
                data, orig_sr=audio.sample_rate, target_sr=ENCODER_SAMPLE_RATE
            )
        inputs = self._feature_extractor(
            data, sampling_rate=ENCODER_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            return self._encoder(**inputs).embeddings[0]

    def _secs(self, ref_embedding, audio: SynthesizedAudio):
        import torch

        out = self._embed(audio)
        return float(
            torch.nn.functional.cosine_similarity(ref_embedding, out, dim=-1).clamp(
                -1.0, 1.0
            )
        )


__all__ = [
    "BASELINE_CROSS_SPEAKER",
    "BASELINE_DEFAULT_VOICE",
    "TTSSpeakerSimilarityTest",
    "cross_speaker_partner",
    "summarize_secs",
]
