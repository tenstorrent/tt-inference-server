# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Naturalness (predicted MOS, UTMOS22-strong) eval for TTS."""

import asyncio
import logging
import time

from .._libritts import iter_speaker_clips
from .._test_common import BaseTest
from .tts_client import SynthesizedAudio, VoiceClip, make_tts_client

# Pinned MOS predictor: torch.hub repo tag + entrypoint.
MOS_PREDICTOR_REPO = "tarepan/SpeechMOS:v1.2.0"
MOS_PREDICTOR_ENTRYPOINT = "utmos22_strong"

DEFAULT_SAMPLE_COUNT = 8
DEFAULT_DATASET_SPLIT = "test.clean"
DEFAULT_UTMOS_THRESHOLD = 3.7

logger = logging.getLogger(__name__)


def naturalness_battery(clips, cloned: bool):
    """Return ``[(text, voice_or_None)]``; a clone never reads its own clip's words."""
    if not cloned:
        return [(text, None) for _, _, text in clips]
    return [
        (clips[(i + 1) % len(clips)][2], VoiceClip(wav, transcript))
        for i, (_, wav, transcript) in enumerate(clips)
    ]


class TTSNaturalnessTest(BaseTest):
    """Predicted-MOS eval: utmos22_strong over a seeded synthesis battery."""

    KIND = "tts_naturalness_evals"
    TASK_TYPE = "audio"

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets or {}, **kwargs)
        self._predictor = None

    async def _run_specific_test_async(self):
        start = time.time()
        self.url = f"{self.base_url}/v1/audio/speech"
        sample_count = self.targets.get("sample_count", DEFAULT_SAMPLE_COUNT)
        threshold = self.targets.get("utmos_threshold", DEFAULT_UTMOS_THRESHOLD)
        split = self.targets.get("dataset_split", DEFAULT_DATASET_SPLIT)
        client = make_tts_client(self.targets)
        # A server without a default voice (Qwen3-TTS Base) is scored on clones.
        cloned = not client.has_default_voice
        logger.info(f"Naturalness: client={client.describe()}, cloned={cloned}")

        clips = list(
            iter_speaker_clips(
                split,
                sample_count,
                min_seconds=self.targets.get("min_reference_seconds"),
                max_seconds=self.targets.get("max_reference_seconds"),
            )
        )
        if not clips or (cloned and len(clips) < 2):
            return {"success": False, "error": "could not load eval texts"}
        battery = naturalness_battery(clips, cloned)

        self._load_predictor()
        try:
            results = []
            for i, (text, voice) in enumerate(battery):
                audio = await client.synthesize(self.url, text, seed=i, voice=voice)
                if audio is None:
                    logger.error(f"Synthesis failed for sample {i}")
                    mos = None
                else:
                    mos = await asyncio.to_thread(self._score, audio)
                results.append(
                    {
                        "index": i,
                        "utmos": mos,
                        "speaker": clips[i][0] if cloned else None,
                        "audio_seconds": audio.seconds if audio else None,
                    }
                )
                logger.info(
                    f"Sample {i}: UTMOS=" + (f"{mos:.3f}" if mos is not None else "N/A")
                )
        finally:
            self._predictor = None

        valid = [r["utmos"] for r in results if r["utmos"] is not None]
        mean_utmos = sum(valid) / len(valid) if valid else None
        success = mean_utmos is not None and mean_utmos >= threshold
        logger.info(
            "TTS Naturalness completed: mean_utmos="
            + (f"{mean_utmos:.3f}" if mean_utmos is not None else "N/A")
            + f", valid={len(valid)}/{len(results)}, threshold={threshold}, "
            f"success={success}, duration={time.time() - start:.1f}s"
        )
        return {
            "success": success,
            "mean_utmos": mean_utmos,
            "utmos_threshold": threshold,
            "client": client.describe(),
            "cloned": cloned,
            "valid_samples": len(valid),
            "sample_count": len(results),
            "per_sample": results,
        }

    def _load_predictor(self):
        import torch

        logger.info(f"Loading MOS predictor {MOS_PREDICTOR_REPO} ...")
        self._predictor = torch.hub.load(
            MOS_PREDICTOR_REPO, MOS_PREDICTOR_ENTRYPOINT, trust_repo=True
        )
        self._predictor.eval()

    def _score(self, audio: SynthesizedAudio):
        import torch

        with torch.no_grad():
            return float(
                self._predictor(
                    torch.from_numpy(audio.samples).unsqueeze(0), audio.sample_rate
                )
            )


__all__ = ["TTSNaturalnessTest", "naturalness_battery"]
