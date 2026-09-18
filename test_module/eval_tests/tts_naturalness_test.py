# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Naturalness (predicted MOS) eval for TTS.

Synthesizes a fixed battery of LibriTTS-R sentences (default voice, seeded)
and scores each clip with UTMOS22-strong, a neural MOS predictor trained on
human listening-test ratings of synthesized speech. Covers the failure class
WER and SECS are both blind to: output that is intelligible and keeps the
speaker's timbre but sounds degraded (vocoder buzz, broken prosody, clipping).

Scores are predictor-relative, not absolute truth: gate against a baseline
measured for THIS model on THIS predictor version, and re-calibrate when
either changes.
"""

import asyncio
import base64
import io
import logging
import time

import aiohttp

from .._libritts import iter_speaker_clips
from .._test_common import BaseTest

# Pinned MOS predictor: torch.hub repo tag + entrypoint.
MOS_PREDICTOR_REPO = "tarepan/SpeechMOS:v1.2.0"
MOS_PREDICTOR_ENTRYPOINT = "utmos22_strong"

DEFAULT_SAMPLE_COUNT = 8
DEFAULT_DATASET_SPLIT = "test.clean"
DEFAULT_UTMOS_THRESHOLD = 3.7

logger = logging.getLogger(__name__)

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class TTSNaturalnessTest(BaseTest):
    """Predicted-MOS eval: utmos22_strong over a seeded synthesis battery."""

    KIND = "tts_naturalness_evals"
    TASK_TYPE = "audio"

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets, **kwargs)
        self._predictor = None

    async def _run_specific_test_async(self):
        start = time.time()
        self.url = f"{self.base_url}/v1/audio/speech"
        sample_count = self.targets.get("sample_count", DEFAULT_SAMPLE_COUNT)
        threshold = self.targets.get("utmos_threshold", DEFAULT_UTMOS_THRESHOLD)
        split = self.targets.get("dataset_split", DEFAULT_DATASET_SPLIT)

        texts = [t for _, _, t in iter_speaker_clips(split, sample_count)]
        if not texts:
            return {"success": False, "error": "could not load eval texts"}

        self._load_predictor()
        try:
            results = []
            for i, text in enumerate(texts):
                mos = await self._sample_mos(i, text)
                results.append({"index": i, "utmos": mos})
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
            "valid_samples": len(valid),
            "sample_count": len(results),
            "per_sample": results,
        }

    async def _sample_mos(self, index, text):
        """Synthesize one seeded text with the default voice and score it."""
        payload = {"text": text, "response_format": "json", "seed": index}
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            async with session.post(self.url, json=payload) as response:
                if response.status != 200:
                    logger.error(
                        f"Synthesis failed for sample {index}: HTTP {response.status}"
                    )
                    return None
                result = await response.json()
        wav_bytes = base64.b64decode(result["audio"])
        return await asyncio.to_thread(self._score, wav_bytes)

    def _load_predictor(self):
        import torch

        logger.info(f"Loading MOS predictor {MOS_PREDICTOR_REPO} ...")
        self._predictor = torch.hub.load(
            MOS_PREDICTOR_REPO, MOS_PREDICTOR_ENTRYPOINT, trust_repo=True
        )
        self._predictor.eval()

    def _score(self, wav_bytes):
        import soundfile as sf
        import torch

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        with torch.no_grad():
            return float(self._predictor(torch.from_numpy(data).unsqueeze(0), sr))


__all__ = ["TTSNaturalnessTest"]
