# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Cross-talk spec test: requests sent together must each speak their own text in
their own voice, as well as when they run alone."""

import asyncio
import logging
import time

from report_module.renderers import NOTES_KEY

from .._libritts import iter_speaker_clips
from .tts_client import decode_wav_file, make_tts_client
from .tts_naturalness_test import naturalness_battery
from .tts_quality_test import TTSQualityTest
from .tts_speaker_similarity_test import TTSSpeakerSimilarityTest

DEFAULT_CONCURRENCY = 4
# Sampling is random, so WER alone and together differ run to run; a larger gap
# means the batch hurts every request.
DEFAULT_MAX_WER_GAP = 0.10
DEFAULT_DATASET_SPLIT = "test.clean"

CROSS_TALK_LEGEND = (
    "**Legend**: C LibriTTS-R speakers, each cloned to read another speaker's "
    "sentence; every request runs alone, then all C at once.\n\n"
    "- **wer_alone / wer_together**: WER of the request run alone / run with the "
    "others.\n"
    "- **wer_other_min**: lowest WER of the batched output against any other "
    "request's text; near 1 means no other text leaked in.\n"
    "- **secs_to_alone**: speaker similarity of the batched output to the same "
    "request run alone.\n"
    "- **secs_to_other_max**: highest similarity to any other request run "
    "alone.\n"
    "- **secs_clip**: similarity to its own reference clip; shown only.\n"
    "- **own_best**: own text has the lowest WER and own voice the highest "
    "similarity. Pass: every request own_best, and wer_gap (mean wer_together − "
    "mean wer_alone) ≤ max_wer_gap."
)

logger = logging.getLogger(__name__)


def _own_best(own_index: int, values: list, higher_is_better: bool) -> bool:
    """Whether ``values[own_index]`` strictly beats every other entry."""
    own = values[own_index]
    others = [v for i, v in enumerate(values) if i != own_index]
    return all(own > v if higher_is_better else own < v for v in others)


def isolation_row(
    index: int,
    speaker: str,
    wer_alone: float,
    wers: list,
    secs: list,
    secs_clip: float,
):
    """One request's row. Its batched output must be nearest its own text (``wers``,
    against every request's text) and its own voice (``secs``, against every
    request's solo output). The speaker encoder scores some LibriTTS-R voices close
    together, so the solo outputs, not the reference clips, are the voice yardstick;
    ``secs_clip`` (batched output against its own clip) is reported only."""
    return {
        "request": index,
        "speaker": speaker,
        "wer_alone": wer_alone,
        "wer_together": wers[index],
        "wer_other_min": min(w for i, w in enumerate(wers) if i != index),
        "secs_to_alone": secs[index],
        "secs_to_other_max": max(s for i, s in enumerate(secs) if i != index),
        "secs_clip": secs_clip,
        "own_best": _own_best(index, wers, False) and _own_best(index, secs, True),
    }


def summarize_isolation(rows: list, max_wer_gap: float) -> dict:
    """Fail on any request that matches another's text or voice better than its own,
    or when batching raises the mean WER by more than ``max_wer_gap``."""
    wer_alone = sum(r["wer_alone"] for r in rows) / len(rows)
    wer_together = sum(r["wer_together"] for r in rows) / len(rows)
    mismatches = sum(not r["own_best"] for r in rows)
    gap = wer_together - wer_alone
    return {
        "success": mismatches == 0 and gap <= max_wer_gap,
        "wer_alone": wer_alone,
        "wer_together": wer_together,
        "wer_gap": gap,
        "mismatches": mismatches,
        "max_wer_gap": max_wer_gap,
    }


class TTSIsolationTest(TTSQualityTest, TTSSpeakerSimilarityTest):
    """Whisper WER and speaker similarity of each request, alone and batched."""

    KIND = "tts_cross_talk"
    TASK_TYPE = "audio"

    def _block(self, data):
        block = super()._block(data)
        block.targets[NOTES_KEY] = [CROSS_TALK_LEGEND]
        return block

    async def _run_specific_test_async(self):
        start = time.time()
        self.url = f"{self.base_url}/v1/audio/speech"
        count = self.targets.get("concurrency", DEFAULT_CONCURRENCY)
        max_wer_gap = self.targets.get("max_wer_gap", DEFAULT_MAX_WER_GAP)
        client = self._client = make_tts_client(self.targets)
        if not client.can_clone:
            return {"success": False, "error": f"{client.describe()} cannot clone"}

        clips = list(
            iter_speaker_clips(
                self.targets.get("dataset_split", DEFAULT_DATASET_SPLIT),
                count,
                min_seconds=self.targets.get("min_reference_seconds"),
                max_seconds=self.targets.get("max_reference_seconds"),
            )
        )
        if len(clips) < count:
            return {"success": False, "error": "could not load speaker clips"}
        # Each request reads another speaker's text, so text and voice are both
        # distinct per request.
        battery = naturalness_battery(clips, cloned=True)
        texts = [text for text, _ in battery]
        logger.info(f"Cross-talk: client={client.describe()}, C={count}")

        alone = []
        for i, (text, voice) in enumerate(battery):
            alone.append(await client.synthesize(self.url, text, seed=i, voice=voice))
        together = await asyncio.gather(
            *(
                client.synthesize(self.url, text, seed=i, voice=voice)
                for i, (text, voice) in enumerate(battery)
            )
        )
        failed = sum(audio is None for audio in alone + list(together))
        if failed:
            return {"success": False, "error": f"{failed} synthesis request(s) failed"}

        self._load_whisper_model()
        self._load_encoder()
        try:
            clip_embeddings = [
                await asyncio.to_thread(self._embed, decode_wav_file(voice.wav_bytes))
                for _, voice in battery
            ]
            alone_embeddings = [
                await asyncio.to_thread(self._embed, audio) for audio in alone
            ]
            rows = []
            for i in range(count):
                heard_alone = await asyncio.to_thread(
                    self._transcribe_samples, alone[i].samples, alone[i].sample_rate
                )
                heard = await asyncio.to_thread(
                    self._transcribe_samples,
                    together[i].samples,
                    together[i].sample_rate,
                )
                batched = await asyncio.to_thread(self._embed, together[i])
                rows.append(
                    isolation_row(
                        i,
                        clips[i][0],
                        self._calculate_wer(texts[i], heard_alone),
                        [self._calculate_wer(text, heard) for text in texts],
                        [self._cosine(batched, solo) for solo in alone_embeddings],
                        self._cosine(batched, clip_embeddings[i]),
                    )
                )
                logger.info(f"Cross-talk request {i}: {rows[-1]} heard={heard!r}")
        finally:
            self._unload_encoder()
            self._unload_whisper_model()

        summary = summarize_isolation(rows, max_wer_gap)
        logger.info(
            f"TTS cross-talk completed: {summary}, duration={time.time() - start:.1f}s"
        )
        return {**summary, "concurrency": count, "per_request": rows}

    @staticmethod
    def _cosine(a, b) -> float:
        import torch

        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))


__all__ = ["TTSIsolationTest", "isolation_row", "summarize_isolation"]
