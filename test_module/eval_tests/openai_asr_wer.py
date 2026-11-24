# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""LibriSpeech WER for a vLLM server, through OpenAI ``/v1/audio/transcriptions``.

lmms-eval's audio models either post tt-media-server's JSON payload (whisper_tt) or
run vLLM in-process (whisper_vllm), so neither reaches a served vLLM model. This
sends lmms-eval's own LibriSpeech split to the server and scores the replies with
lmms-eval's own task functions, so the WER is the one lmms-eval reports.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

DATASET = "lmms-lab/librispeech"
TRANSCRIPTIONS = "/v1/audio/transcriptions"
REQUEST_TIMEOUT_S = 600


def _wav_bytes(audio: dict) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(
        buf, audio["array"], audio["sampling_rate"], format="WAV", subtype="PCM_16"
    )
    return buf.getvalue()


async def _transcribe(session, url: str, model: str, wav: bytes) -> str:
    form = aiohttp.FormData()
    form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", model)
    form.add_field("temperature", "0")
    async with session.post(
        url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"transcription failed: {response.status} {await response.text()}"
            )
        return (await response.json())["text"]


async def transcribe_all(
    base_url: str, model: str, wavs: list[bytes], concurrency: int
) -> list[str]:
    url = base_url + TRANSCRIPTIONS
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async with aiohttp.ClientSession() as session:

        async def one(wav: bytes) -> str:
            nonlocal done
            async with limit:
                text = await _transcribe(session, url, model, wav)
            done += 1
            if done % 100 == 0 or done == len(wavs):
                logger.info(f"Transcribed {done}/{len(wavs)}")
            return text

        return await asyncio.gather(*(one(wav) for wav in wavs))


def librispeech_wer(
    base_url: str,
    model: str,
    task_name: str,
    concurrency: int,
    limit: Optional[int] = None,
) -> float:
    """WER in percent on lmms-eval's ``task_name`` split, e.g. librispeech_test_other."""
    from datasets import load_dataset
    from lmms_eval.tasks.librispeech.utils import (
        librispeech_process_result,
        librispeech_wer as aggregate_wer,
    )

    docs = load_dataset(DATASET, task_name, split=task_name, token=True)
    if limit is not None:
        docs = docs.select(range(min(limit, len(docs))))
    logger.info(f"{task_name}: {len(docs)} utterances, concurrency {concurrency}")
    wavs = [_wav_bytes(doc["audio"]) for doc in docs]
    preds = asyncio.run(transcribe_all(base_url, model, wavs, concurrency))
    results = [
        librispeech_process_result(doc, [pred])["wer"] for doc, pred in zip(docs, preds)
    ]
    # lmms-eval's aggregation takes its task args second; the LibriSpeech one reads none.
    return float(aggregate_wer(results, None))
