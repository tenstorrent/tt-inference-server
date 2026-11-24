# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Audio benchmark client for a vLLM server: OpenAI ``/v1/audio/transcriptions``.

Each timed request is a multipart upload with ``stream=true``; the server answers
with server-sent ``transcription.chunk`` events and a final usage event.

- TTFT: time to the first event carrying text.
- RTR: audio duration over request time. The duration comes from one untimed
  ``verbose_json`` request per payload, as the stream does not carry it.
- T/S/U: completion tokens from the usage event over the time after TTFT. vLLM
  splits audio longer than one model clip into requests that decode at the same
  time, so T/S/U is only reported for audio that fits in one clip.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

TRANSCRIPTIONS = "/v1/audio/transcriptions"
# Qwen3-ASR's clip length in vLLM (get_speech_to_text_config max_audio_clip_s).
MAX_CLIP_S = 30.0
REQUEST_TIMEOUT_S = 300


@dataclass
class TranscriptionTiming:
    ok: bool
    elapsed: float
    ttft: Optional[float] = None
    tsu: Optional[float] = None
    rtr: Optional[float] = None


def _form(model: str, audio: bytes, **fields) -> aiohttp.FormData:
    form = aiohttp.FormData()
    form.add_field("file", audio, filename="audio.mp3", content_type="audio/mpeg")
    form.add_field("model", model)
    for key, value in fields.items():
        form.add_field(key, value)
    return form


async def audio_duration(base_url: str, model: str, audio: bytes) -> Optional[float]:
    """Duration in seconds, as the server measured it (verbose_json)."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            base_url + TRANSCRIPTIONS,
            data=_form(model, audio, response_format="verbose_json"),
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
        ) as response:
            if response.status != 200:
                logger.error(
                    f"verbose_json request failed: {response.status} {await response.text()}"
                )
                return None
            return (await response.json()).get("duration")


async def transcribe_streaming(
    base_url: str, model: str, audio: bytes, duration: Optional[float]
) -> TranscriptionTiming:
    start = time.monotonic()
    ttft: Optional[float] = None
    completion_tokens: Optional[int] = None
    text = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            base_url + TRANSCRIPTIONS,
            data=_form(model, audio, stream="true", stream_include_usage="true"),
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
        ) as response:
            if response.status != 200:
                logger.error(
                    f"streaming request failed: {response.status} {await response.text()}"
                )
                return TranscriptionTiming(ok=False, elapsed=time.monotonic() - start)
            async for raw in response.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                event = json.loads(body)
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta and ttft is None:
                        ttft = time.monotonic() - start
                    text += delta
                usage = event.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens")
    elapsed = time.monotonic() - start
    if not text.strip():
        logger.error("streaming request returned no text")
        return TranscriptionTiming(ok=False, elapsed=elapsed)

    rtr = duration / elapsed if duration else None
    tsu = None
    one_clip = duration is not None and duration <= MAX_CLIP_S
    if one_clip and completion_tokens and ttft is not None and elapsed > ttft:
        tsu = completion_tokens / (elapsed - ttft)
    logger.info(
        f"TTFT={ttft}s elapsed={elapsed:.2f}s tokens={completion_tokens} "
        f"T/S/U={tsu} RTR={rtr} text={text[:80]!r}"
    )
    return TranscriptionTiming(ok=True, elapsed=elapsed, ttft=ttft, tsu=tsu, rtr=rtr)


def decode_payload(audio_b64: str) -> bytes:
    return base64.b64decode(audio_b64)
