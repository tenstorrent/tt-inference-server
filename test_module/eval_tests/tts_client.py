# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Request/response adapters for the TTS evals (media_server, qwen3_cpp)."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# Both TTS servers accept this bearer when no key is configured.
DEFAULT_API_KEY = "your-secret-key"

TTS_API_MEDIA_SERVER = "media_server"
TTS_API_QWEN3_CPP = "qwen3_cpp"
# tt-tts C++ server: JSON {text, speech_ids} in, one HTTP chunk per audio chunk out.
# Benchmarked by the chunk-aware load harness, evaluated by tts_correctness.
TTS_API_TT_TTS = "tt_tts"

QWEN3_RELEASES = ("base", "custom_voice", "voice_design")
# in_context also puts the clip's codec tokens and transcript in the prompt.
QWEN3_CLONE_MODES = ("voice_only", "in_context")

# Shorter output is a failed sample: post-200 errors end the C++ stream early.
MIN_AUDIO_SECONDS = 0.3
# Share of samples that must synthesize for an eval to pass; seeded servers are deterministic.
DEFAULT_MIN_VALID_FRACTION = 1.0


def enough_valid(valid: int, total: int, min_fraction: float) -> bool:
    """Whether ``valid`` of ``total`` samples meet ``min_fraction`` (and at least one did)."""
    return total > 0 and valid > 0 and valid >= min_fraction * total - 1e-9


_STREAMING_SIZE = 0xFFFFFFFF


def auth_headers() -> dict:
    """Bearer header from OPENAI_API_KEY, then API_KEY, else the default key."""
    token = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or DEFAULT_API_KEY
    return {"Authorization": f"Bearer {token}"}


@dataclass(frozen=True)
class VoiceClip:
    """A reference clip to clone, as the original WAV file bytes."""

    wav_bytes: bytes
    # Exact transcript of the clip (only an in-context clone uses it).
    transcript: str = ""


@dataclass
class SynthesizedAudio:
    """Mono float32 samples in [-1, 1] and their sample rate."""

    samples: Any  # numpy.ndarray
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)


@dataclass
class TimedSynthesis:
    """One benchmark request: success, wall time, TTFT and audio length."""

    ok: bool
    elapsed_s: float = 0.0
    ttft_ms: Optional[float] = None
    audio_seconds: Optional[float] = None


class TTSClient(ABC):
    """Synthesize one utterance against a live TTS deployment."""

    #: The deployment can clone a voice from a clip.
    can_clone: bool = True
    #: The deployment can speak without a clip (a default or built-in voice).
    has_default_voice: bool = True

    @abstractmethod
    async def synthesize(
        self,
        url: str,
        text: str,
        *,
        seed: int,
        voice: Optional[VoiceClip] = None,
        timeout_s: float = 300,
    ) -> Optional[SynthesizedAudio]:
        """Return the audio, or None when the request failed."""

    @abstractmethod
    async def timed_synthesize(
        self,
        url: str,
        text: str,
        *,
        voice: Optional[VoiceClip] = None,
        timeout_s: float = 120,
    ) -> TimedSynthesis:
        """Unseeded benchmark request; the audio itself is not decoded."""

    @abstractmethod
    def describe(self) -> str:
        """One-line description for logs and result data."""


class MediaServerTTSClient(TTSClient):
    """Python tt-media-server contract: JSON in, base64 WAV in JSON out."""

    can_clone = True
    has_default_voice = True

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        payload = {
            "text": text,
            "response_format": "json",
            "seed": seed,
        }
        if voice is not None:
            payload["reference_audio"] = base64.b64encode(voice.wav_bytes).decode(
                "utf-8"
            )
        headers = {
            **auth_headers(),
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(
                        f"Synthesis failed (clone={voice is not None}): "
                        f"HTTP {response.status}"
                    )
                    return None
                result = await response.json()
        return decode_wav_file(base64.b64decode(result["audio"]))

    async def timed_synthesize(self, url, text, *, voice=None, timeout_s=120):
        if voice is not None:
            raise ValueError("the media_server benchmark sends no reference audio")
        headers = {
            "accept": "application/json",
            **auth_headers(),
            "Content-Type": "application/json",
        }
        payload = {"text": text, "response_format": "json"}
        start_time = time.monotonic()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"TTS request failed with status {response.status}: {error_text}"
                    )
                    return TimedSynthesis(False)

                content_type = response.headers.get("Content-Type", "").lower()
                logger.debug(f"Response Content-Type: {content_type}")
                if "audio" in content_type or "wav" in content_type:
                    logger.error(
                        f"Received audio/wav response instead of JSON. "
                        f"Make sure response_format='json' is set in request. "
                        f"Content-Type: {content_type}. Request payload was: {payload}"
                    )
                    return TimedSynthesis(False)

                response_start = time.monotonic()
                response_data = await response.json()
                ttft_ms = (response_start - start_time) * 1000

                audio_duration = response_data.get("duration")
                if audio_duration is None:
                    logger.warning("Duration not found in response data")
                else:
                    logger.info(f"Audio duration: {audio_duration}s")

                audio_base64 = response_data.get("audio")
                if not audio_base64:
                    logger.error("Audio data not found in response")
                    return TimedSynthesis(False)

                logger.info(f"Received audio data (base64 length: {len(audio_base64)})")

        return TimedSynthesis(
            True, time.monotonic() - start_time, ttft_ms, audio_duration
        )

    def describe(self):
        return "media_server (seeded JSON, base64 reference_audio)"


class Qwen3CppTTSClient(TTSClient):
    """Qwen3-TTS on the C++ media server: JSON or multipart in, streamed WAV out."""

    def __init__(
        self,
        release: str,
        *,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        clone_mode: str = "voice_only",
    ):
        if release not in QWEN3_RELEASES:
            raise ValueError(
                f"qwen3_release must be one of {QWEN3_RELEASES}, got {release!r}"
            )
        if clone_mode not in QWEN3_CLONE_MODES:
            raise ValueError(
                f"clone_mode must be one of {QWEN3_CLONE_MODES}, got {clone_mode!r}"
            )
        if release == "custom_voice" and not speaker:
            raise ValueError("a custom_voice deployment needs a speaker")
        if release == "voice_design" and not description:
            raise ValueError("a voice_design deployment needs a description")
        self.release = release
        self.speaker = speaker
        self.language = language
        self.description = description
        self.clone_mode = clone_mode
        self.can_clone = release == "base"
        self.has_default_voice = release != "base"

    def build_request(self, text: str, voice: Optional[VoiceClip]) -> dict:
        """Return ``aiohttp`` post kwargs (``json=`` or ``data=``) for one request."""
        fields = {"text": text}
        if self.language:
            fields["language"] = self.language
        if voice is None:
            if not self.has_default_voice:
                raise ValueError("a base deployment cannot speak without a clip")
            if self.speaker:
                fields["speaker"] = self.speaker
            if self.description:
                fields["description"] = self.description
            return {"json": fields}

        if not self.can_clone:
            raise ValueError(f"a {self.release} deployment cannot clone a voice")
        if self.clone_mode == "in_context":
            if not voice.transcript:
                raise ValueError("an in-context clone needs the clip's transcript")
            fields["reference_text"] = voice.transcript
        elif self.description:
            # The server accepts an instruction only for a voice-only clone.
            fields["description"] = self.description
        form = aiohttp.FormData()
        for name, value in fields.items():
            form.add_field(name, value)
        form.add_field(
            "voice",
            voice.wav_bytes,
            filename="reference.wav",
            content_type="audio/wav",
        )
        return {"data": form}

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        request = self.build_request(text, voice)
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            async with session.post(url, **request) as response:
                if response.status != 200:
                    body = (await response.text())[:300]
                    logger.error(
                        f"Synthesis failed (clone={voice is not None}): "
                        f"HTTP {response.status}: {body}"
                    )
                    return None
                body = await response.read()
        try:
            audio = decode_streamed_wav(body)
        except ValueError as e:
            logger.error(f"Unreadable WAV from the server: {e}")
            return None
        if audio.seconds < MIN_AUDIO_SECONDS:
            # The stream ended early: the server failed after the 200.
            logger.error(
                f"Server returned {audio.seconds:.2f}s of audio "
                f"(< {MIN_AUDIO_SECONDS}s); counting the sample as failed"
            )
            return None
        return audio

    async def timed_synthesize(self, url, text, *, voice=None, timeout_s=120):
        # TTFT is the first byte after the WAV header: the header comes at once, audio only when the utterance is done.
        request = self.build_request(text, voice)
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        start = time.monotonic()
        body = bytearray()
        arrivals = []
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            async with session.post(url, **request) as response:
                if response.status != 200:
                    error_text = (await response.text())[:300]
                    logger.error(
                        f"TTS request failed with status {response.status}: {error_text}"
                    )
                    return TimedSynthesis(False)
                async for chunk in response.content.iter_any():
                    body += chunk
                    arrivals.append((len(body), time.monotonic()))
        elapsed = time.monotonic() - start
        try:
            sample_rate, channels, offset = parse_pcm16_wav_header(bytes(body))
        except ValueError as e:
            logger.error(f"Unreadable WAV from the server: {e}")
            return TimedSynthesis(False)
        seconds = (len(body) - offset) // (2 * channels) / float(sample_rate)
        if seconds < MIN_AUDIO_SECONDS:
            logger.error(
                f"Server returned {seconds:.2f}s of audio (< {MIN_AUDIO_SECONDS}s)"
            )
            return TimedSynthesis(False)
        first_audio = next(t for size, t in arrivals if size > offset)
        return TimedSynthesis(True, elapsed, (first_audio - start) * 1000, seconds)

    def describe(self):
        voice = {
            "base": f"{self.clone_mode} clone",
            "custom_voice": f"speaker={self.speaker}",
            "voice_design": "described voice",
        }[self.release]
        return (
            f"qwen3_cpp release={self.release} ({voice}), "
            f"language={self.language or 'server default'}"
        )


def make_tts_client(targets: Mapping[str, Any]) -> TTSClient:
    """Build the adapter an eval's targets ask for (``media_server`` by default)."""
    api = targets.get("tts_api", TTS_API_MEDIA_SERVER)
    if api == TTS_API_MEDIA_SERVER:
        return MediaServerTTSClient()
    if api == TTS_API_QWEN3_CPP:
        return Qwen3CppTTSClient(
            targets.get("qwen3_release", "base"),
            speaker=targets.get("speaker"),
            language=targets.get("language"),
            description=targets.get("description"),
            clone_mode=targets.get("clone_mode", "voice_only"),
        )
    raise ValueError(
        f"Unknown tts_api {api!r}; known: {TTS_API_MEDIA_SERVER}, {TTS_API_QWEN3_CPP}"
    )


# ------------------------------------------------------------ chunked streaming
# The tt-tts stream: a 44-byte WAV header, then mono 16-bit PCM, one HTTP chunk
# per audio chunk.
STREAM_WAV_HEADER_BYTES = 44
STREAM_BYTES_PER_SAMPLE = 2


def tt_tts_payload(text: str, speech_ids: Optional[list] = None) -> dict:
    """Request body for the tt-tts server; ``speech_ids`` is the voice prompt."""
    payload: dict = {"text": text}
    if speech_ids:
        payload["speech_ids"] = list(speech_ids)
    return payload


async def stream_speech(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    *,
    sample_rate: int,
    timeout_s: float = 300,
    max_chunks: int = 0,
    rec: Optional[dict] = None,
    body: Optional[bytearray] = None,
) -> dict:
    """POST one request and time every HTTP chunk of the streamed WAV.

    Returns ``rec`` (a new dict if none is given) with monotonic offsets from the
    send: ``marks`` (arrival of each audio chunk), ``ttfb_s`` (first chunk),
    ``sc_s`` (first->second gap) and ``gaps`` (every later gap), plus
    ``audio_s``, ``nchunks``, ``ok`` and ``error``. ``t_send``/``t_end`` are wall
    clock and always set, so a failed or cancelled request still has a lifetime.
    ``max_chunks`` > 0 stops reading after that many audio chunks (``capped``).
    ``body`` receives the raw response bytes (WAV header and PCM).
    """
    if rec is None:
        rec = {"t_send": time.time()}
    rec.update(
        ok=False,
        capped=False,
        status=None,
        error=None,
        ttfb_s=None,
        sc_s=None,
        gen_s=None,
        gaps=[],
        marks=[],
        nchunks=0,
        bytes=0,
        audio_s=0.0,
        t_end=None,
    )
    marks: list = rec["marks"]
    header_left = STREAM_WAV_HEADER_BYTES
    audio_bytes = 0
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            rec["status"] = response.status
            if response.status != 200:
                text = (await response.text())[:512].strip()
                rec["error"] = f"http {response.status}: {text}"
                return rec
            if "chunked" not in response.headers.get("Transfer-Encoding", "").lower():
                rec["error"] = "response not chunked: cannot time audio chunks"
                return rec
            pending = 0
            async for data, end_of_chunk in response.content.iter_chunks():
                if body is not None:
                    body.extend(data)
                take = min(header_left, len(data))
                header_left -= take
                pending += len(data) - take
                if not end_of_chunk or not pending:
                    continue
                # One whole HTTP chunk of audio has arrived.
                audio_bytes += pending
                pending = 0
                marks.append(time.perf_counter() - t0)
                if max_chunks and len(marks) >= max_chunks:
                    rec["capped"] = True
                    response.close()
                    break
            audio_bytes += pending
        rec["ok"] = audio_bytes > 0
        if not rec["ok"]:
            rec["error"] = "empty audio (header only)"
    except asyncio.CancelledError:
        rec["error"] = "cancelled (still running at the drain deadline)"
        raise
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        # gen_s is the whole request, terminating chunk included: the time the slot was held.
        rec["gen_s"] = time.perf_counter() - t0
        rec["t_end"] = time.time()
        rec["nchunks"] = len(marks)
        rec["bytes"] = audio_bytes + STREAM_WAV_HEADER_BYTES - header_left
        rec["audio_s"] = audio_bytes / STREAM_BYTES_PER_SAMPLE / sample_rate
        if marks:
            rec["ttfb_s"] = marks[0]
            if len(marks) > 1:
                rec["sc_s"] = marks[1] - marks[0]
            rec["gaps"] = [b - a for a, b in zip(marks[1:], marks[2:])]
    return rec


# ------------------------------------------------------------------ WAV decode
def decode_wav_file(wav_bytes: bytes) -> SynthesizedAudio:
    """Decode a finalized WAV file (any format soundfile reads) to mono float32."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return SynthesizedAudio(samples=data, sample_rate=int(sr))


def parse_pcm16_wav_header(data: bytes) -> Tuple[int, int, int]:
    """Return ``(sample_rate, channels, data_offset)`` of a PCM16 WAV."""
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    pos = 12
    fmt = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise ValueError("truncated fmt chunk")
            audio_format, channels, sample_rate, _, _, bits = struct.unpack(
                "<HHIIHH", data[body : body + 16]
            )
            if audio_format != 1 or bits != 16:
                raise ValueError(
                    f"expected PCM16, got format={audio_format} bits={bits}"
                )
            fmt = (sample_rate, channels)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("data chunk before fmt chunk")
            return fmt[0], fmt[1], body
        if size == _STREAMING_SIZE:
            raise ValueError(f"streaming size on non-data chunk {chunk_id!r}")
        pos = body + size + (size & 1)
    raise ValueError("no data chunk")


def decode_streamed_wav(data: bytes) -> SynthesizedAudio:
    """Decode a (possibly streaming-header) PCM16 WAV to mono float32."""
    import numpy as np

    sample_rate, channels, offset = parse_pcm16_wav_header(data)
    frame_bytes = 2 * channels
    payload = data[offset:]
    payload = payload[: len(payload) - len(payload) % frame_bytes]
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return SynthesizedAudio(samples=samples, sample_rate=sample_rate)


__all__ = [
    "DEFAULT_API_KEY",
    "MIN_AUDIO_SECONDS",
    "MediaServerTTSClient",
    "Qwen3CppTTSClient",
    "SynthesizedAudio",
    "TTSClient",
    "TTS_API_MEDIA_SERVER",
    "TTS_API_QWEN3_CPP",
    "TTS_API_TT_TTS",
    "TimedSynthesis",
    "VoiceClip",
    "auth_headers",
    "decode_streamed_wav",
    "decode_wav_file",
    "make_tts_client",
    "parse_pcm16_wav_header",
    "stream_speech",
    "tt_tts_payload",
]
