# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Media-server runner for hexgrad/Kokoro-82M text-to-speech.

Kokoro-82M is a non-autoregressive StyleTTS2 / ISTFTNet TTS model. The entire
pipeline runs on a single Tenstorrent P150: TT plbert (the only attention
transformer), the StyleTTS2 prosody predictor, the text encoder, and the
ISTFTNet vocoder. This runner is a thin adapter over
``models.demos.audio.kokoro.tt.generator.KokoroGenerator``, which owns the text
frontend (grapheme-to-phoneme, plbert-context chunking) and the on-device
synthesis — mirroring how the Whisper runner wraps ``WhisperGenerator``.

The generator synthesizes chunk-by-chunk and can stream each chunk's waveform as
it is produced on device (``KokoroGenerator.stream``); the single-file
``/v1/audio/speech`` response is assembled from that same on-device stream via
``KokoroGenerator.generate``.
"""

import asyncio
import base64
import io
import os
from typing import List

import numpy as np
import soundfile as sf
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time

from models.demos.audio.kokoro.tt.generator import build_generator

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000  # Kokoro / ISTFTNet output rate
DEFAULT_VOICE = "af_heart"
# ISTFTNet vocoder convs need an L1_SMALL scratch region that grows with a chunk's
# frame count; the generator caps chunk size (CHUNK_PHONEME_TOKENS) to stay within
# the validated envelope, and this gives headroom above the 32 KB the per-stage
# device tests use.
L1_SMALL_SIZE = 65536
TRACE_REGION_SIZE = 90_000_000


class TTKokoroRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.generator = None  # KokoroGenerator (fully on-device, P150)

    # --------------------------------------------------------------- device
    def get_pipeline_device_params(self):
        # set_device / close_device use the BaseMetalDeviceRunner defaults; the
        # single-chip mesh-graph descriptor is supplied per-device via the model
        # spec (env_vars: TT_MESH_GRAPH_DESC_PATH), like the other P150 runners.
        return {
            "l1_small_size": L1_SMALL_SIZE,
            "trace_region_size": TRACE_REGION_SIZE,
        }

    def close_device(self):
        if self.generator is not None:
            self.generator.teardown()
            self.generator = None
        return super().close_device()

    # ---------------------------------------------------------- model setup
    def load_weights(self):
        """Verify weights are reachable (HF cache) without touching the device."""
        from huggingface_hub import hf_hub_download

        hf_hub_download(MODEL_ID, "config.json")
        hf_hub_download(MODEL_ID, "kokoro-v1_0.pth")
        return True

    @log_execution_time(
        "Kokoro model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        if self.ttnn_device is None:
            raise ValueError("Device not initialized. Call set_device() first.")
        self.generator = await asyncio.to_thread(
            build_generator, self.ttnn_device, MODEL_ID, DEFAULT_VOICE
        )
        # Exercise the on-device path once so kernels/programs are compiled before
        # the first real request (mirrors the other MEDIA runners' warmup).
        await asyncio.to_thread(self.generator.generate, "Hello.", DEFAULT_VOICE, 1.0)
        self.logger.info(f"Device {self.device_id}: Kokoro warmup complete")
        return True

    # ------------------------------------------------------------ inference
    def _synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        """Full on-device synthesis. ``KokoroGenerator.generate`` consumes the
        per-chunk on-device stream and joins the chunks into one waveform."""
        return self.generator.generate(text, voice, speed)

    def _np_to_b64_wav(self, audio: np.ndarray) -> str:
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _resolve_voice(self, voice: str) -> str:
        """Preload the voice pack (populating the generator cache) with fallback to
        the default voice, so the synth path never re-hits a missing-voice error."""
        try:
            self.generator._load_voice(voice)
            return voice
        except FileNotFoundError:
            self.logger.warning(
                f"Voice '{voice}' not found; falling back to {DEFAULT_VOICE}"
            )
            self.generator._load_voice(DEFAULT_VOICE)
            return DEFAULT_VOICE

    async def _stream_chunks(self, request, text: str, voice: str, speed: float):
        """Async generator: one ``streaming_chunk`` dict per on-device chunk (each a
        base64 WAV of that chunk), then a ``final_result`` sentinel. Consumed by the
        device worker's streaming path; each chunk is synthesized on device and the
        blocking ``next()`` is offloaded so the event loop stays responsive."""
        gen = self.generator.stream(text, voice, speed)
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, gen, sentinel)
            if chunk is sentinel:
                break
            yield {
                "type": "streaming_chunk",
                "chunk": TextToSpeechResponse(
                    audio=self._np_to_b64_wav(chunk),
                    duration=len(chunk) / SAMPLE_RATE,
                    sample_rate=SAMPLE_RATE,
                    format="wav",
                    speaker_id=voice,
                ),
                "task_id": request._task_id,
            }
        # Audio is fully delivered via the chunks above; the final sentinel just
        # closes the stream (return=False -> nothing extra yielded to the client).
        yield {
            "type": "final_result",
            "result": None,
            "task_id": request._task_id,
            "return": False,
        }

    async def _run_async(self, requests: List[TextToSpeechRequest]):
        if self.generator is None:
            raise RuntimeError("Model not loaded. Call warmup() first.")
        if len(requests) > 1:
            self.logger.warning(
                f"Device {self.device_id}: batch not supported; processing first of "
                f"{len(requests)} requests"
            )
        request = requests[0]
        if request is None or not request.text or not request.text.strip():
            raise ValueError("Text cannot be empty")

        voice = await asyncio.to_thread(
            self._resolve_voice, request.speaker_id or DEFAULT_VOICE
        )

        # Streaming: return an async generator the device worker drives chunk-by-chunk.
        if getattr(request, "stream", False):
            return self._stream_chunks(request, request.text, voice, 1.0)

        audio = await asyncio.to_thread(self._synthesize, request.text, voice, 1.0)
        return TextToSpeechResponse(
            audio=self._np_to_b64_wav(audio),
            duration=len(audio) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            format="wav",
            speaker_id=voice,
        )

    @log_execution_time(
        "Run Kokoro inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: List[TextToSpeechRequest]):
        result = asyncio.run(self._run_async(requests))
        return [result] if result is not None else []
