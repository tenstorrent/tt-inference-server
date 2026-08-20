# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import os
import time

import numpy as np
import soundfile as sf
import torch
from config.settings import settings
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.text_chunking import chunk_text

XTTS_SAMPLE_RATE = 24000

WARMUP_TIMEOUT_SECONDS = 6000

# XTTS-v2 was trained on utterances of <= ~250 characters; longer single-shot prompts
# audibly degrade (multi-second mid-utterance silences, rushed pace) and the model hard-caps
# at ~400 text tokens per generate. Long request texts are therefore split at sentence
# boundaries into chunks of at most this size (utils/text_chunking.py, shared with the
# speecht5 runner) and synthesized per chunk, with a short silence stitched between chunks.
CHUNK_CHAR_LIMIT = 240
INTER_CHUNK_SILENCE_SECONDS = 0.25

# Default reference voice: the coqui/XTTS-v2 HF repo ships language sample clips; the
# English one is used when no XTTS_REF_AUDIO is configured, so the documented
# run.py --docker-server flow works with zero extra configuration.
DEFAULT_VOICE_HF_FILE = "samples/en_sample.wav"


class XttsV2Runner(BaseMetalDeviceRunner):
    """XTTS-v2 text-to-speech runner.

    The XttsV2 pipeline class opens its own (1,1) mesh device (z-image pattern),
    so set_device() is a no-op and close_device() delegates to the pipeline.
    English-only in the current tt-metal implementation.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None
        self.default_voice = None

        # Explicitly disable fabric for non-galaxy devices (mirrors speecht5 runner)
        if not settings.is_galaxy:
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    def set_device(self):
        # The XttsV2 pipeline opens its own (1,1) mesh device.
        pass

    def close_device(self):
        if self.pipeline is not None:
            try:
                self.logger.info(
                    f"Device {self.device_id}: Closing XTTS-v2 mesh device..."
                )
                self.pipeline.close()
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed mesh device"
                )
            except Exception as e:
                self.logger.error(
                    f"Device {self.device_id}: Failed to close device: {e}"
                )
                raise RuntimeError(
                    f"Device {self.device_id}: Device cleanup failed: {str(e)}"
                ) from e

    def _resolve_reference_audio_path(self) -> str:
        """Reference-voice precedence: $XTTS_REF_AUDIO (explicit) > an en_sample.wav in
        the downloaded weights dir > fetch the coqui repo's English sample from HF hub.
        The fallbacks make the stock deployment flow work without extra env/mounts."""
        ref_path = os.environ.get("XTTS_REF_AUDIO")
        if ref_path:
            if not os.path.exists(ref_path):
                raise RuntimeError(f"XTTS_REF_AUDIO does not exist: {ref_path}")
            return ref_path
        if settings.model_weights_path:
            candidate = os.path.join(settings.model_weights_path, DEFAULT_VOICE_HF_FILE)
            if os.path.exists(candidate):
                return candidate
        from huggingface_hub import hf_hub_download

        self.logger.info(
            f"Device {self.device_id}: XTTS_REF_AUDIO not set; fetching default voice "
            f"{DEFAULT_VOICE_HF_FILE} from coqui/XTTS-v2"
        )
        return hf_hub_download(repo_id="coqui/XTTS-v2", filename=DEFAULT_VOICE_HF_FILE)

    def _load_reference_audio(self):
        """Load the reference voice clip as (waveform, sample_rate).

        Delegates to the model's own loader rather than reimplementing it: that one covers
        wav/flac/ogg plus every .pt shape people actually have (a bare tensor, a (tensor, sr)
        tuple, a HuggingFace audio dict), and it returns the clip's REAL sample rate instead of
        assuming one — assuming it makes a 44.1 kHz clip clone at the wrong pitch, silently."""
        from models.experimental.xtts_v2.frontend import load_reference_audio

        return load_reference_audio(self._resolve_reference_audio_path())

    @log_execution_time(
        "XTTS-v2 warmup",
        TelemetryEvent.DEVICE_WARMUP,
        lambda: os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading XTTS-v2 ...")

        def load_and_warmup():
            from models.experimental.xtts_v2.tt.ttnn_xtts_model import XttsV2

            # Checkpoint precedence: explicit $XTTS_CKPT (dev override) > the server's own
            # weights-download location (settings.model_weights_path holds the downloaded
            # coqui/XTTS-v2 snapshot when the generic weights step ran; it may also be the
            # bare HF repo id, which the exists() check skips) > None, which lets the model
            # class fetch from HF hub itself.
            ckpt_path = os.environ.get("XTTS_CKPT")
            if not ckpt_path and settings.model_weights_path:
                candidate = os.path.join(settings.model_weights_path, "model.pth")
                if os.path.exists(candidate):
                    ckpt_path = candidate
            self.pipeline = XttsV2(ckpt_path=ckpt_path)
            self.pipeline.warmup()

            ref_audio, sample_rate = self._load_reference_audio()
            self.default_voice = self.pipeline.compute_voice(ref_audio, sample_rate)
            self.logger.info(
                f"Device {self.device_id}: Default voice computed in "
                f"{self.pipeline.last_timings.get('compute_voice_s', -1):.2f}s"
            )

        await asyncio.wait_for(
            asyncio.to_thread(load_and_warmup),
            timeout=WARMUP_TIMEOUT_SECONDS,
        )

        self.logger.info(f"Device {self.device_id}: XTTS-v2 warmup complete")
        return True

    def _synthesize(self, text: str, base_seed: int) -> np.ndarray:
        """Synthesize one request's text: chunk at sentence boundaries, generate per
        chunk (chunk i uses base_seed + i so chunks don't share sampling draws), stitch
        with short silences. Returns float32 samples @ XTTS_SAMPLE_RATE."""
        chunks = chunk_text(text, max_chunk_size=CHUNK_CHAR_LIMIT)
        pieces: list[np.ndarray] = []
        gap = np.zeros(
            int(INTER_CHUNK_SILENCE_SECONDS * XTTS_SAMPLE_RATE), dtype=np.float32
        )
        for i, chunk in enumerate(chunks):
            wav = self.pipeline.generate(chunk, self.default_voice, seed=base_seed + i)
            samples = wav.reshape(-1).detach().to(torch.float32).numpy()
            if samples.size == 0:
                # Rare, seed-dependent: the model produced zero codes for this chunk
                # (its "empty audio" contract). Retry once with a different seed.
                self.logger.warning(
                    f"Device {self.device_id}: empty generation for chunk {i}, retrying"
                )
                wav = self.pipeline.generate(
                    chunk, self.default_voice, seed=base_seed + i + 7919
                )
                samples = wav.reshape(-1).detach().to(torch.float32).numpy()
            pieces.append(samples)
            pieces.append(gap)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces[:-1])

    @log_execution_time(
        "XTTS-v2 inference",
        TelemetryEvent.MODEL_INFERENCE,
        lambda: os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[TextToSpeechRequest]):
        if self.pipeline is None or self.default_voice is None:
            raise RuntimeError("Model not loaded. Call warmup() first.")

        # One response per request (the device worker indexes responses[i] per request).
        # Synthesis is sequential — max_batch_size is pinned to 1 for this runner, but an
        # env override must degrade to slower responses, not an IndexError.
        responses = []
        for request in requests:
            if request is None:
                raise ValueError("Request cannot be None")
            if not request.text or not request.text.strip():
                raise ValueError("Text cannot be empty")

            # Fixed seed -> reproducible audio for identical text; None -> a fresh
            # random base per request.
            base_seed = (
                request.seed
                if request.seed is not None
                else int(torch.seed() % (2**31))
            )

            t_start = time.time()
            samples = self._synthesize(request.text, base_seed)
            elapsed = time.time() - t_start

            audio_buffer = io.BytesIO()
            sf.write(
                audio_buffer, samples, XTTS_SAMPLE_RATE, subtype="PCM_16", format="WAV"
            )
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
            duration = len(samples) / XTTS_SAMPLE_RATE

            self.logger.info(
                f"Device {self.device_id}: Generated {duration:.2f}s of audio in "
                f"{elapsed:.2f}s (base_seed={base_seed}), "
                f"last_chunk_timings={self.pipeline.last_timings}"
            )
            responses.append(
                TextToSpeechResponse(
                    audio=audio_base64,
                    duration=duration,
                    sample_rate=XTTS_SAMPLE_RATE,
                    format="wav",
                )
            )

        return responses
