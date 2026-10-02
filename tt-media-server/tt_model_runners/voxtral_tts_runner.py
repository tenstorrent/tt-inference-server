# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Voxtral-4B-TTS (mistralai/Voxtral-4B-TTS-2603) runner: text + voice preset -> 24 kHz speech.

Model code: tt-metal `models/experimental/voxtral_tts`, `TtVoxtralBatchedPipeline` at B = settings.max_batch_size
users per decode step (B = 1 included), so the whole per-frame loop (sampling, stop, positions, noise, next input
embedding) runs on device. VOXTRAL_HOST_LOOP=1 selects the original single-user TtVoxtralPipeline (host loop).
Both open nothing themselves: the device comes from BaseMetalDeviceRunner.set_device() with the
L1 scratch and trace region the model exports.

Request fields used: text, voice (or speaker_id) -> one of the 20 shipped presets, seed. Long
texts are split at sentence boundaries (the SpeechT5 runner's chunker) and the clips concatenated.

Environment:
  VOXTRAL_DEFAULT_VOICE   preset used when the request names none (default neutral_male)
  VOXTRAL_MAX_SEQ_LEN     KV window per user, prompt + frames (default 2048; 1024 halves cache memory)
  VOXTRAL_CKPT            model directory; else MODEL_WEIGHTS_DIR, else the Hugging Face cache (HF_HOME)

Weights license: CC BY-NC 4.0 (non-commercial), including the voice presets. See the README.
"""

import asyncio
import base64
import io
import os
import re
import time
import wave
from typing import List, Optional, Tuple

import numpy as np
from config.settings import settings
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
import torch
from utils.decorators import log_execution_time

SAMPLE_RATE = 24000
DEFAULT_VOICE = os.environ.get("VOXTRAL_DEFAULT_VOICE", "neutral_male")
# Characters per synthesized chunk. Voxtral's own cap is the KV window (prompt + frames); a
# sentence-bounded chunk of this size stays well inside it at any voice.
CHUNK_CHARS = int(os.environ.get("VOXTRAL_CHUNK_CHARS", "400"))
TORCH_THREADS = int(
    os.environ.get("VOXTRAL_TORCH_THREADS", "4")
)  # host-side torch threads per worker


# Model-side errors caused by the request's content, not by the device: the prompt does not fit the KV window
# (ValueError "leaves no room") or the model ends the clip on its first frame. They fail only their request, even
# when it is alone in the batch, so they never count toward a worker restart.
_REQUEST_ERROR_MARKERS = ("leaves no room", "on the first frame")


def is_request_error(err: BaseException) -> bool:
    return isinstance(err, ValueError) or any(
        m in str(err) for m in _REQUEST_ERROR_MARKERS
    )


_SENTENCE_END = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\u0964\u061f])\s+")


def chunk_sentences(text: str, max_chars: int) -> List[str]:
    """Split at sentence boundaries (Latin, CJK, Devanagari and Arabic enders) and pack greedily
    up to `max_chars`; a single sentence longer than that is cut at the last space before the
    limit. Never returns an empty chunk."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    pieces = []
    for sent in _SENTENCE_END.split(text):
        sent = sent.strip()
        while len(sent) > max_chars:
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            pieces.append(sent)
    chunks, cur = [], ""
    for piece in pieces:
        if not cur:
            cur = piece
        elif len(cur) + 1 + len(piece) <= max_chars:
            cur = f"{cur} {piece}"
        else:
            chunks.append(cur)
            cur = piece
    if cur:
        chunks.append(cur)
    return chunks


def waveform_to_numpy(wav) -> np.ndarray:
    """torch tensor ([1,1,N] or [N]) or array-like -> float32 numpy [N]. No torch import: the
    server's unit tests stub torch out, and the runner only ever reads the pipeline's output."""
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().float().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def wav_bytes_from_waveform(wav, sample_rate: int = SAMPLE_RATE) -> bytes:
    """waveform float in [-1,1] -> 16-bit PCM mono WAV bytes."""
    samples = np.clip(waveform_to_numpy(wav), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return buf.getvalue()


class TTVoxtralTTSRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        if device_id != "-1":
            # Per-frame host work (sampling for B users) runs in torch; the base default of one
            # thread costs ~2 ms per 80 ms frame. (The matmul throttle is disabled for this runner
            # in settings: it costs 30% per frame.)
            # Intra-op only: the base class already fixed the interop pool, which torch allows once.
            torch.set_num_threads(TORCH_THREADS)
        self.pipeline = None
        self.voices: List[str] = []
        self.model_dir: Optional[str] = None
        self.batch = max(1, int(getattr(self.settings, "max_batch_size", 1) or 1))
        if not settings.is_galaxy:
            # Single-chip model: no fabric needed (as the SpeechT5 runner does).
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    # ---- model code imports, kept lazy so the module imports without tt-metal on the path ----
    @staticmethod
    def _pipeline_module():
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pm

        return pm

    def get_pipeline_device_params(self):
        pm = self._pipeline_module()
        return {
            "l1_small_size": pm.L1_SMALL_SIZE,
            "trace_region_size": pm.TRACE_REGION_SIZE,
        }

    def _ckpt_path(self) -> Optional[str]:
        """Explicit model directory if the launcher mounted one; else let the model resolve
        ($VOXTRAL_CKPT, then the Hugging Face cache, downloading if needed)."""
        p = getattr(self.settings, "model_weights_path", "") or ""
        return p if p and os.path.isdir(p) else None

    def load_weights(self):
        """Resolve (and if needed download) the checkpoint; used by the download-only pass."""
        from models.experimental.voxtral_tts.reference.voxtral_paths import (
            resolve_model_dir,
        )

        self.model_dir = resolve_model_dir(self._ckpt_path())
        self.logger.info(
            f"Device {self.device_id}: Voxtral weights at {self.model_dir}"
        )
        return True

    @log_execution_time(
        "Voxtral model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        if self.ttnn_device is None:
            raise ValueError("Device not initialized. Call set_device() first.")
        try:
            await asyncio.to_thread(self._build_and_warm)
            self.logger.info(
                f"Device {self.device_id}: Voxtral ready (batch={self.batch}, voices={len(self.voices)}, "
                f"default voice={DEFAULT_VOICE}, warmed={getattr(self.pipeline, 'warmed', {})})"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Voxtral model loading failed: {e}"
            )
            self.pipeline = None
            raise RuntimeError(
                f"Device {self.device_id}: Model loading failed: {str(e)}"
            ) from e

    def _build_and_warm(self):
        from models.experimental.voxtral_tts import frontend

        pm = self._pipeline_module()
        max_seq_len = int(os.environ.get("VOXTRAL_MAX_SEQ_LEN", "2048"))
        ckpt = self._ckpt_path()
        if os.environ.get("VOXTRAL_HOST_LOOP", "0") == "1" and self.batch == 1:
            self.pipeline = pm.TtVoxtralPipeline(
                self.ttnn_device, ckpt_path=ckpt, max_seq_len=max_seq_len
            )
        else:
            from models.experimental.voxtral_tts.tt.ttnn_voxtral_batched import (
                TtVoxtralBatchedPipeline,
            )

            self.pipeline = TtVoxtralBatchedPipeline(
                self.ttnn_device,
                ckpt_path=ckpt,
                max_batch=self.batch,
                max_seq_len=max_seq_len,
            )
        self.model_dir = self.pipeline.model_dir
        self.voices = list(frontend.voices(self.model_dir))
        if DEFAULT_VOICE not in self.voices:
            raise ValueError(
                f"VOXTRAL_DEFAULT_VOICE={DEFAULT_VOICE!r} is not a shipped preset: {self.voices}"
            )
        self.pipeline.warmup()

    # ---- requests ----
    def _resolve(self, request: TextToSpeechRequest) -> Tuple[str, str, int]:
        text = (request.text or "").strip()
        if not text:
            raise ValueError("Text cannot be empty")
        voice = (
            getattr(request, "voice", None)
            or getattr(request, "speaker_id", None)
            or DEFAULT_VOICE
        )
        if self.voices and voice not in self.voices:
            raise ValueError(
                f"unknown voice {voice!r}; one of: {', '.join(self.voices)}"
            )
        seed = getattr(request, "seed", None)
        return text, voice, int(seed) if seed is not None else 0

    @staticmethod
    def _chunks(text: str) -> List[str]:
        return chunk_sentences(text, CHUNK_CHARS)

    def _synthesize_jobs(self, jobs: List[Tuple[str, str, int]]):
        """[(text, voice, seed)] -> [waveform], batching through the pipeline when it can."""
        self._truncated = []  # job indices that hit the frame cap instead of emitting [END_AUDIO]
        if self.batch > 1 and len(jobs) > 1:
            out = []
            for i in range(0, len(jobs), self.batch):
                out.extend(self.pipeline.synthesize_batch(jobs[i : i + self.batch]))
                self._note_truncated(i)
            return out
        out = []
        for i, (t, v, s) in enumerate(jobs):
            out.append(self.pipeline.synthesize(t, v, seed=s))
            self._note_truncated(i)
        return out

    def _note_truncated(self, first_job: int):
        timings = getattr(self.pipeline, "last_timings", None)
        natural = (
            timings.get("stopped_naturally") if isinstance(timings, dict) else None
        ) or []
        self._truncated.extend(first_job + k for k, ok in enumerate(natural) if not ok)

    @log_execution_time(
        "Run Voxtral inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def _synthesize_requests(self, resolved):
        """{request index: (text, voice, seed)} -> ({index: [numpy waveform per chunk]}, [request index of each
        chunk cut off by the frame cap], chunks)."""
        jobs, owner = [], []
        for i, (text, voice, seed) in resolved.items():
            for c in self._chunks(text):
                jobs.append((c, voice, seed))
                owner.append(i)
        wavs = self._synthesize_jobs(jobs)
        parts = {i: [] for i in resolved}
        for i, w in zip(owner, wavs):
            parts[i].append(waveform_to_numpy(w))
        return parts, [owner[j] for j in self._truncated], len(jobs)

    def run(self, requests: List[TextToSpeechRequest]):
        """One response per request, in order. A bad request fails alone: its slot holds an Exception, which the
        service re-raises for that client only (base_service), so it neither fails the co-batched requests nor
        counts toward a worker restart. A content error (is_request_error) fails its request even when it runs alone.
        run() raises only when nothing was synthesized and some error was not the requests' own (likely the device)."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline not loaded. Call warmup() first.")
        t0 = time.perf_counter()
        out = [None] * len(requests)
        resolved = {}
        for i, r in enumerate(requests):
            try:
                resolved[i] = self._resolve(r)
            except Exception as e:  # noqa: BLE001  unknown voice, empty text: the client's error, isolated
                out[i] = ValueError(str(e))
        parts, truncated, n_jobs = {}, [], 0
        if resolved:
            try:
                parts, truncated, n_jobs = self._synthesize_requests(resolved)
            except Exception as batch_err:
                if len(resolved) == 1:
                    if not is_request_error(batch_err):
                        raise  # likely the device: the worker counts it
                    # The request's own content (no room, empty clip): fail it alone, no restart count.
                    out[next(iter(resolved))] = RuntimeError(
                        f"{type(batch_err).__name__}: {batch_err}"
                    )
                else:
                    self.logger.warning(
                        f"Device {self.device_id}: Voxtral batch of {len(resolved)} failed "
                        f"({type(batch_err).__name__}: {batch_err}); retrying one request at a time to isolate it"
                    )
                    device_like = 0
                    for i, job in resolved.items():
                        try:
                            p, t, n = self._synthesize_requests({i: job})
                            parts.update(p)
                            truncated += t
                            n_jobs += n
                        except Exception as e:  # noqa: BLE001  any model-side error fails only its request
                            out[i] = RuntimeError(
                                f"{type(e).__name__}: {e}"
                            )  # plain type: crosses the worker queue
                            device_like += not is_request_error(e)
                    if not parts and device_like:
                        raise  # nothing synthesized, not the requests' fault: re-raise (likely the device)
        for i, chunks in parts.items():
            samples = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            out[i] = TextToSpeechResponse(
                audio=base64.b64encode(wav_bytes_from_waveform(samples)).decode(
                    "ascii"
                ),
                duration=float(samples.size) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                format="wav",
                speaker_id=resolved[i][1],
            )
        failed = [i for i, r in enumerate(out) if isinstance(r, Exception)]
        if failed:
            self.logger.warning(
                f"Device {self.device_id}: Voxtral request(s) {failed} failed: {[str(out[i]) for i in failed]}"
            )
        if truncated:
            self.logger.warning(
                f"Device {self.device_id}: Voxtral {len(truncated)} chunk(s) hit the frame cap and were cut off "
                f"(requests {sorted(set(truncated))}); the audio is truncated"
            )
        timings = getattr(self.pipeline, "last_timings", {}) or {}
        audio_s = sum(r.duration for r in out if isinstance(r, TextToSpeechResponse))
        self.logger.info(
            f"Device {self.device_id}: Voxtral {len(requests)} request(s), {n_jobs} chunk(s), "
            f"audio {audio_s:.1f}s in {time.perf_counter() - t0:.2f}s; "
            f"decode_ms_per_frame={timings.get('decode_ms_per_frame', float('nan')):.1f}"
        )
        return out
