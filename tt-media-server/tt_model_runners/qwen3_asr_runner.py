# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Media-server runner for Qwen3-ASR-1.7B on Tenstorrent (n150/n300/p150).

Mirrors the Whisper runner contract (``BaseMetalDeviceRunner``): the device is
opened by ``set_device`` using ``get_pipeline_device_params``; ``warmup`` loads
the model and compiles the traces; ``run`` transcribes one request (optionally
split into VAD/diarization segments by ``audio_manager`` upstream).

Reuses the proven, gated tt-metal demo pipeline
(``models/demos/audio/qwen3_asr/demo/transcribe.py``): Whisper feature-extractor
mel + Qwen3-ASR chat prompt -> TT audio encoder -> baked-in 68 tok/s decoder
(on-device argmax + decode trace + in-graph token/pos + 2CQ, BFP8 KV/attn/lm_head).

STABILITY + LATENCY: the decode trace requires a constant prefill program. All
clips up to ~39s round to the same 512-token prefill, so instead of padding every
request to one fixed 28s length we pad to the smallest of a few audio-length
buckets (see QWEN3_ASR_BUCKETS_SEC). This keeps a single prefill program (trace
stays valid) while letting the encoder input shrink for short clips -- the encoder
attention is quadratic in length, so bucketing is what removes the short-clip
latency tax that a flat 28s pad imposed.
"""

import asyncio
import os
import sys
import time

import numpy as np
import torch
from config.constants import ResponseFormat
from domain.audio_processing_request import AudioProcessingRequest
from domain.audio_text_response import (
    AudioStreamChunk,
    AudioTextResponse,
    AudioTextSegment,
)
from safetensors import safe_open
from transformers import AutoTokenizer, WhisperFeatureExtractor
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

SR = 16000

# Device params (match the demo): 2CQ + trace region for the decode-trace fast path.
QWEN3_ASR_L1_SMALL_SIZE = 32768
QWEN3_ASR_TRACE_REGION_SIZE = 200_000_000
QWEN3_ASR_NUM_COMMAND_QUEUES = 2

# n150: ~28s of audio -> ~364 audio tokens + prompt scaffold < 512, so a clip of
# any length up to ~39s lands in the SAME 512-token prefill program (the decoder
# rounds prefill up to a multiple of 512, min 512). QWEN3_ASR_FIXED_SEC is the top
# bucket / long-audio chunk window.
QWEN3_ASR_FIXED_SEC = float(os.environ.get("QWEN3ASR_FIXED_SEC", "28.0"))
QWEN3_ASR_MAX_NEW_TOKENS = int(os.environ.get("QWEN3ASR_MAX_NEW_TOKENS", "256"))

# Audio-length buckets (seconds). Every request is padded UP to the smallest bucket
# >= its true duration. Because all buckets are <= ~39s they share one prefill
# program (so the decode trace stays valid across mixed-length requests), while the
# encoder input length varies per bucket. The encoder's attention is quadratic in
# length, so a short clip bucketed to 4-8s avoids the bulk of the full-28s encoder
# cost -> much lower short-clip latency. The top bucket is always FIXED_SEC.
QWEN3_ASR_BUCKETS_SEC = tuple(
    sorted(
        {
            *(
                float(x)
                for x in os.environ.get("QWEN3ASR_BUCKETS_SEC", "4,8,16").split(",")
                if x.strip()
            ),
            QWEN3_ASR_FIXED_SEC,
        }
    )
)
DEFAULT_HF_REPO = "Qwen/Qwen3-ASR-1.7B"


def _qwen_demo_root():
    """Locate the tt-metal qwen3_asr demo dir so its `tt/`, `reference/`, `demo/` modules import."""
    candidates = []
    tt_home = os.environ.get("TT_METAL_HOME")
    if tt_home:
        candidates.append(os.path.join(tt_home, "models/demos/audio/qwen3_asr"))
    try:
        import models

        for base in getattr(models, "__path__", []):
            candidates.append(os.path.join(base, "demos/audio/qwen3_asr"))
    except Exception:
        pass
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "demo")):
            return c
    raise RuntimeError(
        "Could not locate models/demos/audio/qwen3_asr (set TT_METAL_HOME)."
    )


_QWEN_ROOT = _qwen_demo_root()
for _sub in ("demo", "tt", "reference"):
    _p = os.path.join(_QWEN_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audio_encoder as tt_enc  # noqa: E402
import audio_encoder_ref as ref  # noqa: E402
import transcribe as tq  # noqa: E402  (find_snap / build_inputs / parse_asr / extract helpers)
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402


class TTQwen3AsrRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.model = None
        self.enc_params = None
        self.embed = None
        self.tok = None
        self.fe = None
        self.chat_template = None

    def get_pipeline_device_params(self):
        return {
            "l1_small_size": QWEN3_ASR_L1_SMALL_SIZE,
            "trace_region_size": QWEN3_ASR_TRACE_REGION_SIZE,
            "num_command_queues": QWEN3_ASR_NUM_COMMAND_QUEUES,
        }

    def load_weights(self):
        # Nothing pipeline-specific to preload; let the service's generic HF
        # snapshot_download fetch settings.model_weights_path (the full audio
        # tower + tokenizer + feature extractor + chat template).
        return False

    def _resolve_snapshot(self):
        """Return the local Qwen3-ASR-1.7B snapshot dir (download via HF cache if needed)."""
        weights = self.settings.model_weights_path or DEFAULT_HF_REPO
        if os.path.isdir(weights) and os.path.exists(
            os.path.join(weights, "config.json")
        ):
            return weights
        try:
            return tq.find_snap()
        except SystemExit:
            from huggingface_hub import snapshot_download

            return snapshot_download(weights)

    async def warmup(self) -> bool:
        try:
            if self.ttnn_device is None:
                raise RuntimeError(
                    "TTNN device not initialized (set_device not called)"
                )
            self.logger.info(f"Device {self.device_id}: Loading Qwen3-ASR-1.7B...")
            await asyncio.to_thread(self._load_model)
            self.logger.info(
                f"Device {self.device_id}: Model loaded; warming traces..."
            )
            await asyncio.to_thread(self._warm)
            self.logger.info(f"Device {self.device_id}: Qwen3-ASR ready.")
            return True
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Model loading failed: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Model loading failed: {str(e)}"
            ) from e

    def _load_model(self):
        import json

        dev = self.ttnn_device
        snap = self._resolve_snapshot()
        ckpt = tq.find_text_decoder(
            snap
        )  # auto-extracts the plain Qwen3 decoder once (cached)
        # tt_transformers' ModelArgs loads the decoder weights/tokenizer from HF_MODEL.
        os.environ["HF_MODEL"] = ckpt
        os.environ["QWEN3ASR_TEXT_DECODER"] = ckpt
        self.logger.info(
            f"Device {self.device_id}: snapshot={snap} text_decoder={ckpt}"
        )

        self.tok = AutoTokenizer.from_pretrained(ckpt)
        with open(os.path.join(snap, "chat_template.json")) as fh:
            self.chat_template = json.load(fh)["chat_template"]
        self.fe = WhisperFeatureExtractor.from_pretrained(snap)
        with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as h:
            self.embed = h.get_tensor("model.embed_tokens.weight").float()

        w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
        self.enc_params = tt_enc.preprocess_weights(w, dev)
        margs = ModelArgs(dev, max_batch_size=1, max_seq_len=2048)
        sd = margs.load_state_dict()
        self.model = Qwen3ASRDecoder(
            margs,
            ttnn.bfloat16,
            dev,
            sd,
            margs.weight_cache_path(ttnn.bfloat16),
            use_paged_kv_cache=False,
        )

    def _warm(self):
        # One pass per bucket compiles that bucket's encoder + prefill programs so the
        # first real request at any length is already warm. The decode kernels/trace
        # compile once and are shared across buckets (prefill program is the same 512).
        for sec in QWEN3_ASR_BUCKETS_SEC:
            dummy = np.zeros(int(round(sec * SR)), dtype=np.float32)
            try:
                self._infer(dummy)
            except Exception as e:
                self.logger.warning(
                    f"Device {self.device_id}: warmup pass ({sec:g}s) skipped: {e}"
                )
                break

    def _bucket_length(self, wav: np.ndarray) -> np.ndarray:
        """Pad up to the smallest bucket >= the clip duration (cap at the top bucket).

        Keeps the prefill in one 512-token program (decode trace stays valid) while
        letting the encoder input length track the audio, so short clips skip the
        bulk of the full-28s encoder cost."""
        dur = len(wav) / SR
        target = next(
            (b for b in QWEN3_ASR_BUCKETS_SEC if dur <= b), QWEN3_ASR_BUCKETS_SEC[-1]
        )
        n = int(round(target * SR))
        if len(wav) >= n:
            return wav[:n]
        return np.concatenate([wav, np.zeros(n - len(wav), dtype=np.float32)])

    def _encode_splice(self, wav: np.ndarray):
        """Bucketed wav -> merged prefill embeddings (1, S, dim) torch: Whisper mel +
        Qwen3-ASR prompt, TT audio encoder, audio embeds spliced at the audio-token
        positions. Shared by the batch (_infer) and streaming (_run_stream) paths."""
        wav = self._bucket_length(np.asarray(wav, dtype=np.float32))
        input_ids, mel = tq.build_inputs(wav, self.fe, self.tok, self.chat_template)
        audio_embeds = tt_enc.encode_mel(mel, self.enc_params, self.ttnn_device).float()
        inp = self.embed[input_ids].clone()
        mask = input_ids == tq.AUDIO_TOKEN_ID
        n_mask = int(mask.sum())
        if audio_embeds.shape[0] > n_mask:
            audio_embeds = audio_embeds[:n_mask]
        elif audio_embeds.shape[0] < n_mask:
            pad = torch.zeros(n_mask - audio_embeds.shape[0], audio_embeds.shape[1])
            audio_embeds = torch.cat([audio_embeds, pad], 0)
        inp[mask] = audio_embeds
        return inp.unsqueeze(0)

    def _infer(self, wav: np.ndarray, max_new_tokens: int = QWEN3_ASR_MAX_NEW_TOKENS):
        """Full TT pipeline on a 16k mono waveform. Returns (lang, text, ntok, t_enc, t_dec)."""
        t0 = time.time()
        inp = self._encode_splice(wav)
        t_enc = time.time() - t0
        t0 = time.time()
        ids = self.model.generate(inp, max_new_tokens=max_new_tokens)
        t_dec = time.time() - t0
        lang, text = tq.parse_asr(self.tok.decode(ids, skip_special_tokens=False))
        return lang, text, len(ids), t_enc, t_dec

    def run(self, requests: list):
        """Synchronous entry point used by the scheduler."""
        return asyncio.run(self._run_async(requests))

    async def _run_async(self, requests: list):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call warmup() first.")
        if not requests:
            raise ValueError("Empty requests list provided")
        request: AudioProcessingRequest = requests[0]
        if request._audio_array is None or len(request._audio_array) == 0:
            raise ValueError("Audio data is empty")

        if getattr(request, "stream", False):
            # device_worker awaits _run_async then iterates the returned async gen.
            return self._run_stream(request)
        if request._segments:
            return await asyncio.to_thread(self._run_segments, request)
        return await asyncio.to_thread(self._run_full, request)

    def _infer_long(self, wav: np.ndarray):
        """Transcribe an arbitrary-length window. The decode trace needs a bounded
        prefill, so split into <=FIXED_SEC chunks, transcribe each, and join. Any
        caller-supplied window (a full clip or one VAD/diarization segment) can be
        longer than the top bucket, so this is the single place that guards against
        silently truncating the tail. Returns (text, ntok, t_dec)."""
        parts, ntok, t_dec = [], 0, 0.0
        for chunk in tq.chunk_wav(
            np.asarray(wav, dtype=np.float32), QWEN3_ASR_FIXED_SEC
        ):
            if chunk is None or len(chunk) == 0:
                continue
            _lang, text, nt, _te, td = self._infer(chunk)
            if text:
                parts.append(text)
            ntok += nt
            t_dec += td
        return " ".join(parts), ntok, t_dec

    def _run_full(self, request: AudioProcessingRequest):
        """No pre-segmentation: chunk long audio into fixed windows, transcribe, join."""
        text, ntok, t_dec = self._infer_long(request._audio_array)
        self.logger.info(
            f"Device {self.device_id}: qwen3-asr {request._duration:.1f}s -> {ntok} tok "
            f"({ntok / max(t_dec, 1e-6):.1f} tok/s decode)"
        )
        return [AudioTextResponse(text=text, duration=request._duration)]

    def _run_segments(self, request: AudioProcessingRequest):
        """VAD/diarization segments: transcribe each, keep speaker labels + spans."""
        sr = self.settings.default_sample_rate
        segments, full_text, speakers = [], [], set()
        for i, seg in enumerate(request._segments):
            start_t, end_t = float(seg["start"]), float(seg["end"])
            speaker = seg.get("speaker", f"SPEAKER_{i:02d}")
            seg_audio = request._audio_array[int(start_t * sr) : int(end_t * sr)]
            if len(seg_audio) == 0:
                continue
            text, _nt, _td = self._infer_long(seg_audio)
            segments.append(
                AudioTextSegment(
                    id=i, speaker=speaker, start_time=start_t, end_time=end_t, text=text
                )
            )
            if text:
                full_text.append(text)
            speakers.add(speaker)
        return [
            AudioTextResponse(
                text=" ".join(full_text),
                duration=request._duration,
                segments=segments,
                speaker_count=len(speakers),
                speakers=sorted(speakers),
            )
        ]

    _ASR_MARKER = "<asr_text>"

    def _stream_window_text(self, wav: np.ndarray, max_new_tokens: int):
        """Yield transcription-text deltas for one audio window. Decodes the growing
        id sequence each step and emits only what follows the <asr_text> marker, so
        the leading `language <Lang>` scaffold is never streamed."""
        inp = self._encode_splice(wav)
        ids, emitted = [], 0
        for tid in self.model.generate_iter(inp, max_new_tokens=max_new_tokens):
            ids.append(tid)
            full = self.tok.decode(ids, skip_special_tokens=False)
            if self._ASR_MARKER not in full:
                continue
            text = full.split(self._ASR_MARKER, 1)[1]
            if len(text) > emitted:
                yield text[emitted:]
                emitted = len(text)

    async def _run_stream(self, request: AudioProcessingRequest):
        """Async generator emitting SSE chunks then a final result, matching the
        Whisper runner's streaming protocol so the shared endpoint/service handle it
        identically. device_worker serialises streaming (one request at a time), so
        the blocking device work runs inline; each yielded token hands control back
        to the worker which forwards the chunk to the client."""
        if request._segments:
            sr = self.settings.default_sample_rate
            raw = [
                request._audio_array[
                    int(float(s["start"]) * sr) : int(float(s["end"]) * sr)
                ]
                for s in request._segments
            ]
        else:
            raw = [request._audio_array]
        # Split each window into <=FIXED_SEC trace-safe chunks (a single segment can
        # exceed the top bucket; _bucket_length would otherwise truncate its tail).
        windows = []
        for r in raw:
            windows.extend(
                tq.chunk_wav(np.asarray(r, dtype=np.float32), QWEN3_ASR_FIXED_SEC)
            )

        chunk_id, parts = 0, []
        for win in windows:
            if win is None or len(win) == 0:
                continue
            seg = []
            for delta in self._stream_window_text(win, QWEN3_ASR_MAX_NEW_TOKENS):
                seg.append(delta)
                chunk_id += 1
                yield {
                    "type": "streaming_chunk",
                    "chunk": AudioStreamChunk(text=delta, chunk_id=chunk_id),
                    "task_id": request._task_id,
                }
            joined = "".join(seg).strip()
            if joined:
                parts.append(joined)

        final = AudioTextResponse(text=" ".join(parts), duration=request._duration)
        yield {
            "type": "final_result",
            "result": final,
            "task_id": request._task_id,
            "return": request.response_format.lower() != ResponseFormat.TEXT.value,
        }
