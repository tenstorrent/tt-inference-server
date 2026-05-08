# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Qwen3-TTS runner for tt-media-server.

Wraps the production tt-metal Qwen3-TTS pipeline (traced Talker + traced
fp32 Code Predictor) behind the model-agnostic ``BaseMetalDeviceRunner`` /
TextToSpeechService contract. Defaults ``TT_QWEN3_CP_FP32=1`` for voice-clone
fidelity (RTF ~0.733× at 12.5 fps codec).

Inherits device lifecycle (``set_device``, ``close_device``), HF-env wiring,
``self.logger`` / ``self.settings``, and telemetry hooks from the base class.
The runner only implements TTS-specific bits: weight load, model init,
warmup (which captures all CP+Talker traces), and the request handler.

Reused tt-metal helpers — no copy-paste:
  - ``load_weights``                (demo_full_ttnn_tts.py:148)
  - ``encode_reference_audio``      (demo_full_ttnn_tts.py:172)  -- caches via .refcache.pt
  - ``create_icl_embedding_ttnn``   (demo_full_ttnn_tts.py:244)
  - ``init_server_context``         (demo_full_ttnn_tts.py:2259) -- pre-captures CP traces
  - ``run_inference``               (demo_full_ttnn_tts.py:2549) -- AR loop with EOS handling
  - ``decode_audio``                (demo_full_ttnn_tts.py:2948) -- 24 kHz CPU Mimi decode
  - ``Qwen3TTS.extract_speaker_embedding`` (TTNN ECAPA on device)
"""
import os

# Default voice-clone-fidelity ON; user can opt out by setting "0" before launch.
os.environ.setdefault("TT_QWEN3_CP_FP32", "1")

import asyncio
import base64
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import soundfile as sf
import torch

from config.constants import SupportedModels
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.voice_prompts import VoicePromptManager


SAMPLE_RATE_HZ = 24000
DEFAULT_VOICE_ID = "jim"


class TTQwen3TTSRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        # Override env vars set by runner_utils.setup_runner_environment that
        # break qwen3_tts trace capture. The qwen3_tts demo runs without any
        # of these set; reproducing that environment in the worker is the
        # cleanest way to make init_server_context's trace path succeed.
        os.environ.pop("TT_MM_THROTTLE_PERF", None)
        # Use the same JIT cache layout as the standalone demo (auto-derived)
        # rather than the worker-specific path that runner_utils sets, so any
        # stale per-worker cache entries from prior runs don't replay.
        os.environ.pop("TT_METAL_CACHE", None)

        # Per-instance state, populated in _initialize_models() / warmup().
        self.model = None              # tt qwen3_tts.Qwen3TTS
        self.ctx = None                # demo_full_ttnn_tts.TTSServerContext
        self.tokenizer = None
        self.main_weights = None
        self.decoder_weights = None    # CPU Mimi decoder weights (fp32)
        self.config = None             # demo_full_ttnn_tts.TTSConfig
        self.voice_prompts: Optional[VoicePromptManager] = None
        # 2 CQ overlaps H2D on CQ1 with trace exec on CQ0 — fastest single-shot,
        # but leaves pending CQ1 events that wedge the worker on the synchronize_device
        # in run_inference's finally-block on the SECOND request. Disable for now;
        # see task #42. Per-frame perf hit is ~5%.
        self._use_2cq = True

        # Single-device runs: tt-metal fabric only matters on Galaxy.
        if not self.settings.is_galaxy:
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    # ─── Device params (consumed by BaseMetalDeviceRunner._mesh_device) ───
    def get_pipeline_device_params(self):
        # Trace region must hold all Talker prefill traces (3 buckets) + 14 CP
        # decode traces + CP prefill trace + Talker decode trace.
        # 100 MB is what the standalone demo (web_demo.py:90) uses.
        return {
            "l1_small_size": 32768,
            "trace_region_size": 512_000_000,
            "num_command_queues": 2 if self._use_2cq else 1,
        }

    def set_device(self):
        """Open a single TT device (not a mesh).

        ``init_server_context`` from the qwen3_tts demo issues
        ``copy_host_to_device_tensor`` calls inside the trace-capture region.
        The mesh-command-queue rejects writes during trace capture, while the
        single-device CQ permits them. Single Qwen3-TTS only runs on one chip
        (no tensor parallelism), so a plain device is appropriate.
        """
        import ttnn

        if self.ttnn_device is None:
            params = self.get_updated_device_params(self.get_pipeline_device_params())
            # Strip mesh-only kwargs that ttnn.open_device does not accept.
            params.pop("dispatch_core_config", None)
            self.ttnn_device = ttnn.open_device(device_id=0, **params)
            self.ttnn_device.enable_program_cache()
        self.max_batch_size = self.settings.max_batch_size
        return self.ttnn_device

    def close_device(self):
        import ttnn

        try:
            self.logger.info(f"Device {self.device_id}: Closing TT device...")
            if self.ttnn_device is not None:
                ttnn.close_device(self.ttnn_device)
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed TT device"
                )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise

    # ─── Weight download path used by run.py download_weights flow ───
    def load_weights(self) -> bool:
        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import load_weights

        self.logger.info(
            f"Device {self.device_id}: Downloading Qwen3-TTS weights "
            f"({SupportedModels.QWEN3_TTS.value})"
        )
        self.main_weights, self.decoder_weights = load_weights()
        self.logger.info(
            f"Device {self.device_id}: Loaded "
            f"{len(self.main_weights)} main + {len(self.decoder_weights)} decoder weights"
        )
        return True

    # ─── Model + traces (heavy CPU + device work; runs in a thread) ───
    def _initialize_models(self) -> None:
        from transformers import AutoTokenizer

        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
            TTSConfig,
            init_server_context,
            load_weights,
        )
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        if self.ttnn_device is None:
            raise RuntimeError("ttnn_device not initialized; set_device() must run first")

        # Idempotent load (load_weights() may have run already during the
        # download_weights flow with device_id="-1"; on the worker we still
        # need them in memory).
        if self.main_weights is None or self.decoder_weights is None:
            self.main_weights, self.decoder_weights = load_weights()

        self.logger.info(f"Device {self.device_id}: Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            SupportedModels.QWEN3_TTS.value, trust_remote_code=True
        )

        self.logger.info(
            f"Device {self.device_id}: Building Qwen3TTS model "
            f"(TT_QWEN3_CP_FP32={os.environ.get('TT_QWEN3_CP_FP32', '0')})"
        )
        self.model = Qwen3TTS(device=self.ttnn_device, state_dict=self.main_weights)

        # max_new_tokens=1500 + repetition_penalty=1.15 mirror web_demo.py
        self.config = TTSConfig(max_new_tokens=1500)
        self.config.repetition_penalty = 1.15

        self.logger.info(
            f"Device {self.device_id}: Capturing TTS server context "
            "(prefill bucket warmup + CP traces)..."
        )
        self.ctx = init_server_context(
            self.ttnn_device, self.model, self.config, self.main_weights
        )

        # Pre-encode the built-in voice prompts via Mimi (cached on disk via .refcache.pt).
        self.voice_prompts = VoicePromptManager()
        self.voice_prompts.preload()
        # Pre-run ECAPA on each registered voice once. The on-device conv2d
        # weights need re-preparation per call, and that path occasionally
        # stalls on the second request — caching the embedding sidesteps it.
        self.voice_prompts.precompute_speaker_embeddings(self.model)
        # Keep on-device ECAPA on the UNTRACED path. After the host-conv1d
        # round-trip fix in _conv1d_same_padding (torch weights passed
        # directly), post-trace-exec untraced ECAPA gives cos≈0.98 with the
        # cached embedding — close enough for the model to converge.
        self.logger.info(
            f"Device {self.device_id}: Voice prompts ready: "
            f"{self.voice_prompts.list_available()}"
        )

    @log_execution_time(
        "Qwen3-TTS warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        device_id_int = int(self.device_id) if self.device_id else 0
        self.logger.info(f"Device {device_id_int}: Loading Qwen3-TTS model...")
        try:
            await asyncio.to_thread(self._initialize_models)
        except Exception as e:
            self.logger.error(f"Device {device_id_int}: Warmup failed: {e}")
            raise RuntimeError(f"Qwen3-TTS warmup failed: {e}") from e
        self.logger.info(f"Device {device_id_int}: Qwen3-TTS warmup complete")
        return True

    # ─── Request → audio flow ────────────────────────────────────────────
    def _resolve_voice(
        self, request: TextToSpeechRequest
    ) -> Tuple[torch.Tensor, str, torch.Tensor, str]:
        """Returns (ref_codes, ref_text, audio_data, voice_id).

        Priority:
          1. ad-hoc cloning  — base64 ``voice_clone_audio`` + ``voice_clone_text``
          2. registered      — ``speaker_id`` looked up in VoicePromptManager
          3. default         — ``DEFAULT_VOICE_ID`` from VoicePromptManager
        """
        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import encode_reference_audio

        clone_audio_b64 = getattr(request, "voice_clone_audio", None)
        clone_text = getattr(request, "voice_clone_text", None)
        if clone_audio_b64 and clone_text:
            self.logger.info("Voice resolution: ad-hoc clone from request payload")
            audio_bytes = base64.b64decode(clone_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                ref_codes, audio_data = encode_reference_audio(tmp_path, main_weights=None)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            return ref_codes, clone_text, audio_data, "<adhoc>"

        voice_id = request.speaker_id or DEFAULT_VOICE_ID
        prompt = self.voice_prompts.get(voice_id) if self.voice_prompts else None
        if prompt is None:
            available = self.voice_prompts.list_available() if self.voice_prompts else []
            raise ValueError(
                f"Unknown voice_id={voice_id!r}. Available: {available}. "
                "Or pass voice_clone_audio + voice_clone_text for ad-hoc cloning."
            )
        return prompt.ref_codes, prompt.ref_text, prompt.audio_data, voice_id

    async def _generate(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
            create_icl_embedding_ttnn,
            decode_audio,
            run_inference,
        )
        from models.demos.qwen3_tts.demo.reference_icl_utils import (
            trim_reference_for_icl_conditioning,
        )

        if self.model is None or self.ctx is None:
            raise RuntimeError("Model not warmed up; warmup() must run first")

        # 1. Resolve voice prompt (ref_codes + ref_text + raw audio for ECAPA).
        ref_codes, ref_text, audio_data, voice_id = self._resolve_voice(request)

        # 2. Trim ref_codes / audio_data so target text fits ICL window. The
        # extracted speaker_embedding must match the trimmed reference codes —
        # if extracted from the full audio while ref_codes are trimmed, the
        # model fails to converge to EOS.
        ref_codes, audio_data = trim_reference_for_icl_conditioning(
            ref_codes, audio_data, self.tokenizer, ref_text, request.text
        )

        # 3. Speaker embedding. Registered voices use a cached vector
        # (computed at init from the un-trimmed audio); ad-hoc clones extract
        # on the device from the trimmed audio to match the trimmed codes.
        cached_prompt = (
            self.voice_prompts.get(voice_id)
            if (self.voice_prompts and voice_id != "<adhoc>")
            else None
        )
        if cached_prompt is not None and cached_prompt.speaker_embedding is not None:
            speaker_embedding = cached_prompt.speaker_embedding
        else:
            speaker_embedding = self.model.extract_speaker_embedding(audio_data)

        # 4. ICL embedding build (TTNN).
        inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = create_icl_embedding_ttnn(
            target_text=request.text,
            ref_text=ref_text,
            ref_codes=ref_codes,
            speaker_embedding=speaker_embedding,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.ttnn_device,
            config=self.config,
            main_weights=self.main_weights,
        )

        # 5. AR generation (uses pre-captured CP traces).
        codes, timings, _perf_text = run_inference(
            ctx=self.ctx,
            model=self.model,
            device=self.ttnn_device,
            inputs_embeds_tt=inputs_embeds_tt,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            config=self.config,
            use_2cq=self._use_2cq,
        )
        if codes is None:
            raise RuntimeError("Generation produced no codes (EOS at prefill?)")

        # 6. Trim leading reference echo (default 4 frames = 0.32 s).
        if self.config.trim_codec_frames > 0 and len(codes) > self.config.trim_codec_frames:
            codes = codes[self.config.trim_codec_frames :]

        # 7. Decode codes → 24 kHz audio (CPU Mimi).
        audio = decode_audio(codes, self.decoder_weights)
        audio_np = audio.squeeze().detach().cpu().float().numpy()
        duration_s = float(len(audio_np)) / SAMPLE_RATE_HZ

        # 8. Pack as base64 WAV (service handles MP3 / OGG re-encoding).
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE_HZ, format="WAV")
        b64_audio = base64.b64encode(buf.getvalue()).decode("ascii")

        self.logger.info(
            f"Device {self.device_id}: voice={voice_id} frames={len(codes)} "
            f"audio={duration_s:.2f}s prefill={timings.get('prefill', 0):.0f}ms "
            f"decode_loop={timings.get('decode_loop', 0):.0f}ms"
        )

        return TextToSpeechResponse(
            audio=b64_audio,
            duration=duration_s,
            sample_rate=SAMPLE_RATE_HZ,
            format="wav",
            speaker_id=None if voice_id == "<adhoc>" else voice_id,
        )

    async def _run_async(self, requests):
        if not requests:
            raise ValueError("Empty request list")
        if len(requests) > 1:
            self.logger.warning(
                f"Device {self.device_id}: Qwen3-TTS supports batch=1; "
                f"processing first of {len(requests)} requests"
            )
        return await self._generate(requests[0])

    @log_execution_time(
        "Qwen3-TTS inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests):
        result = asyncio.run(self._run_async(requests))
        return [result] if result is not None else []
