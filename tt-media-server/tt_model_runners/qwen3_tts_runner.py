# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Qwen3-TTS media-server runner.

N150: ``ttnn.open_device`` (mesh CQ rejects H2D during trace capture).
N300 TP=2: mesh ``(1, 2)`` + ``FABRIC_1D``.
Voice: ``speaker_id`` (default ``jim``) or ``voice_clone_audio`` + ``voice_clone_text``.
"""

from __future__ import annotations

import os

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
from config.settings import settings
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.logger import log_exception_chain
from utils.voice_prompts import DEFAULT_VOICE_ID, VoicePromptManager

_DEFAULT_HF_ID = SupportedModels.QWEN3_TTS_1_7B.value
_KNOWN_LANGUAGES = (
    "english",
    "chinese",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "japanese",
    "korean",
    "french",
    "russian",
)
SAMPLE_RATE_HZ = 24000


def _looks_japanese(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)


def _tts_api():
    from models.demos.qwen3_tts.tt import server as api

    return api


class Qwen3TTSConstants:
    L1_SMALL_SIZE = 32768
    TRACE_REGION_SIZE = 512_000_000
    NUM_COMMAND_QUEUES = 2
    MAX_NEW_TOKENS = 256


class TTQwen3TTSRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        os.environ.pop("TT_MM_THROTTLE_PERF", None)
        os.environ.pop("TT_METAL_CACHE", None)

        self.model = None
        self.ctx = None
        self.tokenizer = None
        self.main_weights = None
        self.decoder_weights = None
        self.config = None
        self.voice_prompts: Optional[VoicePromptManager] = None
        self._post_warmup_rng_state = None
        self._opened_mesh = False
        self.hf_id = self._resolve_hf_id()

        rows, cols = self.settings.device_mesh_shape
        self.is_tensor_parallel = (rows * cols) > 1
        if self.is_tensor_parallel:
            os.environ.pop("TT_METAL_FABRIC_DISABLE", None)
        elif not settings.is_galaxy:
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    def _resolve_hf_id(self) -> str:
        weights = self.settings.model_weights_path or _DEFAULT_HF_ID
        name = Path(weights).name
        if "0.6B" in name:
            return SupportedModels.QWEN3_TTS_0_6B.value
        if "1.7B" in name:
            return SupportedModels.QWEN3_TTS_1_7B.value
        if isinstance(weights, str) and weights.startswith("Qwen/"):
            return weights
        return _DEFAULT_HF_ID

    def get_pipeline_device_params(self):
        import ttnn

        device_params = {
            "l1_small_size": Qwen3TTSConstants.L1_SMALL_SIZE,
            "trace_region_size": Qwen3TTSConstants.TRACE_REGION_SIZE,
            "num_command_queues": Qwen3TTSConstants.NUM_COMMAND_QUEUES,
        }
        if self.is_tensor_parallel:
            device_params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
        return device_params

    def _configure_fabric(self, updated_device_params):
        import ttnn

        try:
            fabric_config = updated_device_params.pop("fabric_config", None)
            if fabric_config:
                ttnn.set_fabric_config(fabric_config)
            return fabric_config
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Fabric configuration failed", e
            )
            raise RuntimeError(f"Fabric configuration failed: {str(e)}") from e

    def set_device(self):
        """N150: plain ``open_device`` (trace H2D). N300 TP=2: mesh (1, 2)."""
        if self.is_tensor_parallel:
            self._opened_mesh = True
            return super().set_device()

        import ttnn

        if self.ttnn_device is None:
            params = self.get_updated_device_params(self.get_pipeline_device_params())
            params.pop("dispatch_core_config", None)
            params.pop("fabric_config", None)
            self.ttnn_device = ttnn.open_device(device_id=0, **params)
            self.ttnn_device.enable_program_cache()
        self.max_batch_size = self.settings.max_batch_size
        return self.ttnn_device

    def close_device(self):
        import ttnn

        try:
            if self.ttnn_device is None:
                return True
            if self._opened_mesh:
                ttnn.close_mesh_device(self.ttnn_device)
            else:
                ttnn.close_device(self.ttnn_device)
            self.ttnn_device = None
            return True
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise

    def _load_qwen_weights(self):
        return _tts_api().load_weights(self.hf_id)

    def load_weights(self) -> bool:
        self.logger.info(
            f"Device {self.device_id}: Prefetching Qwen3-TTS weights ({self.hf_id})"
        )
        self.main_weights, self.decoder_weights = self._load_qwen_weights()
        return True

    def _language_for(self, request: TextToSpeechRequest) -> str:
        env_lang = os.environ.get("QWEN3_TTS_LANGUAGE", "").strip().lower()
        speaker = (request.speaker_id or "").strip().lower()
        if speaker in _KNOWN_LANGUAGES:
            return speaker
        if env_lang in _KNOWN_LANGUAGES:
            return env_lang
        if _looks_japanese(request.text):
            return "japanese"
        return "english"

    def _initialize_models(self) -> None:
        from transformers import AutoTokenizer

        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        api = _tts_api()
        if self.ttnn_device is None:
            raise RuntimeError("ttnn_device not initialized; set_device() must run first")

        if self.main_weights is None or self.decoder_weights is None:
            self.main_weights, self.decoder_weights = self._load_qwen_weights()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True
        )

        from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id

        talker_config = talker_config_for_hf_id(self.hf_id)
        model_kwargs = {
            "device": self.ttnn_device,
            "state_dict": self.main_weights,
            "talker_config": talker_config,
        }

        self.logger.info(
            f"Device {self.device_id}: Building Qwen3TTS ({self.hf_id}) "
            f"TT_QWEN3_CP_FP32={os.environ.get('TT_QWEN3_CP_FP32', '0')}"
        )
        self.model = Qwen3TTS(**model_kwargs)

        max_new = int(
            os.environ.get("TT_QWEN3_MAX_NEW_TOKENS", Qwen3TTSConstants.MAX_NEW_TOKENS)
        )
        self.config = api.TTSConfig(max_new_tokens=max_new)
        self.config.greedy = False
        self.config.repetition_penalty = float(
            os.environ.get("TT_QWEN3_REP_PENALTY", "1.15")
        )
        self.config.hidden_size = talker_config.hidden_size

        self.logger.info(
            f"Device {self.device_id}: Capturing TTS server context (traces)..."
        )
        self.ctx = api.init_server_context(
            self.ttnn_device, self.model, self.config, self.main_weights
        )

        self.voice_prompts = VoicePromptManager()
        self.voice_prompts.preload()
        self.voice_prompts.precompute_speaker_embeddings(self.model)
        self.logger.info(
            f"Device {self.device_id}: Voice prompts ready: "
            f"{self.voice_prompts.list_available()}"
        )
        self._post_warmup_rng_state = torch.get_rng_state()

    @log_execution_time(
        "Qwen3-TTS warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        try:
            if self.ttnn_device is None:
                raise ValueError("Device not initialized. Call set_device() first.")
            await asyncio.to_thread(self._initialize_models)
            self.logger.info(f"Device {self.device_id}: Qwen3-TTS warmup complete")
            return True
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Qwen3-TTS load failed: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Model loading failed: {str(e)}"
            ) from e

    def _resolve_voice(
        self, request: TextToSpeechRequest
    ) -> Tuple[torch.Tensor, str, torch.Tensor, str]:
        api = _tts_api()
        clone_audio_b64 = request.voice_clone_audio
        clone_text = request.voice_clone_text
        if clone_audio_b64 and clone_text:
            self.logger.info("Voice resolution: ad-hoc clone from request payload")
            audio_bytes = base64.b64decode(clone_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                ref_codes, audio_data = api.encode_reference_audio(
                    tmp_path, main_weights=None
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            return ref_codes, clone_text, audio_data, "<adhoc>"

        voice_id = request.speaker_id or DEFAULT_VOICE_ID
        if voice_id in _KNOWN_LANGUAGES:
            voice_id = DEFAULT_VOICE_ID
        prompt = self.voice_prompts.get(voice_id) if self.voice_prompts else None
        if prompt is None:
            available = (
                self.voice_prompts.list_available() if self.voice_prompts else []
            )
            raise ValueError(
                f"Unknown voice_id={voice_id!r}. Available: {available}. "
                "Or pass voice_clone_audio + voice_clone_text for ad-hoc cloning."
            )
        return prompt.ref_codes, prompt.ref_text, prompt.audio_data, voice_id

    def _trim_ref(self, ref_codes, audio_data, ref_text, target_text):
        from models.demos.qwen3_tts.demo.reference_icl_utils import (
            trim_reference_for_icl_conditioning,
        )

        return trim_reference_for_icl_conditioning(
            ref_codes, audio_data, self.tokenizer, ref_text, target_text
        )

    def _synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        import time as _time

        api = _tts_api()
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model not loaded. Call warmup() first.")

        t_total = _time.perf_counter()
        ref_codes, ref_text, audio_data, voice_id = self._resolve_voice(request)
        ref_codes, audio_data = self._trim_ref(
            ref_codes, audio_data, ref_text, request.text
        )

        cached = (
            self.voice_prompts.get(voice_id)
            if (self.voice_prompts and voice_id != "<adhoc>")
            else None
        )
        if cached is not None and cached.speaker_embedding is not None:
            speaker_embedding = cached.speaker_embedding
        else:
            speaker_embedding = self.model.extract_speaker_embedding(audio_data)

        language = self._language_for(request)
        inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = (
            api.create_icl_embedding_ttnn(
                target_text=request.text,
                ref_text=ref_text,
                ref_codes=ref_codes,
                speaker_embedding=speaker_embedding,
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.ttnn_device,
                config=self.config,
                main_weights=self.main_weights,
                language=language,
            )
        )

        if self._post_warmup_rng_state is not None:
            torch.set_rng_state(self._post_warmup_rng_state)

        codes, _timings, _perf = api.run_inference(
            ctx=self.ctx,
            model=self.model,
            device=self.ttnn_device,
            inputs_embeds_tt=inputs_embeds_tt,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            config=self.config,
            use_2cq=True,
        )
        if codes is None:
            raise RuntimeError("Qwen3-TTS generation returned no codec frames")

        if (
            getattr(self.config, "trim_codec_frames", 0) > 0
            and len(codes) > self.config.trim_codec_frames
        ):
            codes = codes[self.config.trim_codec_frames :]

        audio = api.decode_audio(codes, self.decoder_weights)
        audio_np = audio.squeeze().detach().cpu().float().numpy()
        duration_s = float(len(audio_np)) / SAMPLE_RATE_HZ

        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE_HZ, format="WAV")
        b64_audio = base64.b64encode(buf.getvalue()).decode("ascii")

        total_ms = (_time.perf_counter() - t_total) * 1000
        rtf = (total_ms / 1000.0) / duration_s if duration_s > 0 else float("inf")
        self.logger.info(
            f"Device {self.device_id}: voice={voice_id} lang={language} "
            f"frames={len(codes)} audio={duration_s:.2f}s "
            f"total={total_ms:.0f}ms RTF={rtf:.3f}"
        )
        return TextToSpeechResponse(
            audio=b64_audio,
            duration=duration_s,
            sample_rate=SAMPLE_RATE_HZ,
            format="wav",
            speaker_id=None if voice_id == "<adhoc>" else voice_id,
        )

    async def _run_async(self, requests: list[TextToSpeechRequest]):
        if not requests:
            raise ValueError("Empty request list")
        if len(requests) > 1:
            self.logger.warning(
                f"Device {self.device_id}: Qwen3-TTS supports batch=1; "
                f"processing first of {len(requests)} requests"
            )
        request = requests[0]
        if request is None or not request.text or not request.text.strip():
            raise ValueError("Text cannot be empty")
        return await asyncio.to_thread(self._synthesize, request)

    @log_execution_time(
        "Qwen3-TTS inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[TextToSpeechRequest]):
        result = asyncio.run(self._run_async(requests))
        return [result] if result is not None else []
