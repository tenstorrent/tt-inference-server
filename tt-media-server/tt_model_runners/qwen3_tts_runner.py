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
    # The native-conv1d decoder (TT_QWEN3_DECODE_CONV=conv1d) needs 131072 for its
    # sharding config buffers; the default matmul decoder fits in 32768.
    L1_SMALL_SIZE = int(os.environ.get("TT_QWEN3_L1_SMALL_SIZE", "32768"))
    TRACE_REGION_SIZE = 512_000_000
    NUM_COMMAND_QUEUES = 2
    MAX_NEW_TOKENS = 256
    # Longest reference (in 12.5 fps codec frames, so 150 ~= 12s) the device decoder
    # warms buckets for. Ad-hoc voice clones above this are rejected, not decoded.
    MAX_REF_FRAMES = 150


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
        self.device_decoder = None
        # Decode the speech tokenizer on device by default. Set
        # TT_QWEN3_DEVICE_DECODE=0 to fall back to the CPU decode_icl_audio.
        #
        # The decoder shares the chip with the traced talker, so it is built and
        # warmed before the talker's traces are captured (see _initialize_models).
        #
        # TT_QWEN3_DEVICE_DECODE_MODE:
        #   continue (default) -- decode_icl_audio with the conv back-end on device:
        #       the host runs the exact front-end from a cached per-voice reference
        #       state, and the device decodes 12 context + the generated frames.
        #   full -- decode_audio_device: the whole decoder on device over
        #       cat(ref, generated), then the reference's audio is cut off.
        # TT_QWEN3_DECODE_COMPARE=1 logs device-vs-CPU SNR per request.
        self._use_device_decode = os.environ.get("TT_QWEN3_DEVICE_DECODE", "1") == "1"
        self._decode_continue = (
            os.environ.get("TT_QWEN3_DEVICE_DECODE_MODE", "continue") == "continue"
        )
        # voice_id -> (ref_codes, prepare_icl_decoder_state(ref_codes)), host only.
        self._ref_states = {}
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
            raise RuntimeError(
                "ttnn_device not initialized; set_device() must run first"
            )

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

        # Host-only (reference codes come from the CPU encoder / refcache), so it can
        # run before anything touches the device.
        self.voice_prompts = VoicePromptManager()
        self.voice_prompts.preload()
        for voice_id in self.voice_prompts.list_available():
            self._ref_state_for(
                api, voice_id, self.voice_prompts.get(voice_id).ref_codes
            )

        if self._use_device_decode:
            # The decoder MUST be built and warmed BEFORE init_server_context captures
            # the talker/CP/ECAPA traces. A trace replays its captured ops at the
            # addresses its intermediates had during capture; those buffers are freed
            # afterwards, so anything allocated while a trace exists can land on them
            # and is overwritten every time the trace executes (tt-metal warns:
            # "Allocating device buffers is unsafe due to the existence of an active
            # trace"). Built after capture, the decoder scored +19 dB before the first
            # request and -21 dB after it. Every persistent decoder tensor (weights,
            # prepared conv weights, RoPE tables) is created by the warmup below, and
            # the frozen cache guarantees no request allocates a new one.
            #
            # Built after the voice prompts so the warmed buckets can cover the
            # reference too: decode_audio_device decodes cat([ref_codes, codes]), so
            # sizing buckets off the generated count alone leaves live requests
            # un-warmed. Ad-hoc clones carry a caller-supplied reference, so allow
            # headroom past the presets; a longer one is then rejected by the frozen
            # decoder instead of preparing conv weights on an already-traced device.
            preset_ref_frames = 0
            for voice_id in self.voice_prompts.list_available():
                prompt = self.voice_prompts.get(voice_id)
                if prompt is not None:
                    preset_ref_frames = max(
                        preset_ref_frames, int(prompt.ref_codes.shape[0])
                    )
            max_ref_frames = max(
                preset_ref_frames,
                int(
                    os.environ.get(
                        "TT_QWEN3_MAX_REF_FRAMES", Qwen3TTSConstants.MAX_REF_FRAMES
                    )
                ),
            )
            self.logger.info(
                f"Device {self.device_id}: Building on-device speech decoder "
                f"(ref<={max_ref_frames} + gen<={max_new} frames)..."
            )
            self.device_decoder, warm_buckets = api.prepare_device_decoder(
                self.ttnn_device,
                self.decoder_weights,
                max_ref_frames,
                max_new,
                icl_continue=self._decode_continue,
            )
            self.logger.info(
                f"Device {self.device_id}: On-device speech decoder ready "
                f"({'continue' if self._decode_continue else 'full'} mode); "
                f"warmed and froze buckets {warm_buckets}"
            )
            if os.environ.get("TT_QWEN3_DECODE_COMPARE", "0") == "1":
                self._check_decoder_against_cpu("pre-capture")

        self.logger.info(
            f"Device {self.device_id}: Capturing TTS server context (traces)..."
        )
        self.ctx = api.init_server_context(
            self.ttnn_device, self.model, self.config, self.main_weights
        )

        self.voice_prompts.precompute_speaker_embeddings(self.model)
        self.logger.info(
            f"Device {self.device_id}: Voice prompts ready: "
            f"{self.voice_prompts.list_available()}"
        )
        if (
            self._use_device_decode
            and os.environ.get("TT_QWEN3_DECODE_COMPARE", "0") == "1"
        ):
            # The [decode-compare] lines of the warm-up requests then show whether the
            # decoder survives the talker's traced execution.
            self._check_decoder_against_cpu("post-capture")

        # Every request restarts from this RNG state, so one eval run is one fixed draw
        # per prompt. TT_QWEN3_SEED picks a different (still reproducible) draw, to tell a
        # quality regression from one unlucky sample.
        if os.environ.get("TT_QWEN3_SEED"):
            torch.manual_seed(int(os.environ["TT_QWEN3_SEED"]))
        self._post_warmup_rng_state = torch.get_rng_state()

    def _warmup_inference(self) -> None:
        # init_server_context captures load traces only. First live /v1/audio/speech
        # still JIT-compiles ICL + decode for that language/length (~12s per worker).
        for text in ("テスト", "Hello, this is a test."):
            self.logger.info(f"Device {self.device_id}: Warm-up synth text={text!r}")
            self._synthesize(TextToSpeechRequest(text=text, response_format="wav"))

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
            await asyncio.to_thread(self._warmup_inference)
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

    def _ref_state_for(self, api, voice_id: str, ref_codes: torch.Tensor):
        """Cached decoder front-end state for a preset voice; None for ad-hoc clones."""
        cached = self._ref_states.get(voice_id)
        if cached is not None and torch.equal(cached[0], ref_codes):
            return cached[1]
        if voice_id == "<adhoc>":
            return None
        state = api.prepare_icl_decoder_state(ref_codes, self.decoder_weights)
        self._ref_states[voice_id] = (ref_codes, state)
        return state

    def _device_decode(self, api, ref_codes, codes, ref_state=None) -> torch.Tensor:
        """Device decode of generated ``codes`` (reference as context), generated audio only."""
        if self._decode_continue:
            return api.decode_icl_audio(
                ref_codes,
                codes,
                self.decoder_weights,
                ref_state=ref_state,
                device_decoder=self.device_decoder,
            )
        return api.decode_audio_device(ref_codes, codes, self.device_decoder)

    def _check_decoder_against_cpu(self, label: str) -> None:
        """Score the device decoder against the CPU reference on the built-in prompt.

        Splits the default voice's reference codes into a 20-frame "reference" and a
        "generated" continuation and decodes them through the serving path, so it only
        touches warmed buckets in either mode.
        """
        import math

        try:
            prompt = (
                self.voice_prompts.get(DEFAULT_VOICE_ID) if self.voice_prompts else None
            )
            if prompt is None:
                self.logger.warning(
                    f"[decoder-check:{label}] no default voice prompt, skipped"
                )
                return
            api = _tts_api()
            ref, gen = prompt.ref_codes[:20], prompt.ref_codes[20:]
            dev = (
                self._device_decode(api, ref, gen)
                .squeeze()
                .detach()
                .cpu()
                .float()
                .flatten()
            )
            cpu = (
                api.decode_icl_audio(ref, gen, self.decoder_weights)
                .squeeze()
                .float()
                .flatten()
            )

            n = min(dev.numel(), cpu.numel())
            noise = (cpu[:n] - dev[:n]).pow(2).mean().item()
            sig = cpu[:n].pow(2).mean().item()
            snr = (
                10.0 * math.log10(sig / noise)
                if noise > 0 and sig > 0
                else float("nan")
            )
            self.logger.info(
                f"[decoder-check:{label}] snr={snr:.2f}dB "
                f"dev_rms={dev.pow(2).mean().sqrt().item():.5f} cpu_rms={cpu.pow(2).mean().sqrt().item():.5f} "
                f"frames={gen.shape[0]}"
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                f"[decoder-check:{label}] failed: {type(e).__name__}: {e}"
            )

    def _compare_decodes(self, api, ref_codes, codes, device_audio) -> None:
        """Diagnostic: device vs CPU decode of the SAME codes, with an offset search.

        The decoder measures 18-19 dB SNR against the CPU reference standalone, yet
        produces WER 95% in serving. This decides between two very different causes:

          * best lag != 0 with good SNR there -> the waveform is correct but
            mis-sliced (an offset/length bug), which is a trivial fix.
          * low SNR at every lag -> the samples themselves are wrong, i.e. sharing
            the device with the talker corrupts the decode.

        Enable with TT_QWEN3_DECODE_COMPARE=1.
        """
        import math

        try:
            cpu_audio = api.decode_icl_audio(ref_codes, codes, self.decoder_weights)
            dev = device_audio.squeeze().detach().cpu().float().flatten()
            ref = cpu_audio.squeeze().detach().cpu().float().flatten()

            def snr_at(lag: int) -> float:
                # positive lag: device is late relative to cpu
                d = dev[lag:] if lag >= 0 else dev[: dev.numel() + lag]
                r = ref[: d.numel()] if lag >= 0 else ref[-lag:]
                n = min(d.numel(), r.numel())
                if n < 1920:
                    return float("-inf")
                d, r = d[:n], r[:n]
                noise = (r - d).pow(2).mean().item()
                sig = r.pow(2).mean().item()
                if noise <= 0:
                    return float("inf")
                return 10.0 * math.log10(sig / noise) if sig > 0 else float("-inf")

            spf = 1920
            lags = [f * spf for f in range(-4, 5)]  # +/- 4 codec frames
            scored = sorted(((snr_at(lag), lag) for lag in lags), reverse=True)
            best_snr, best_lag = scored[0]

            self.logger.info(
                f"[decode-compare] dev_len={dev.numel()} cpu_len={ref.numel()} "
                f"dev_rms={dev.pow(2).mean().sqrt().item():.5f} "
                f"cpu_rms={ref.pow(2).mean().sqrt().item():.5f} "
                f"snr@0={snr_at(0):.2f}dB best_snr={best_snr:.2f}dB "
                f"best_lag={best_lag} samples ({best_lag / spf:.0f} frames)"
            )

            # Control: a fixed real-speech input through the same path. Bad here too
            # means the decoder itself is broken on this device, not these codes.
            self._check_decoder_against_cpu("control")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"[decode-compare] failed: {type(e).__name__}: {e}")

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

        # Decode the reference and generated codes together, then cut the
        # reference's portion, as HF Qwen3-TTS does. Decoding the generated
        # codes on their own leaves artifacts at the start, which the old
        # trim_codec_frames=4 masked by dropping 4 codec frames of real
        # speech -- that is what truncated the first word(s) of the output.
        # tt-metal #57964 adds decode_icl_audio and defaults trim_codec_frames
        # to 0 (deprecated), so the old trim branch is removed here.
        #
        # On-device decode (default): decode_audio_device runs the same cat+cut on
        # the TT device (length-flat ~0.3-0.45s warm) instead of the CPU path that
        # scaled with frame count and inflated TTFT. Both return generated speech
        # only. CPU path kept as a fallback via TT_QWEN3_DEVICE_DECODE=0.
        ref_state = self._ref_state_for(api, voice_id, ref_codes)
        if self._use_device_decode and self.device_decoder is not None:
            audio = self._device_decode(api, ref_codes, codes, ref_state)
            if os.environ.get("TT_QWEN3_DECODE_COMPARE", "0") == "1":
                self._compare_decodes(api, ref_codes, codes, audio)
        else:
            audio = api.decode_icl_audio(
                ref_codes, codes, self.decoder_weights, ref_state=ref_state
            )
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
