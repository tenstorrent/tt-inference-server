# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import os
import uuid
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import ttnn
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from domain.voice_encode_request import VoiceEncodeRequest
from domain.voice_encode_response import VoiceEncodeResponse
from domain.voice_list_request import VoiceListRequest
from domain.voice_list_response import VoiceInfo, VoiceListResponse
from models.demos.inworld_tts import tt_modeling
from models.demos.inworld_tts.tt.decoder_tts2 import TtDecoder
from models.demos.inworld_tts.tt.speechlm_ttnn import TtSpeechLmConfig, TtTransformersSpeechLM
from models.perf.benchmarking_utils import BenchmarkProfiler
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.logger import log_exception_chain
from utils.voice_clone_cache import VoiceCloneCacheManager

# Matches models/demos/inworld_tts/main_tp8.py's own constants (see its docstring).
CODEC_SAMPLE_RATE = 16000
DECODER_SAMPLE_RATE = 48000
CODEC_TOKENS_PER_SEC = 50


def _load_wav_from_bytes(data: bytes, target_sample_rate: int) -> torch.Tensor:
    """In-memory-buffer counterpart of ``tt_modeling.load_wav`` (which only
    accepts a file path): decodes ``data`` as an audio file, downmixes to
    mono, and resamples to ``target_sample_rate`` if needed.

    Returns:
        [1, num_samples] float tensor.
    """
    wav, sr = torchaudio.load(io.BytesIO(data))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sample_rate and sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    return wav


class TTInworldTTSRunner(BaseMetalDeviceRunner):
    """Inworld TTS-2 runner: SpeechLM (VQ-code generation) + traced audio decoder
    + traced audio encoder (voice cloning) sharing one mesh device.

    Mesh-shape-agnostic: works both as a single tensor-parallel (1, 8) mesh
    (DeviceTypes.P150X8 -- mirrors ``models/demos/inworld_tts/main_tp8.py``) and
    as one of N independent single-chip (1, 1) data-parallel workers
    (DeviceTypes.BLACKHOLE_GALAXY, 32-way DP -- mirrors
    ``models/demos/inworld_tts/main.py``). The SpeechLM/decoder/encoder TTNN
    ops shard/all-reduce across whatever chips the mesh actually has and degrade
    to a no-op on a 1-chip mesh (see speechlm_ttnn.py ``is_multi_device`` guards).
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self._speechlm = None
        self._tokenizer = None
        self._audio_decoder = None
        self._audio_encoder = None
        self._voice_cache = None

    def get_pipeline_device_params(self):
        return {"l1_small_size": 16384, "trace_region_size": 200_000_000}

    def _configure_fabric(self, updated_device_params):
        try:
            # Single-chip (1,1) DP workers have no inter-chip CCL, so the 1D
            # fabric is unnecessary (and only makes sense for the multi-chip TP
            # mesh). Mirror main.py's proven single-chip path, which never sets
            # a fabric config. Only the multi-chip (TP) mesh gets FABRIC_1D.
            if tuple(self.settings.device_mesh_shape) == (1, 1):
                updated_device_params.pop("fabric_config", None)
                self.logger.info(
                    f"Device {self.device_id}: single-chip (1,1) mesh -- "
                    "skipping FABRIC_1D (no inter-chip CCL needed)."
                )
                return None
            fabric_config = updated_device_params.pop(
                "fabric_config", ttnn.FabricConfig.FABRIC_1D
            )
            fabric_tensix_config = updated_device_params.pop(
                "fabric_tensix_config", ttnn.FabricTensixConfig.DISABLED
            )
            reliability_mode = updated_device_params.pop(
                "reliability_mode", ttnn.FabricReliabilityMode.STRICT_INIT
            )
            fabric_router_config = updated_device_params.pop(
                "fabric_router_config", ttnn.FabricRouterConfig()
            )
            ttnn.set_fabric_config(
                fabric_config,
                reliability_mode,
                None,
                fabric_tensix_config,
                ttnn.FabricUDMMode.DISABLED,
                ttnn.FabricManagerMode.DEFAULT,
                fabric_router_config,
            )
            return fabric_config
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Fabric configuration failed",
                e,
            )
            raise RuntimeError(f"Fabric configuration failed: {str(e)}") from e

    def load_weights(self):
        """Validate the (local, non-HF) checkpoint paths this runner needs.

        SpeechLM/decoder/encoder checkpoints are read from local disk paths
        given via INWORLD_TTS_{SPEECHLM,DECODER,ENCODER}_PATH -- there is no
        HF repo to download, so this only checks the paths exist.
        """
        required_env_vars = (
            "INWORLD_TTS_SPEECHLM_PATH",
            "INWORLD_TTS_DECODER_PATH",
            "INWORLD_TTS_ENCODER_PATH",
        )
        all_present = True
        for env_var in required_env_vars:
            path = os.environ.get(env_var)
            if not path or not Path(path).exists():
                self.logger.error(
                    f"Device {self.device_id}: {env_var}={path!r} is not set or does not exist."
                )
                all_present = False
        return all_present

    @log_execution_time(
        "Inworld TTS warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        try:
            if self.ttnn_device is None:
                raise ValueError("Device not initialized. Call set_device() first.")

            # Force host-side (torch) sampling for this long-lived, multi-request
            # server. The on-device stochastic sampler's hardware PRNG free-runs
            # across generations and CANNOT be reset deterministically between
            # requests via re-seeding (verified: identical logits + identical
            # pushed seed -> different sampled tokens on the 2nd+ request), so
            # device sampling only produces correct output for the FIRST
            # generation in a process. host sampling re-seeds torch per request
            # (see ``_synthesize_with_perf``) and is fully deterministic and
            # reproducible, so every HTTP request behaves like a fresh
            # single-shot run. main_tp8.py (one generation per process) is
            # unaffected and keeps the faster on-device sampling path.
            os.environ.setdefault("INWORLD_TTS_FORCE_HOST_SAMPLING", "1")

            speechlm_path = os.environ.get("INWORLD_TTS_SPEECHLM_PATH")
            decoder_path = os.environ.get("INWORLD_TTS_DECODER_PATH")
            encoder_path = os.environ.get("INWORLD_TTS_ENCODER_PATH")

            def _build_pipeline():
                # 1. SpeechLM (traced, paged attention). Tensor-parallel across
                #    the mesh when >1 chip, plain single-chip when the mesh is
                #    (1, 1) (DP fleet worker).
                num_chips = self.ttnn_device.get_num_devices()
                self.logger.info(
                    f"Device {self.device_id}: Loading TTNN SpeechLM "
                    f"(mesh={tuple(self.settings.device_mesh_shape)}, "
                    f"num_chips={num_chips}) from {speechlm_path}..."
                )
                self._speechlm = TtTransformersSpeechLM(
                    mesh_device=self.ttnn_device,
                    config=TtSpeechLmConfig(
                        checkpoint_path=speechlm_path,
                        enable_trace=True,
                        use_paged_attention=True,
                    ),
                )

                # 2. Tokenizer.
                self.logger.info(f"Device {self.device_id}: Loading tokenizer...")
                self._tokenizer = tt_modeling.get_tokenizer(speechlm_path)

                # 3. Decoder: constructed BEFORE SpeechLM's warmup (below) and
                #    BEFORE the encoder. This ordering -- decoder, then SpeechLM
                #    warmup, then encoder -- mirrors main_tp8.py's own construction
                #    order, which was determined empirically (see its comments) to
                #    be the one combination that avoids two reproducible failures
                #    on this shared (1, 8) mesh: SpeechLM's on-device-sampling
                #    decode trace corrupting sampled tokens if captured after the
                #    decoder/encoder had already claimed persistent trace buffers,
                #    and the decoder's own execute_trace() hanging if SpeechLM's
                #    trace was captured before the decoder was constructed.
                self.logger.info(
                    f"Device {self.device_id}: Creating TTNN decoder from {decoder_path}..."
                )
                tt_decoder = TtDecoder(
                    device=self.ttnn_device,
                    state_dict_path=decoder_path,
                    use_torch_istft=False,
                )
                self._audio_decoder = tt_modeling.TtAudioDecoder(
                    tt_decoder,
                    sample_rate=DECODER_SAMPLE_RATE,
                    token_rate=CODEC_TOKENS_PER_SEC,
                    use_trace=True,
                )

                # 4. Warmup SpeechLM (untimed, forces prefill + decode trace capture).
                tt_modeling.warmup_speechlm(self._speechlm, self._tokenizer)

                # 5. Encoder (for voice cloning via POST /v1/audio/voices).
                self.logger.info(
                    f"Device {self.device_id}: Creating TTNN encoder from {encoder_path}..."
                )
                self._audio_encoder = tt_modeling.CachingAudioEncoder(
                    encoder_path, device=self.ttnn_device, use_trace=True
                )

                # 6. Voice-clone VQ-code registration cache.
                self._voice_cache = VoiceCloneCacheManager()

            await asyncio.to_thread(_build_pipeline)
            self.logger.info(f"Device {self.device_id}: Inworld TTS warmup complete.")
            return True
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inworld TTS warmup failed: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Inworld TTS warmup failed: {str(e)}"
            ) from e

    def close_device(self):
        # Release the decoder's persistent trace buffers BEFORE closing the mesh
        # device -- releasing traces after closing the mesh device crashes (see
        # main_tp8.py's teardown ordering comment).
        if self._audio_decoder is not None:
            try:
                self._audio_decoder._decoder.release_trace()
            except Exception as e:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to release decoder trace: {e}"
                )
        return super().close_device()

    def _encode_voice(self, request: VoiceEncodeRequest) -> VoiceEncodeResponse:
        reference_audio = request.reference_audio
        if isinstance(reference_audio, str):
            audio_bytes = base64.b64decode(reference_audio)
        else:
            audio_bytes = reference_audio

        wav = _load_wav_from_bytes(audio_bytes, target_sample_rate=CODEC_SAMPLE_RATE)

        voice_id = request.voice_id or uuid.uuid4().hex
        speech_ids = self._audio_encoder.encode(voice_id, wav)
        self._voice_cache.register_voice(
            voice_id,
            speech_ids,
            language=request.language,
            description=request.description,
        )

        return VoiceEncodeResponse(
            voice_id=voice_id,
            num_codes=len(speech_ids),
            language=request.language,
            description=request.description,
        )

    def _list_voices(self, request: VoiceListRequest) -> VoiceListResponse:
        # DP-fleet coherence: refresh from the shared on-disk pickle so this
        # worker reports voices registered by any other worker (see
        # VoiceCloneCacheManager.reload_from_disk / get_voice for the rationale).
        self._voice_cache.reload_from_disk()
        voices = [
            VoiceInfo(**info)
            for info in self._voice_cache.list_voices_with_metadata()
        ]
        return VoiceListResponse(voices=voices)

    def _log_perf_summary(self, profiler, speech_ids, prompt_len, decode_calls, decode_elapsed_ms):
        """Log a one-line per-request timing breakdown from the profiler that
        ``synthesize_tp8`` populated, mirroring ``main_tp8.py``'s CLI perf
        summary (lines ~261-306). Best-effort: any missing key is reported as
        NaN rather than failing the request.
        """

        def _dur_ms(key):
            try:
                if profiler.contains_step(key):
                    return profiler.get_duration(key) * 1000.0
            except Exception:
                pass
            return float("nan")

        reset_ms = _dur_ms("reset_state")
        ttft_ms = _dur_ms("inference_prefill")

        # Steady-state decode: calls 1..decode_calls-1 (call 0 is the
        # trace-compile call, excluded), matching main_tp8.py.
        num_steady = max(0, (decode_calls or 0) - 1)
        total_decode_ms = 0.0
        for i in range(1, decode_calls or 0):
            d = _dur_ms(f"inference_decode_time_{i}")
            if d == d:  # not NaN
                total_decode_ms += d
        if num_steady > 0 and total_decode_ms > 0:
            ms_per_token = total_decode_ms / num_steady
            tok_s = num_steady / (total_decode_ms / 1000.0)
        else:
            ms_per_token = float("nan")
            tok_s = float("nan")

        num_codes = len(speech_ids) if speech_ids else 0
        self.logger.info(
            f"Device {self.device_id}: [tts-perf] reset_state={reset_ms:.1f}ms "
            f"TTFT/prefill={ttft_ms:.1f}ms "
            f"decode={total_decode_ms:.1f}ms over {num_steady} steady-state tokens "
            f"({ms_per_token:.2f} ms/token, {tok_s:.2f} tok/s) "
            f"audio_decode={decode_elapsed_ms:.1f}ms "
            f"| {num_codes} VQ codes, real_prompt_len={prompt_len}"
        )

    def _synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        speech_ids_prompt = None
        if request.voice_id:
            speech_ids_prompt = self._voice_cache.get_voice(request.voice_id)

        # Per-request perf instrumentation. ``synthesize_tp8`` already computes
        # a full timing breakdown internally (via ``BenchmarkProfiler`` in
        # ``_synthesize_with_perf`` plus the ``decode_elapsed_ms`` audio-decoder
        # timing) but the server previously discarded all of it, leaving
        # everything inside the outer ``[run]`` wrapper a black box. Pass an
        # explicit profiler in and log a one-line summary mirroring
        # ``main_tp8.py``'s CLI perf print (reset_state / TTFT / steady-state
        # decode ms/token + tok/s / audio-decode ms) so per-stage cost is
        # visible in the server log for every request. Purely additive -- no
        # change to synthesis behavior.
        profiler = BenchmarkProfiler()

        wav, _speech_ids, _prompt_len, _decode_calls, _decode_elapsed_ms = tt_modeling.synthesize_tp8(
            self._speechlm,
            self._tokenizer,
            self._audio_decoder,
            request.text,
            speech_ids_prompt=speech_ids_prompt,
            profiler=profiler,
        )

        self._log_perf_summary(profiler, _speech_ids, _prompt_len, _decode_calls, _decode_elapsed_ms)

        # Base64/WAV-encoding matches speecht5_runner.py's own approach exactly.
        audio_buffer = io.BytesIO()
        sf.write(
            audio_buffer,
            wav.squeeze().detach().cpu().numpy(),
            DECODER_SAMPLE_RATE,
            format="WAV",
        )
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
        duration = wav.shape[-1] / DECODER_SAMPLE_RATE

        return TextToSpeechResponse(
            audio=audio_base64,
            duration=duration,
            sample_rate=DECODER_SAMPLE_RATE,
            format="wav",
            speaker_id=request.voice_id,
        )

    async def _run_async(self, requests: list):
        if self._speechlm is None or self._audio_decoder is None:
            raise RuntimeError("Model components not loaded. Call warmup() first.")
        if self.ttnn_device is None:
            raise ValueError("TTNN device not initialized")

        if len(requests) > 1:
            self.logger.warning(
                f"Device {self.device_id}: Batch processing not implemented. "
                f"Processing only first of {len(requests)} requests"
            )

        request = requests[0]
        if request is None:
            raise ValueError("Request cannot be None")

        try:
            if isinstance(request, VoiceEncodeRequest):
                return await asyncio.to_thread(self._encode_voice, request)
            if isinstance(request, VoiceListRequest):
                return await asyncio.to_thread(self._list_voices, request)
            if isinstance(request, TextToSpeechRequest):
                return await asyncio.to_thread(self._synthesize, request)
            raise ValueError(
                f"Unsupported request type for Inworld TTS runner: {type(request).__name__}"
            )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    @log_execution_time(
        "Run Inworld TTS inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list):
        """Synchronous wrapper for async inference"""
        result = asyncio.run(self._run_async(requests))
        return [result] if result is not None else []
