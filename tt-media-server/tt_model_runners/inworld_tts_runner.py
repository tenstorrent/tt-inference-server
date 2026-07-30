# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import os
import threading
import uuid
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import ttnn
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import (
    TextToSpeechChunkOutput,
    TextToSpeechChunkResult,
    TextToSpeechResponse,
)
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

# Matches tt_model_runners/vllm_runner.py's streaming-chunk type strings --
# device_worker.py/base_service.py dispatch on these literal values regardless
# of model domain (LLM text vs. TTS audio).
CHUNK_TYPE = "streaming_chunk"
FINAL_TYPE = "final_result"

# Streaming audio-decoder chunk size in codec tokens (~0.64s of audio at
# CODEC_TOKENS_PER_SEC=50). User-confirmed: matches the main tt-metal repo's
# CLI scripts (main_tp8.py/main_tp4.py/main_tp1.py), which default to T=32
# for lower per-chunk/TTFC latency, accepting the same known tradeoff T=64
# was originally chosen to avoid -- T=32 < the decoder backbone's
# sliding_window=40, so independently-decoded chunks lose real cross-chunk
# context at every boundary (small audible seam). See
# tt_modeling.create_streaming_decoder's docstring for the full tradeoff
# writeup and the session that confirmed it.
STREAMING_CHUNK_SIZE = 32


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
        # trace_region_size: large enough to fit all three coexisting traces
        # (SpeechLM decode + the T=32 streaming decoder + the main
        # DECODER_MAX_T=1024 decoder), but NOT larger than needed --
        # confirmed on real hardware (an earlier session) that an oversized
        # trace region measurably degrades PREFILL and DECODE throughput even
        # though nothing else changed: 200MB gave TTFT=110ms/decode=91ms per
        # token (vs the ~65ms/~40ms baseline), while 100MB fully recovered
        # baseline perf and fit all three traces without OOM -- AT THE OLD
        # DECODER_MAX_T=480/STREAMING_CHUNK_SIZE=64 sizes. Re-verify this
        # value on real hardware after the DECODER_MAX_T=1024 bump (roughly
        # 2x the old main-decoder trace) -- 100MB may no longer be sufficient
        # or optimal.
        return {"l1_small_size": 16384, "trace_region_size": 100_000_000}

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

            # Host sampling (above) does real per-token CPU work (top-p
            # filtering, repetition penalty -- both torch ops over the
            # ~16k-vocab logits tensor). This device-worker subprocess is
            # forked from the main FastAPI process, which by the time it
            # forks has already called torch.set_num_threads(1) via
            # model_services/cpu_workload_handler.py's
            # setup_cpu_threading_limits("2") (deliberately conservative --
            # it sizes ITS OWN worker pool for the full DP=32 fleet sharing
            # one host). fork() means this subprocess inherits that
            # process-wide torch thread-count setting even though it has
            # nothing to do with CpuWorkloadHandler -- confirmed on real
            # hardware via py-spy (~92% of decode wall-time was in
            # single-threaded sample_top_p/_apply_repetition_penalty, not
            # device compute) and via a direct fix/re-measure (decode
            # dropped from ~91ms/token back to the ~40ms/token baseline once
            # threads were restored). Re-derive and set an explicit,
            # worker-count-aware thread count here so it's correct at any DP
            # scale: plenty of headroom for a single-worker deployment,
            # matching CpuWorkloadHandler's own "2 threads per worker"
            # convention at DP=32 to avoid reintroducing oversubscription.
            # torch.set_num_threads() (intra-op) is safe to call repeatedly,
            # unlike set_num_interop_threads() (would raise if already
            # locked by the parent's pre-fork call) -- the sampling ops here
            # only use intra-op parallelism, so that's all that's needed.
            num_dp_workers = max(1, len(self.settings.device_ids.replace(" ", "").split("),(")))
            num_torch_threads = max(1, (os.cpu_count() or 1) // num_dp_workers)
            if torch.get_num_threads() != num_torch_threads:
                torch.set_num_threads(num_torch_threads)
            self.logger.info(
                f"Device {self.device_id}: set torch.num_threads={num_torch_threads} "
                f"({os.cpu_count()} CPUs / {num_dp_workers} DP workers)"
            )

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

                # 3. Streaming (T=STREAMING_CHUNK_SIZE) decoder trace, created
                #    FIRST -- before the main DECODER_MAX_T decoder below. Verified
                #    (an earlier session's
                #    scratchpad/diag_two_decoders.py): capturing a SMALLER-T
                #    TtDecoder trace AFTER a LARGER-T one already exists on the
                #    same device silently corrupts the second-captured decoder's
                #    fsq_dequantize output with actual inf values. This ordering
                #    requirement is independent of (and in addition to) the
                #    decoder-before-SpeechLM-warmup ordering documented below.
                self.logger.info(
                    f"Device {self.device_id}: Creating streaming (T={STREAMING_CHUNK_SIZE}) "
                    "decoder FIRST..."
                )
                streaming_decoder = tt_modeling.create_streaming_decoder(
                    decoder_path, self.ttnn_device, chunk_size=STREAMING_CHUNK_SIZE
                )

                # 4. Decoder: constructed BEFORE SpeechLM's warmup (below) and
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
                self._audio_decoder.attach_streaming(
                    streaming_decoder, chunk_size=STREAMING_CHUNK_SIZE
                )

                # 5. Warmup SpeechLM (untimed, forces prefill + decode trace capture),
                #    then a real end-to-end (discarded) warmup call through the full
                #    pipeline -- SpeechLM's own trace and the audio decoder's trace
                #    have each been warmed in isolation by this point, but never yet
                #    run back-to-back in the same request. Confirmed on real hardware
                #    that skipping this leaks a one-time compile cost into the first
                #    real request whenever a new trace-shape combination (e.g.
                #    DECODER_MAX_T/STREAMING_CHUNK_SIZE) is exercised for the first
                #    time on a given machine -- see warmup_full_pipeline's docstring.
                tt_modeling.warmup_speechlm(self._speechlm, self._tokenizer)
                tt_modeling.warmup_full_pipeline(
                    self._speechlm, self._tokenizer, self._audio_decoder
                )

                # 6. Encoder (for voice cloning via POST /v1/audio/voices).
                #
                # use_trace=False (eager path -- see CachingAudioEncoder's own
                # docstring, this fallback already existed for exactly this
                # situation): a captured trace's L1 buffers stay permanently
                # reserved for the trace's whole lifetime, and this device now
                # also holds SpeechLM's decode trace plus TWO TtDecoder traces
                # (streaming T=STREAMING_CHUNK_SIZE + main DECODER_MAX_T, added
                # in an earlier session for Stage 1 streaming support --
                # DECODER_MAX_T since bumped 480->1024, STREAMING_CHUNK_SIZE
                # since changed 64->32, net L1 impact not re-verified against
                # this specific encoder-eager-mode constraint). Verified on real hardware
                # (Stage 1h) that adding the streaming decoder's persistent L1
                # footprint on top of the pre-existing two traces pushes the
                # encoder's own traced conv/convtranspose capture over the L1
                # budget (TT_THROW: static CBs clash with an L1 buffer on core
                # range [0-0 - 11-7]) -- confirmed via a controlled A/B
                # (warmup succeeds without the streaming decoder, fails with
                # it). Voice-clone registration (POST /v1/audio/voices) is a
                # rare, non-realtime operation (once per registered voice, not
                # per-TTS-request), so the plan was to trade its trace-capture
                # speed for a slower-but-correct eager path.
                #
                # ** KNOWN REGRESSION, NOT YET FIXED **: the eager path itself
                # then hit a SEPARATE, previously-unexercised device bug on
                # real reference audio (TT_FATAL in ttnn.layer_norm: "Sharded
                # layernorm does not support non-rectangular core grids" --
                # the eager path's core-grid selection, likely _pick_grid/
                # _block_shard in tt/mlp.py, picks a bad grid for this input
                # shape). Voice-cloning (POST /v1/audio/voices) is therefore
                # currently BROKEN on this runner -- confirmed via a live
                # request against Ashley_en.wav (Stage 1h, 2026-07-30). Kept
                # as use_trace=False anyway per explicit user decision: ship
                # streaming now, accept broken voice-cloning as a follow-up
                # (see BRINGUP_LOG.md Session 13). Do not assume this path
                # works without re-verifying.
                self.logger.info(
                    f"Device {self.device_id}: Creating TTNN encoder from {encoder_path} "
                    "(eager, use_trace=False -- see comment above)..."
                )
                self._audio_encoder = tt_modeling.CachingAudioEncoder(
                    encoder_path, device=self.ttnn_device, use_trace=False
                )

                # 7. Voice-clone VQ-code registration cache.
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
            try:
                self._audio_decoder.release_streaming()
            except Exception as e:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to release streaming decoder trace: {e}"
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

    @staticmethod
    def _wav_base64(wav: torch.Tensor, sample_rate: int) -> tuple[str, float]:
        """Encode a mono float waveform tensor as a base64 WAV, matching
        speecht5_runner.py's approach. Returns (base64_str, duration_seconds).
        """
        audio_buffer = io.BytesIO()
        samples = wav.squeeze().detach().cpu().numpy()
        sf.write(audio_buffer, samples, sample_rate, format="WAV")
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
        duration = samples.shape[-1] / sample_rate
        return audio_base64, duration

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

        audio_base64, duration = self._wav_base64(wav, DECODER_SAMPLE_RATE)

        return TextToSpeechResponse(
            audio=audio_base64,
            duration=duration,
            sample_rate=DECODER_SAMPLE_RATE,
            format="wav",
            speaker_id=request.voice_id,
        )

    def _log_streaming_perf_summary(self, profiler, chunk_index, done_state):
        """Streaming counterpart of ``_log_perf_summary``: same TTFT/decode
        ms-per-token breakdown, plus per-chunk audio-decode latency (each
        chunk's ``decode_chunk`` call is timed separately -- see
        ``synthesize_tp8_streaming``'s ``audio_decode_chunk`` profiler steps
        in tt_modeling.py) instead of one aggregate audio-decode number for
        the whole utterance.
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

        decode_calls = (done_state or {}).get("decode_calls") or 0
        num_steady = max(0, decode_calls - 1)
        total_decode_ms = 0.0
        for i in range(1, decode_calls):
            d = _dur_ms(f"inference_decode_time_{i}")
            if d == d:  # not NaN
                total_decode_ms += d
        if num_steady > 0 and total_decode_ms > 0:
            ms_per_token = total_decode_ms / num_steady
            tok_s = num_steady / (total_decode_ms / 1000.0)
        else:
            ms_per_token = float("nan")
            tok_s = float("nan")

        chunk_ms = []
        for i in range(chunk_index):
            if profiler.contains_step("audio_decode_chunk", iteration=i):
                chunk_ms.append(profiler.get_duration("audio_decode_chunk", iteration=i) * 1000.0)
        avg_chunk_ms = sum(chunk_ms) / len(chunk_ms) if chunk_ms else float("nan")
        chunk_ms_str = ", ".join(f"{m:.1f}" for m in chunk_ms)

        self.logger.info(
            f"Device {self.device_id}: [tts-perf-streaming] reset_state={reset_ms:.1f}ms "
            f"TTFT/prefill={ttft_ms:.1f}ms "
            f"decode={total_decode_ms:.1f}ms over {num_steady} steady-state tokens "
            f"({ms_per_token:.2f} ms/token, {tok_s:.2f} tok/s) "
            f"| {chunk_index} audio chunks (64 tokens each, last may be shorter), "
            f"avg_chunk_audio_decode={avg_chunk_ms:.1f}ms, per_chunk_ms=[{chunk_ms_str}] "
            f"| converged={(done_state or {}).get('converged')}, "
            f"real_prompt_len={(done_state or {}).get('prompt_len')}"
        )

    async def _generate_streaming_tts(self, request: TextToSpeechRequest):
        """Streaming counterpart of ``_synthesize``: yields one
        ``TextToSpeechChunkOutput`` per progressively-decoded audio chunk,
        then a terminal empty-audio ``FINAL_TYPE`` marker -- mirrors
        ``vllm_runner.py``'s ``_generate_streaming`` shape (device_worker.py's
        streaming dispatch and base_service.py's process_streaming key off
        the same ``type``/``data`` envelope regardless of model domain).

        Runs synchronously (blocking device calls, no thread hop): the
        streaming dispatch in device_workers/device_worker.py already drains
        this generator via a dedicated ``loop.run_until_complete(...)`` call
        that handles one streaming request at a time, so there is no
        concurrent event-loop work to protect during the (blocking) device
        calls -- matching ``_synthesize``'s single blocking call, just
        incremental instead of all-at-once.
        """
        speech_ids_prompt = None
        if request.voice_id:
            speech_ids_prompt = self._voice_cache.get_voice(request.voice_id)

        profiler = BenchmarkProfiler()
        chunk_index = 0
        total_samples = 0
        done_state = None

        for kind, payload in tt_modeling.synthesize_tp8_streaming(
            self._speechlm,
            self._tokenizer,
            self._audio_decoder,
            request.text,
            speech_ids_prompt=speech_ids_prompt,
            profiler=profiler,
        ):
            if kind == "chunk":
                audio_base64, chunk_duration = self._wav_base64(payload, DECODER_SAMPLE_RATE)
                total_samples += payload.reshape(-1).shape[0]
                yield TextToSpeechChunkOutput(
                    type=CHUNK_TYPE,
                    data=TextToSpeechChunkResult(
                        audio_base64=audio_base64,
                        chunk_index=chunk_index,
                        is_final=False,
                        sample_rate=DECODER_SAMPLE_RATE,
                        format="wav",
                        duration=chunk_duration,
                        speaker_id=request.voice_id,
                    ),
                )
                chunk_index += 1
            else:
                done_state = payload

        total_duration = total_samples / DECODER_SAMPLE_RATE
        self._log_streaming_perf_summary(profiler, chunk_index, done_state)

        yield TextToSpeechChunkOutput(
            type=FINAL_TYPE,
            data=TextToSpeechChunkResult(
                audio_base64="",
                chunk_index=chunk_index,
                is_final=True,
                sample_rate=DECODER_SAMPLE_RATE,
                format="wav",
                duration=total_duration,
                speaker_id=request.voice_id,
            ),
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
                if getattr(request, "stream", False):
                    return self._generate_streaming_tts(request)
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
