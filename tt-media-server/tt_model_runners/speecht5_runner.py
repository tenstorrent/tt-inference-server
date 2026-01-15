# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import sys
from config.constants import SupportedModels
from config.settings import settings
import torch
import os
import io
import base64
from typing import Dict, Any, AsyncGenerator
import soundfile as sf

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Import TTNN SpeechT5 components
tt_metal_path = os.environ.get("TT_METAL_HOME")
if tt_metal_path:
    sys.path.append(tt_metal_path)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNSpeechT5Encoder,
    TTNNEncoderConfig,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNSpeechT5Decoder,
    TTNNDecoderConfig,
    preprocess_decoder_parameters,
    init_kv_cache,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNSpeechT5SpeechDecoderPostnet,
    TTNNPostNetConfig,
    preprocess_postnet_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import (
    SpeechT5Generator,
)

from device_workers.worker_utils import setup_cpu_threading_limits
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import (
    TextToSpeechResponse,
    PartialStreamingAudioResponse,
)
from telemetry.telemetry_client import TelemetryEvent
from utils.speaker_embeddings import SpeakerEmbeddingsManager
import ttnn
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time


class SpeechT5Constants:
    MAX_STEPS = 950  # Maximum generation steps for long text
    SAMPLE_RATE = 16000
    REDUCTION_FACTOR = 2
    NUM_MEL_BINS = 80
    STREAMING_CHUNK_SIZE = 20  # Generate audio chunks every 20 mel frames
    MAX_CLEANUP_RETRIES = 3
    RETRY_DELAY_SECONDS = 1
    MIN_STEPS_FOR_STOP_CHECK = 10  # Don't stop too early


class SpeechT5ModelError(Exception):
    """Base exception for SpeechT5 model errors"""

    pass


class ModelNotLoadedError(SpeechT5ModelError):
    """Raised when attempting inference without loaded model"""

    pass


class TextProcessingError(SpeechT5ModelError):
    """Raised when text processing fails"""

    pass


class AudioGenerationError(SpeechT5ModelError):
    """Raised when audio generation fails"""

    pass


class DeviceInitializationError(SpeechT5ModelError):
    """Raised when device initialization fails"""

    pass


class InferenceError(SpeechT5ModelError):
    """Error occurred during model inference"""

    pass


class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds timeout limit"""

    pass


class DeviceCleanupError(SpeechT5ModelError):
    """Error occurred during device cleanup"""

    pass


class TTSpeechT5Runner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.ttnn_device = None
        self.processor = None
        self.vocoder = None
        self.ttnn_encoder = None
        self.ttnn_decoder = None
        self.ttnn_postnet = None
        self.generator = None  # For trace execution
        self.speaker_manager = None
        self.decoder_config = None  # Store for KV cache initialization

        # Limit threading for stability during inference
        setup_cpu_threading_limits("1")

        # Explicitly disable fabric for non-galaxy devices
        if not settings.is_galaxy:
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    def get_pipeline_device_params(self):
        """Device parameters optimized for SpeechT5 with 2CQ support."""
        device_params = {
            "l1_small_size": 300000,  # Increased for KV cache
            "trace_region_size": 10000000,
            "num_command_queues": 2,  # Enable 2CQ for async overlap
        }
        return device_params

    def _initialize_models(self):
        """Initialize SpeechT5 models and components"""
        try:
            # Load HuggingFace models
            model_name = SupportedModels.SPEECHT5_TTS.value
            self.logger.info(
                f"Device {self.device_id}: Loading SpeechT5 models from {model_name}"
            )

            self.processor = SpeechT5Processor.from_pretrained(model_name)
            model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Initialize speaker embeddings manager
            self.speaker_manager = SpeakerEmbeddingsManager()
            self.speaker_manager.preload_embeddings()

            # Create model configurations
            encoder_config = TTNNEncoderConfig(
                vocab_size=model.config.vocab_size,
                hidden_size=model.config.hidden_size,
                num_layers=model.config.encoder_layers,
                num_heads=model.config.encoder_attention_heads,
                ffn_dim=model.config.encoder_ffn_dim,
                max_position_embeddings=model.config.max_length,
                layer_norm_eps=model.config.layer_norm_eps,
            )

            self.decoder_config = TTNNDecoderConfig(
                hidden_size=model.config.hidden_size,
                num_layers=model.config.decoder_layers,
                num_heads=model.config.decoder_attention_heads,
                ffn_dim=model.config.decoder_ffn_dim,
                max_position_embeddings=model.config.max_length,
                layer_norm_eps=model.config.layer_norm_eps,
                num_mel_bins=model.config.num_mel_bins,
                reduction_factor=model.config.reduction_factor,
                speech_decoder_prenet_units=model.config.speech_decoder_prenet_units,
                speech_decoder_prenet_layers=model.config.speech_decoder_prenet_layers,
                speech_decoder_prenet_dropout=model.config.speech_decoder_prenet_dropout,
                speaker_embedding_dim=model.config.speaker_embedding_dim,
            )

            postnet_config = TTNNPostNetConfig(
                hidden_size=model.config.hidden_size,
                num_mel_bins=model.config.num_mel_bins,
                reduction_factor=model.config.reduction_factor,
                postnet_layers=model.config.speech_decoder_postnet_layers,
                postnet_units=model.config.speech_decoder_postnet_units,
                postnet_kernel=model.config.speech_decoder_postnet_kernel,
            )

            # Get a default speaker embedding for model initialization
            available_speakers = self.speaker_manager.list_available_speakers()
            if not available_speakers:
                # Create a zero embedding if no speakers are available
                self.logger.warning(
                    "No speaker embeddings available, using zero embedding for initialization"
                )
                default_speaker_embedding = torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0)
            else:
                default_speaker_embedding = self.speaker_manager.get_speaker_embedding(
                    available_speakers[0]
                )

            # Create TTNN models
            self.logger.info(f"Device {self.device_id}: Creating TTNN encoder")
            self.ttnn_encoder = TTNNSpeechT5Encoder(
                self.ttnn_device,
                preprocess_encoder_parameters(
                    model.speecht5.encoder, encoder_config, self.ttnn_device
                ),
                encoder_config,
            )

            self.logger.info(f"Device {self.device_id}: Creating TTNN decoder")
            self.ttnn_decoder = TTNNSpeechT5Decoder(
                self.ttnn_device,
                preprocess_decoder_parameters(
                    model.speecht5.decoder,
                    self.decoder_config,
                    self.ttnn_device,
                    default_speaker_embedding,
                ),
                self.decoder_config,
                max_sequence_length=SpeechT5Constants.MAX_STEPS,
            )

            self.logger.info(f"Device {self.device_id}: Creating TTNN postnet")
            self.ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
                self.ttnn_device,
                preprocess_postnet_parameters(
                    model.speech_decoder_postnet, postnet_config, self.ttnn_device
                ),
                postnet_config,
            )

            # Initialize trace generator for faster inference
            try:
                self.logger.info(f"Device {self.device_id}: Creating trace generator")
                self.generator = SpeechT5Generator(
                    encoder=self.ttnn_encoder,
                    decoder=self.ttnn_decoder,
                    postnet=self.ttnn_postnet,
                    device=self.ttnn_device,
                    decoder_config=self.decoder_config,
                    max_steps=SpeechT5Constants.MAX_STEPS,
                    max_batch_size=1,
                    encoder_seq_len=128,
                )
                self.logger.info(
                    f"Device {self.device_id}: Trace generator initialized"
                )
            except Exception as e:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to initialize trace generator: {e}"
                )
                self.generator = None

            self.logger.info(
                f"Device {self.device_id}: All SpeechT5 models initialized successfully"
            )

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to initialize models: {e}"
            )
            raise SpeechT5ModelError(f"Model initialization failed: {str(e)}") from e

    def _warmup_models(self):
        """Warmup models by pre-compiling kernels and capturing traces."""
        try:
            self.logger.info(f"Device {self.device_id}: Pre-compiling postnet kernels")

            # Pre-compile postnet kernels BEFORE any trace capture
            dummy_decoder_output = ttnn.from_torch(
                torch.randn(1, 1, 1, self.decoder_config.hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            _ = self.ttnn_postnet(dummy_decoder_output)
            ttnn.deallocate(dummy_decoder_output)
            self.logger.info(f"Device {self.device_id}: Postnet kernels compiled")

            # Capture traces for all supported encoder sizes
            if self.generator is not None:
                self.logger.info(
                    f"Device {self.device_id}: Capturing traces for all encoder sizes"
                )
                self.generator.capture_all_traces(self.processor, batch_size=1)

                # Reset KV caches after warmup to clear stale values
                self.generator._reset_kv_caches()
                self.logger.info(
                    f"Device {self.device_id}: Traces captured and KV caches reset"
                )

        except Exception as e:
            self.logger.warning(
                f"Device {self.device_id}: Warmup failed (will compile on first request): {e}"
            )

    @log_execution_time(
        "SpeechT5 model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def load_model(self) -> bool:
        try:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.info(f"Device {device_id_int}: Loading SpeechT5 model...")

            # Device should already be initialized by set_device()
            if self.ttnn_device is None:
                raise DeviceInitializationError(
                    "Device not initialized. Call set_device() first."
                )

            # Initialize models
            try:
                await asyncio.to_thread(self._initialize_models)
                self.logger.info(
                    f"Device {device_id_int}: Model initialization completed"
                )
            except SpeechT5ModelError as e:
                self.logger.error(
                    f"Device {device_id_int}: Model initialization failed: {e}"
                )
                raise

            # Warmup: pre-compile kernels and capture traces
            try:
                await asyncio.to_thread(self._warmup_models)
                self.logger.info(f"Device {device_id_int}: Warmup completed")
            except Exception as e:
                self.logger.warning(
                    f"Device {device_id_int}: Warmup failed, will compile on first request: {e}"
                )

            return True

        except (DeviceInitializationError, SpeechT5ModelError):
            raise
        except Exception as e:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.error(f"Device {device_id_int}: Model loading failed: {e}")
            raise SpeechT5ModelError(
                f"Device {device_id_int}: Model loading failed: {str(e)}"
            ) from e

    def _prepare_speaker_embedding(self, request: TextToSpeechRequest) -> torch.Tensor:
        """Prepare speaker embedding for the request"""
        if request.speaker_embedding:
            # User provided embedding
            return self.speaker_manager.process_user_embedding(
                request.speaker_embedding
            )
        elif request.speaker_id:
            # Use pre-configured speaker
            return self.speaker_manager.get_speaker_embedding(request.speaker_id)
        else:
            # Use default speaker
            available_speakers = self.speaker_manager.list_available_speakers()
            if available_speakers:
                return self.speaker_manager.get_speaker_embedding(available_speakers[0])
            else:
                # No speakers available, use zero embedding
                self.logger.warning(
                    "No speaker embeddings available, using zero embedding"
                )
                return torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0)

    async def _generate_audio_with_task_id(
        self, text: str, speaker_embedding: torch.Tensor, task_id: str
    ):
        """Streaming audio generation with task ID"""
        async for result in self._generate_audio_optimized(text, speaker_embedding, True):
            result["task_id"] = task_id
            yield result

    async def _generate_audio_optimized(
        self, text: str, speaker_embedding: torch.Tensor, stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Optimized audio generation with KV cache, trace execution, and 2CQ overlap.

        This implementation follows the optimized pattern from demo_ttnn.py:
        - KV cache for O(1) per-step complexity
        - Trace execution for faster decoder inference
        - 2CQ event synchronization for overlapping position updates with CPU work
        - CPU accumulation of mel frames to avoid device allocations during trace
        """
        import time

        try:
            device = self.ttnn_device
            batch_size = 1
            num_mel_bins = SpeechT5Constants.NUM_MEL_BINS
            max_steps = SpeechT5Constants.MAX_STEPS

            # Performance tracking
            generation_start = time.time()
            ttft = None  # Time To First Token/Frame

            # Process input text
            inputs = self.processor(text=text, return_tensors="pt")
            token_ids = inputs["input_ids"]

            # Convert inputs to TTNN with L1 memory
            ttnn_input_ids = ttnn.from_torch(
                token_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_speaker_embeddings = ttnn.from_torch(
                speaker_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Determine if trace and KV cache are available
            use_kv_cache = self.generator is not None and self.decoder_config is not None
            enable_trace = use_kv_cache

            # Encoder forward pass (runs only once)
            encoder_start = time.time()
            encoder_output = self.ttnn_encoder(ttnn_input_ids)[0]
            encoder_time = time.time() - encoder_start

            # Setup for trace mode
            if enable_trace:
                self.generator.copy_encoder_output(encoder_output)
                encoder_output_for_decoder = self.generator.encoder_hidden_states
                kv_cache = self.generator.kv_cache
                cross_attn_cache = self.generator.cross_attn_cache
                self.generator._invalidate_cross_attn_cache()
            else:
                encoder_output_for_decoder = encoder_output
                kv_cache = None
                cross_attn_cache = None
                if use_kv_cache:
                    encoder_seq_len = encoder_output.shape[1]
                    kv_cache, cross_attn_cache = init_kv_cache(
                        self.decoder_config,
                        device,
                        max_batch_size=batch_size,
                        max_seq_len=max_steps + 10,
                        encoder_seq_len=encoder_seq_len,
                    )

            # Initial mel frame (zeros)
            output_sequence_ttnn = ttnn.from_torch(
                torch.zeros(batch_size, 1, num_mel_bins),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # For trace mode: accumulate mel frames on CPU to avoid device allocations
            spectrogram_frames_cpu = []
            spectrogram_ttnn = None
            steps_completed = 0
            current_input_ttnn = output_sequence_ttnn

            # 2CQ: Events for overlapping position updates with CPU work
            use_2cq = enable_trace and self.generator is not None
            op_event = None
            pos_event = None

            # Autoregressive generation loop
            decoder_loop_start = time.time()
            for step in range(max_steps):
                # Log progress every 50 steps
                if step > 0 and step % 50 == 0:
                    elapsed = time.time() - decoder_loop_start
                    current_rate = step / elapsed if elapsed > 0 else 0
                    self.logger.debug(
                        f"Device {self.device_id}: Step {step}/{max_steps}, "
                        f"Rate: {current_rate:.1f} tokens/s"
                    )

                if use_kv_cache and kv_cache is not None:
                    # KV cache mode: pass only the current frame (seq_len=1 after step 0)

                    if enable_trace and self.generator is not None:
                        # 2CQ: Wait for async position update from previous iteration
                        if use_2cq and pos_event is not None:
                            ttnn.wait_for_event(0, pos_event)  # CQ0 waits for CQ1
                            pos_event = None
                        else:
                            # Step 0 or non-2CQ: update position synchronously
                            self.generator._reset_decode_pos(step, batch_size)

                        # Preprocess: run prenet + PE addition OUTSIDE trace
                        preprocessed_hidden_states = self.ttnn_decoder.preprocess_decoder_inputs(
                            decoder_input_values=current_input_ttnn,
                            position_offset=step,
                        )

                        if step == 0:
                            # First iteration: non-traced to populate cross-attention cache
                            decoder_hidden_states = self.ttnn_decoder(
                                decoder_input_values=None,
                                encoder_hidden_states=encoder_output_for_decoder,
                                speaker_embeddings=None,
                                kv_cache=kv_cache,
                                cross_attn_cache=cross_attn_cache,
                                cross_attn_cache_valid=False,
                                current_decode_pos=self.generator.current_decode_pos,
                                preprocessed_hidden_states=preprocessed_hidden_states,
                                encoder_attention_mask=self.generator.encoder_attention_mask,
                            )
                            self.generator.cross_attn_cache_valid = True

                            # Capture trace after first iteration
                            if not self.generator.trace_compiled:
                                self.generator._capture_decoder_trace(preprocessed_hidden_states)
                        else:
                            # Step 1+: Execute trace (non-blocking for 2CQ overlap)
                            decoder_hidden_states = self.generator._execute_decoder_trace(
                                preprocessed_hidden_states, blocking=False
                            )
                            # Sync: create a copy to ensure trace output is ready for postnet
                            decoder_hidden_states = ttnn.to_memory_config(
                                decoder_hidden_states,
                                ttnn.L1_MEMORY_CONFIG,
                            )
                    else:
                        # KV cache without trace
                        current_pos = ttnn.from_torch(
                            torch.tensor([step], dtype=torch.int32),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )

                        decoder_hidden_states = self.ttnn_decoder(
                            decoder_input_values=current_input_ttnn,
                            encoder_hidden_states=encoder_output_for_decoder,
                            speaker_embeddings=ttnn_speaker_embeddings,
                            kv_cache=kv_cache,
                            cross_attn_cache=cross_attn_cache,
                            cross_attn_cache_valid=(step > 0),
                            current_decode_pos=current_pos,
                            position_offset=step,
                        )
                else:
                    # Standard mode: pass full sequence (slow, no KV cache)
                    decoder_hidden_states = self.ttnn_decoder(
                        decoder_input_values=output_sequence_ttnn,
                        encoder_hidden_states=encoder_output_for_decoder,
                        speaker_embeddings=ttnn_speaker_embeddings,
                    )

                # Postnet inference
                mel_before, mel_after, stop_logits = self.ttnn_postnet(decoder_hidden_states)

                # Capture TTFT after step 0's postnet (first mel frame produced)
                if step == 0 and ttft is None:
                    ttft = time.time() - generation_start

                # 2CQ: Start async position update for next iteration on CQ1
                if use_2cq and step < max_steps - 1:
                    op_event = ttnn.record_event(device, 0)  # Record on CQ0
                    ttnn.wait_for_event(1, op_event)  # CQ1 waits for CQ0
                    self.generator._reset_decode_pos(step + 1, batch_size, cq_id=1)
                    pos_event = ttnn.record_event(device, 1)  # Record on CQ1

                # Check stopping condition
                if step >= SpeechT5Constants.MIN_STEPS_FOR_STOP_CHECK:
                    if use_kv_cache and kv_cache is not None:
                        current_stop_logits = stop_logits
                    else:
                        reduction_factor = SpeechT5Constants.REDUCTION_FACTOR
                        stop_logits_shape = stop_logits.shape
                        total_mel_frames = stop_logits_shape[-1]
                        current_stop_logits = ttnn.slice(
                            stop_logits,
                            [0, total_mel_frames - reduction_factor],
                            [batch_size, total_mel_frames],
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )

                    sigmoid_logits = ttnn.sigmoid(
                        current_stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG
                    )
                    sum_prob = ttnn.sum(
                        sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
                    )
                    should_stop = ttnn.ge(
                        sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG
                    )
                    any_stop_scalar = ttnn.sum(should_stop)
                    if ttnn.to_torch(any_stop_scalar).item() > 0:
                        break

                # Extract new mel frames
                if use_kv_cache and kv_cache is not None:
                    # KV cache mode: mel_after has shape [batch, reduction_factor, mel_bins]
                    mel_after_shape = mel_after.shape
                    if len(mel_after_shape) == 4:
                        mel_frames = mel_after_shape[2]
                    else:
                        mel_frames = mel_after_shape[1]

                    new_frames_ttnn = mel_after
                    if len(mel_after_shape) == 4:
                        new_frames_ttnn = ttnn.reshape(
                            mel_after, [batch_size, mel_frames, num_mel_bins]
                        )

                    # CPU accumulation for trace mode
                    if enable_trace and self.generator is not None:
                        spectrogram_frames_cpu.append(
                            ttnn.to_torch(new_frames_ttnn).clone()
                        )
                    else:
                        if spectrogram_ttnn is None:
                            spectrogram_ttnn = new_frames_ttnn
                        else:
                            spectrogram_ttnn = ttnn.concat(
                                [spectrogram_ttnn, new_frames_ttnn],
                                dim=1,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                            )

                    # Get last frame for next iteration input
                    last_frame_ttnn = ttnn.slice(
                        new_frames_ttnn,
                        [0, mel_frames - 1, 0],
                        [batch_size, mel_frames, num_mel_bins],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    current_input_ttnn = last_frame_ttnn
                else:
                    # Standard mode: extract frames from full mel_after
                    current_seq_len = output_sequence_ttnn.shape[1]
                    start_idx = (current_seq_len - 1) * SpeechT5Constants.REDUCTION_FACTOR
                    end_idx = start_idx + SpeechT5Constants.REDUCTION_FACTOR

                    new_frames_ttnn = ttnn.slice(
                        mel_after,
                        [0, start_idx, 0],
                        [batch_size, end_idx, num_mel_bins],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )

                    if spectrogram_ttnn is None:
                        spectrogram_ttnn = new_frames_ttnn
                    else:
                        spectrogram_ttnn = ttnn.concat(
                            [spectrogram_ttnn, new_frames_ttnn],
                            dim=1,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )

                    # Extend sequence with last frame
                    last_frame_idx = start_idx + 1
                    last_frame_ttnn = ttnn.slice(
                        mel_after,
                        [0, last_frame_idx, 0],
                        [batch_size, last_frame_idx + 1, num_mel_bins],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    output_sequence_ttnn = ttnn.concat(
                        [output_sequence_ttnn, last_frame_ttnn],
                        dim=1,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )

                steps_completed += 1

                # Streaming: yield audio chunks periodically
                if stream and steps_completed % SpeechT5Constants.STREAMING_CHUNK_SIZE == 0:
                    if enable_trace and spectrogram_frames_cpu:
                        partial_spectrogram = torch.cat(spectrogram_frames_cpu, dim=1)
                    elif spectrogram_ttnn is not None:
                        partial_spectrogram = ttnn.to_torch(spectrogram_ttnn)
                    else:
                        continue

                    partial_audio = self.vocoder(partial_spectrogram)

                    audio_buffer = io.BytesIO()
                    sf.write(
                        audio_buffer,
                        partial_audio.squeeze().detach().numpy(),
                        SpeechT5Constants.SAMPLE_RATE,
                        format="WAV",
                    )
                    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode(
                        "utf-8"
                    )

                    yield {
                        "type": "streaming_chunk",
                        "chunk": PartialStreamingAudioResponse(
                            audio_chunk=audio_base64,
                            chunk_id=steps_completed // SpeechT5Constants.STREAMING_CHUNK_SIZE,
                            format="wav",
                            sample_rate=SpeechT5Constants.SAMPLE_RATE,
                        ),
                        "task_id": None,
                    }

            # End decoder loop timing
            decoder_loop_time = time.time() - decoder_loop_start

            # Calculate and log performance metrics
            if ttft is None:
                ttft = encoder_time  # Fallback if no steps completed
            avg_token_time = decoder_loop_time / max(steps_completed, 1)
            tokens_per_sec = 1.0 / avg_token_time if avg_token_time > 0 else 0

            self.logger.info(
                f"Device {self.device_id}: Generation completed - "
                f"Steps: {steps_completed}, "
                f"TTFT: {ttft*1000:.1f}ms, "
                f"Token/s: {tokens_per_sec:.2f}, "
                f"Encoder: {encoder_time*1000:.1f}ms, "
                f"Decoder loop: {decoder_loop_time:.3f}s"
            )

            # Build final spectrogram
            if enable_trace and spectrogram_frames_cpu:
                final_spectrogram = torch.cat(spectrogram_frames_cpu, dim=1)
            elif spectrogram_ttnn is not None:
                final_spectrogram = ttnn.to_torch(spectrogram_ttnn)
            else:
                final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

            # Generate final audio
            final_audio = self.vocoder(final_spectrogram)

            # Convert to base64
            audio_buffer = io.BytesIO()
            sf.write(
                audio_buffer,
                final_audio.squeeze().detach().numpy(),
                SpeechT5Constants.SAMPLE_RATE,
                format="WAV",
            )
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")

            # Calculate duration
            duration = len(final_audio.squeeze()) / SpeechT5Constants.SAMPLE_RATE

            # Log final summary with audio duration
            total_time = time.time() - generation_start
            rtf = total_time / duration if duration > 0 else 0  # Real-time factor
            self.logger.info(
                f"Device {self.device_id}: Audio generated - "
                f"Duration: {duration:.2f}s, "
                f"Total time: {total_time:.2f}s, "
                f"RTF: {rtf:.2f}x"
            )

            # Cleanup TTNN tensors
            ttnn.deallocate(ttnn_input_ids)
            ttnn.deallocate(ttnn_speaker_embeddings)
            ttnn.deallocate(encoder_output)
            ttnn.deallocate(output_sequence_ttnn)
            if spectrogram_ttnn is not None:
                ttnn.deallocate(spectrogram_ttnn)

            yield {
                "type": "final_result",
                "result": TextToSpeechResponse(
                    audio=audio_base64,
                    duration=duration,
                    sample_rate=SpeechT5Constants.SAMPLE_RATE,
                    format="wav",
                ),
                "task_id": None,
            }

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Audio generation failed: {e}")
            raise AudioGenerationError(f"Audio generation failed: {str(e)}") from e

    async def _run_inference_async(self, requests: list[TextToSpeechRequest]):
        """Main inference method"""
        try:
            # Validate prerequisites
            if (
                self.ttnn_encoder is None
                or self.ttnn_decoder is None
                or self.ttnn_postnet is None
            ):
                raise ModelNotLoadedError(
                    "Model components not loaded. Call load_model() first."
                )
            if self.ttnn_device is None:
                raise DeviceInitializationError("TTNN device not initialized")

            # Process single request (batch processing not implemented yet)
            if len(requests) > 1:
                self.logger.warning(
                    f"Device {self.device_id}: Batch processing not implemented. Processing only first of {len(requests)} requests"
                )

            request = requests[0]
            if request is None:
                raise TextProcessingError("Request cannot be None")

            if not request.text or not request.text.strip():
                raise TextProcessingError("Text cannot be empty")

            # Prepare speaker embedding
            speaker_embedding = self._prepare_speaker_embedding(request)
            request._speaker_embedding_array = speaker_embedding.detach().numpy()

            # Reset KV caches before each generation for clean state
            if self.generator is not None:
                self.generator._reset_kv_caches()

            # Return appropriate result based on streaming
            if request.stream:
                # For streaming, return the async generator
                return self._generate_audio_with_task_id(
                    request.text, speaker_embedding, request._task_id
                )
            else:
                # For non-streaming, collect and return final result
                final_result = None
                async for result in self._generate_audio_optimized(
                    request.text, speaker_embedding, False
                ):
                    result["task_id"] = request._task_id
                    final_result = result
                # Extract the actual TextToSpeechResponse from the dictionary
                if final_result and "result" in final_result:
                    return final_result["result"]
                return final_result

        except (
            ModelNotLoadedError,
            DeviceInitializationError,
            TextProcessingError,
            AudioGenerationError,
        ):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise InferenceError(f"Inference failed: {str(e)}") from e

    @log_execution_time(
        "Run SpeechT5 inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run_inference(self, requests: list[TextToSpeechRequest]):
        """Synchronous wrapper for async inference"""
        result = asyncio.run(self._run_inference_async(requests))
        # Wrap result in list as expected by device_worker
        return [result] if result is not None else []
