# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import base64
import io
import os
from typing import Any, AsyncGenerator, Dict

import soundfile as sf
import torch
import ttnn
from config.constants import SupportedModels
from config.settings import settings
from device_workers.worker_utils import setup_cpu_threading_limits
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import (
    TextToSpeechResponse,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNDecoderConfig,
    TTNNSpeechT5Decoder,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNEncoderConfig,
    TTNNSpeechT5Encoder,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import (
    SpeechT5Generator,
)
from models.experimental.speecht5_tts.demo_ttnn import generate_speech_ttnn
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNPostNetConfig,
    TTNNSpeechT5SpeechDecoderPostnet,
    preprocess_postnet_parameters,
)
from telemetry.telemetry_client import TelemetryEvent
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.speaker_embeddings import SpeakerEmbeddingsManager


class SpeechT5Constants:
    MAX_STEPS = 768  # Maximum generation steps
    MIN_STEPS = 10  # Minimum steps before checking stop condition
    SAMPLE_RATE = 16000
    REDUCTION_FACTOR = 2


class TTSpeechT5Runner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id)
        self.processor = None
        self.vocoder = None
        self.ttnn_encoder = None
        self.ttnn_decoder = None
        self.ttnn_postnet = None
        self.generator = None  # For trace execution with KV cache
        self.speaker_manager = None
        self.decoder_config = None  # Store for generator use
        self.default_speaker_embedding = None

        # Limit threading for stability during inference
        setup_cpu_threading_limits("1")

        # Explicitly disable fabric for non-galaxy devices
        if not settings.is_galaxy:
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    def get_pipeline_device_params(self):
        # Enable 2 command queues (required by the generator)
        device_params = {
            "l1_small_size": 300000,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
        return device_params

    def _initialize_models(self):
        """Initialize SpeechT5 models and components"""
        try:
            # Get actual device from mesh device (like demo_ttnn.py does)
            if hasattr(self.ttnn_device, "get_devices"):
                actual_device = self.ttnn_device.get_devices()[0]
                self.logger.info(f"Device {self.device_id}: Using actual device from mesh")
            else:
                actual_device = self.ttnn_device

            # Store for later use
            self._actual_device = actual_device

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

            decoder_config = TTNNDecoderConfig(
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

            # Store decoder_config for generator use
            self.decoder_config = decoder_config

            # Get a default speaker embedding for model initialization
            available_speakers = self.speaker_manager.list_available_speakers()
            if not available_speakers:
                # Create a zero embedding if no speakers are available
                self.logger.warning(
                    "No speaker embeddings available, using zero embedding for initialization"
                )
                self.default_speaker_embedding = torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0)
            else:
                self.default_speaker_embedding = (
                    self.speaker_manager.get_speaker_embedding(available_speakers[0])
                )
            default_speaker_embedding = self.default_speaker_embedding

            # Create TTNN models (matching demo_ttnn.py pattern:
            # encoder uses self.ttnn_device, decoder/postnet/generator use actual_device)
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
                actual_device,
                preprocess_decoder_parameters(
                    model.speecht5.decoder,
                    decoder_config,
                    actual_device,
                    default_speaker_embedding,
                ),
                decoder_config,
                max_sequence_length=SpeechT5Constants.MAX_STEPS,
            )

            self.logger.info(f"Device {self.device_id}: Creating TTNN postnet")
            self.ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
                actual_device,
                preprocess_postnet_parameters(
                    model.speech_decoder_postnet, postnet_config, actual_device
                ),
                postnet_config,
            )

            # CRITICAL: Pre-compile postnet kernels BEFORE any trace capture
            # This prevents conv2d kernel recompilation while trace is active (causes hangs)
            self.logger.info(f"Device {self.device_id}: Pre-compiling postnet kernels...")
            dummy_decoder_output = ttnn.from_torch(
                torch.randn(1, 1, 1, decoder_config.hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=actual_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            _ = self.ttnn_postnet(dummy_decoder_output)
            ttnn.deallocate(dummy_decoder_output)
            self.logger.info(f"Device {self.device_id}: Postnet kernels pre-compiled!")

            # Initialize trace generator for faster inference with KV cache
            try:
                self.generator = SpeechT5Generator(
                    encoder=self.ttnn_encoder,
                    decoder=self.ttnn_decoder,
                    postnet=self.ttnn_postnet,
                    device=actual_device,
                    decoder_config=decoder_config,
                    max_steps=SpeechT5Constants.MAX_STEPS,
                    max_batch_size=1,
                    encoder_seq_len=128,
                )
                self.logger.info(
                    f"Device {self.device_id}: Trace generator initialized with KV cache support"
                )
            except Exception as e:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to initialize trace generator: {e}"
                )
                self.generator = None

            self.logger.info(
                f"Device {self.device_id}: All SpeechT5 models initialized successfully"
            )

            # Warmup pattern (following demo_ttnn.py):
            # Run synchronously in this thread to match demo behavior
            if self.generator is not None:
                self.logger.info(f"Device {self.device_id}: Running warmup inference...")

                # Get a speaker embedding for warmup
                available_speakers = self.speaker_manager.list_available_speakers()
                if available_speakers:
                    warmup_speaker = self.speaker_manager.get_speaker_embedding(available_speakers[0])
                else:
                    warmup_speaker = torch.zeros(512, dtype=torch.float32).unsqueeze(0)

                # Warmup inference
                _ = generate_speech_ttnn(
                    text="Hello world, this is a warmup test.",
                    speaker_embeddings=warmup_speaker,
                    processor=self.processor,
                    vocoder=self.vocoder,
                    ttnn_encoder=self.ttnn_encoder,
                    ttnn_decoder=self.ttnn_decoder,
                    ttnn_postnet=self.ttnn_postnet,
                    device=actual_device,
                    max_steps=SpeechT5Constants.MAX_STEPS,
                    warmup_mode=True,
                    generator=self.generator,
                    use_kv_cache=True,
                    decoder_config=self.decoder_config,
                )
                self.logger.info(f"Device {self.device_id}: Warmup inference completed")

                # Capture all traces
                self.logger.info(f"Device {self.device_id}: Capturing traces for all encoder sizes...")
                self.generator.capture_all_traces(self.processor, batch_size=1)
                self.logger.info(f"Device {self.device_id}: All traces captured")

                # Reset KV caches
                self.generator._reset_kv_caches()
                self.logger.info(f"Device {self.device_id}: KV caches reset - ready for inference")
            else:
                self.logger.warning(
                    f"Device {self.device_id}: No generator available, running without trace optimization"
                )

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to initialize models: {e}"
            )
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    @log_execution_time(
        "SpeechT5 model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        try:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.info(f"Device {device_id_int}: Loading SpeechT5 model...")

            # Device should already be initialized by set_device()
            if self.ttnn_device is None:
                raise ValueError("Device not initialized. Call set_device() first.")

            # Enable program cache for faster inference (already done in set_device)
            # self.ttnn_device.enable_program_cache()

            # Initialize models
            try:
                await asyncio.to_thread(self._initialize_models)
                self.logger.info(
                    f"Device {device_id_int}: Model initialization completed"
                )
            except RuntimeError as e:
                self.logger.error(
                    f"Device {device_id_int}: Model initialization failed: {e}"
                )
                raise

            # Warmup is done in _initialize_models (runs in thread)
            self.logger.info(f"Device {device_id_int}: Warmup complete")
            return True

        except Exception as e:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.error(f"Device {device_id_int}: Model loading failed: {e}")
            raise RuntimeError(
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

    async def _generate_audio_sync(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Audio generation using optimized generate_speech_ttnn with trace"""
        try:
            # Use the optimized generate_speech_ttnn from demo_ttnn.py
            # This uses KV cache and trace for fast inference
            speech = generate_speech_ttnn(
                text=text,
                speaker_embeddings=speaker_embedding,
                processor=self.processor,
                vocoder=self.vocoder,
                ttnn_encoder=self.ttnn_encoder,
                ttnn_decoder=self.ttnn_decoder,
                ttnn_postnet=self.ttnn_postnet,
                device=self.ttnn_device,
                max_steps=SpeechT5Constants.MAX_STEPS,
                return_stats=False,
                warmup_mode=False,
                generator=self.generator,
                use_kv_cache=True,
                decoder_config=self.decoder_config,
            )

            # Convert audio to base64
            audio_buffer = io.BytesIO()
            sf.write(
                audio_buffer,
                speech.squeeze().detach().numpy(),
                SpeechT5Constants.SAMPLE_RATE,
                format="WAV",
            )
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")

            # Calculate duration
            duration = len(speech.squeeze()) / SpeechT5Constants.SAMPLE_RATE

            yield {
                "type": "final_result",
                "result": TextToSpeechResponse(
                    audio=audio_base64,
                    duration=duration,
                    sample_rate=SpeechT5Constants.SAMPLE_RATE,
                    format="wav",
                ),
                "task_id": None,  # Will be set by caller
            }

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Audio generation failed: {e}")
            raise RuntimeError(f"Audio generation failed: {str(e)}") from e

    async def _run_async(self, requests: list[TextToSpeechRequest]):
        """Main inference method"""
        try:
            # Validate prerequisites
            if (
                self.ttnn_encoder is None
                or self.ttnn_decoder is None
                or self.ttnn_postnet is None
            ):
                raise RuntimeError("Model components not loaded. Call warmup() first.")
            if self.ttnn_device is None:
                raise ValueError("TTNN device not initialized")

            # Process single request (batch processing not implemented yet)
            if len(requests) > 1:
                self.logger.warning(
                    f"Device {self.device_id}: Batch processing not implemented. Processing only first of {len(requests)} requests"
                )

            request = requests[0]
            if request is None:
                raise ValueError("Request cannot be None")

            if not request.text or not request.text.strip():
                raise ValueError("Text cannot be empty")

            # Prepare speaker embedding
            speaker_embedding = self._prepare_speaker_embedding(request)
            request._speaker_embedding_array = speaker_embedding.detach().numpy()

            # Collect and return final result
            final_result = None
            async for result in self._generate_audio_sync(
                request.text, speaker_embedding
            ):
                result["task_id"] = request._task_id
                final_result = result
            # Extract the actual TextToSpeechResponse from the dictionary
            if final_result and "result" in final_result:
                return final_result["result"]
            return final_result

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    @log_execution_time(
        "Run SpeechT5 inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[TextToSpeechRequest]):
        """Synchronous wrapper for async inference"""
        result = asyncio.run(self._run_async(requests))
        # Wrap result in list as expected by device_worker
        return [result] if result is not None else []
