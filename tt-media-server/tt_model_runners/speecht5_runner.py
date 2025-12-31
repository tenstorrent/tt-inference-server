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
sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts_final/tt-metal")
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNSpeechT5Encoder,
    TTNNEncoderConfig,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNSpeechT5Decoder,
    TTNNDecoderConfig,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNSpeechT5SpeechDecoderPostnet,
    TTNNPostNetConfig,
    preprocess_postnet_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import SpeechT5Generator

from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse, PartialStreamingAudioResponse
from telemetry.telemetry_client import TelemetryEvent
from utils.speaker_embeddings import SpeakerEmbeddingsManager
import ttnn
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time

class SpeechT5Constants:
    MAX_STEPS = 10  # Reduced from 100 to avoid kernel compilation issues during warmup
    SAMPLE_RATE = 16000
    REDUCTION_FACTOR = 2
    STREAMING_CHUNK_SIZE = 20  # Generate audio chunks every 20 mel frames
    MAX_CLEANUP_RETRIES = 3
    RETRY_DELAY_SECONDS = 1

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

        # Limit threading for stability during inference
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        # Explicitly disable fabric for non-galaxy devices
        if not settings.is_galaxy:
            os.environ['TT_METAL_FABRIC_DISABLE'] = '1'

    def _set_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        # For now use all available devices
        return self._mesh_device()

    def _prepare_device_params(self):
        try:
            device_params = {'l1_small_size': 150000, 'trace_region_size': 10000000}
            return self.get_updated_device_params(device_params)
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Device parameter preparation failed: {e}")
            raise DeviceInitializationError(f"Device parameter preparation failed: {str(e)}") from e

    def _configure_fabric(self, updated_device_params):
        try:
            fabric_config = updated_device_params.pop("fabric_config", None)

            # For non-galaxy devices, ensure fabric is disabled
            if not settings.is_galaxy:
                self.logger.info(f"Device {self.device_id}: Disabling fabric for non-galaxy device")
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                return None

            self._set_fabric(fabric_config)
            return fabric_config
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Fabric configuration failed: {e}")
            raise DeviceInitializationError(f"Fabric configuration failed: {str(e)}") from e

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        try:
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
        except Exception as e:
            try:
                self._reset_fabric(fabric_config)
            except Exception as reset_error:
                self.logger.warning(f"Device {self.device_id}: Failed to reset fabric after device initialization failure: {reset_error}")
            self.logger.error(f"Device {self.device_id}: Mesh device initialization failed: {e}")
            raise DeviceInitializationError(f"Mesh device initialization failed: {str(e)}") from e
        return mesh_device

    def _mesh_device(self):
        try:
            # Get available devices
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise DeviceInitializationError("No TTNN devices available")
            self.logger.info(f"Device {self.device_id}: Found {len(device_ids)} available TTNN devices: {device_ids}")

            mesh_shape = ttnn.MeshShape(settings.device_mesh_shape)
            updated_device_params = self._prepare_device_params()
            fabric_config = self._configure_fabric(updated_device_params)
            mesh_device = self._initialize_mesh_device(mesh_shape, updated_device_params, fabric_config)

            self.logger.info(f"Device {self.device_id}: Successfully created multidevice with {mesh_device.get_num_devices()} devices")
            return mesh_device

        except DeviceInitializationError:
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Unexpected error during device initialization: {e}")
            raise DeviceInitializationError(f"Unexpected device initialization error: {str(e)}") from e

    def set_device(self):
        """Override to use simple device initialization like the demo"""
        if self.ttnn_device is None:
            # Handle empty device_id by defaulting to 0
            device_id_int = int(self.device_id) if self.device_id else 0
            
            # N150 device validation: ensure we're using device 0
            if device_id_int != 0:
                self.logger.warning(f"Device {device_id_int}: TTS on N150 is designed for device 0, but got device {device_id_int}")
            
            self.logger.info(f"Device {device_id_int}: Initializing simple TTNN device for N150 (like demo_ttnn.py)")
            # Use same device initialization as demo_ttnn.py
            self.ttnn_device = ttnn.open_device(device_id=device_id_int, l1_small_size=24576)
            self.ttnn_device.enable_program_cache()
            self.logger.info(f"Device {device_id_int}: Simple device initialized successfully for N150")
        self.max_batch_size = self.settings.max_batch_size
        return self.ttnn_device

    def close_device(self):
        """Override to use simple device closing like the demo"""
        try:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.info(f"Device {device_id_int}: Closing simple device...")
            if self.ttnn_device is not None:
                ttnn.close_device(self.ttnn_device)
                self.logger.info(f"Device {device_id_int}: Successfully closed simple device")
            else:
                self.logger.info(f"Device {device_id_int}: Device is None, no need to close")
        except Exception as e:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.error(f"Device {device_id_int}: Failed to close device: {e}")
            raise RuntimeError(f"Device {device_id_int}: Device cleanup failed: {str(e)}") from e

    def _handle_load_failure_cleanup(self, device):
        if device is None:
            try:
                self.close_device(None)
            except Exception as cleanup_error:
                self.logger.warning(f"Device {self.device_id}: Failed to cleanup device after failure: {cleanup_error}")

    def _initialize_models(self):
        """Initialize SpeechT5 models and components"""
        try:
            # Load HuggingFace models
            model_name = SupportedModels.SPEECHT5_TTS.value
            self.logger.info(f"Device {self.device_id}: Loading SpeechT5 models from {model_name}")

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

            # Get a default speaker embedding for model initialization
            available_speakers = self.speaker_manager.list_available_speakers()
            if not available_speakers:
                # Create a zero embedding if no speakers are available
                self.logger.warning("No speaker embeddings available, using zero embedding for initialization")
                default_speaker_embedding = torch.zeros(self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32).unsqueeze(0)
            else:
                default_speaker_embedding = self.speaker_manager.get_speaker_embedding(available_speakers[0])

            # Create TTNN models
            self.logger.info(f"Device {self.device_id}: Creating TTNN encoder")
            self.ttnn_encoder = TTNNSpeechT5Encoder(
                self.ttnn_device,
                preprocess_encoder_parameters(model.speecht5.encoder, encoder_config, self.ttnn_device),
                encoder_config,
            )

            self.logger.info(f"Device {self.device_id}: Creating TTNN decoder")
            self.ttnn_decoder = TTNNSpeechT5Decoder(
                self.ttnn_device,
                preprocess_decoder_parameters(model.speecht5.decoder, decoder_config, self.ttnn_device, default_speaker_embedding),
                decoder_config,
                max_sequence_length=SpeechT5Constants.MAX_STEPS,
            )

            self.logger.info(f"Device {self.device_id}: Creating TTNN postnet")
            self.ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
                self.ttnn_device,
                preprocess_postnet_parameters(model.speech_decoder_postnet, postnet_config, self.ttnn_device),
                postnet_config,
            )

            # Optional: Initialize trace generator for faster inference
            try:
                self.generator = SpeechT5Generator(
                    self.ttnn_encoder,
                    self.ttnn_decoder,
                    self.ttnn_postnet,
                    self.ttnn_device,
                    default_speaker_embedding
                )
                self.logger.info(f"Device {self.device_id}: Trace generator initialized")
            except Exception as e:
                self.logger.warning(f"Device {self.device_id}: Failed to initialize trace generator: {e}")
                self.generator = None

            self.logger.info(f"Device {self.device_id}: All SpeechT5 models initialized successfully")

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to initialize models: {e}")
            raise SpeechT5ModelError(f"Model initialization failed: {str(e)}") from e

    @log_execution_time("SpeechT5 model load", TelemetryEvent.DEVICE_WARMUP, os.environ.get("TT_VISIBLE_DEVICES"))
    async def load_model(self) -> bool:
        try:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.info(f"Device {device_id_int}: Loading SpeechT5 model...")

            # Device should already be initialized by set_device()
            if self.ttnn_device is None:
                raise DeviceInitializationError("Device not initialized. Call set_device() first.")

            # Enable program cache for faster inference (already done in set_device)
            # self.ttnn_device.enable_program_cache()

            # Initialize models
            try:
                await asyncio.to_thread(self._initialize_models)
                self.logger.info(f"Device {device_id_int}: Model initialization completed")
            except SpeechT5ModelError as e:
                self.logger.error(f"Device {device_id_int}: Model initialization failed: {e}")
                self._handle_load_failure_cleanup(self.ttnn_device)
                raise

            # Skip warmup for now to avoid kernel compilation issues
            # Kernels will be compiled on first inference request
            self.logger.info(f"Device {device_id_int}: Skipping warmup - kernels will compile on first request")

            return True

        except (DeviceInitializationError, SpeechT5ModelError):
            raise
        except Exception as e:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.error(f"Device {device_id_int}: Model loading failed: {e}")
            raise SpeechT5ModelError(f"Device {device_id_int}: Model loading failed: {str(e)}") from e

    def _prepare_speaker_embedding(self, request: TextToSpeechRequest) -> torch.Tensor:
        """Prepare speaker embedding for the request"""
        if request.speaker_embedding:
            # User provided embedding
            return self.speaker_manager.process_user_embedding(request.speaker_embedding)
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
                self.logger.warning("No speaker embeddings available, using zero embedding")
                return torch.zeros(self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32).unsqueeze(0)

    async def _generate_audio_with_task_id(self, text: str, speaker_embedding: torch.Tensor, task_id: str):
        """Streaming audio generation with task ID"""
        async for result in self._generate_audio_sync(text, speaker_embedding, True):
            result['task_id'] = task_id
            yield result

    async def _generate_audio_sync(self, text: str, speaker_embedding: torch.Tensor, stream: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """Synchronous audio generation that can yield streaming chunks"""
        try:
            # Process input text
            inputs = self.processor(text=text, return_tensors="pt")
            token_ids = inputs["input_ids"]

            # Convert inputs to TTNN with L1 memory
            ttnn_input_ids = ttnn.from_torch(
                token_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.ttnn_device, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn_speaker_embeddings = ttnn.from_torch(
                speaker_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Initialize decoder sequence
            batch_size = token_ids.shape[0]
            num_mel_bins = 80  # Standard for SpeechT5
            output_sequence_ttnn = ttnn.from_torch(
                torch.zeros(batch_size, 1, num_mel_bins),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            spectrogram_ttnn = None  # Will be built incrementally on device
            steps_completed = 0

            # Autoregressive generation loop
            for step in range(SpeechT5Constants.MAX_STEPS):
                # Decoder inference
                decoder_hidden_states = self.ttnn_decoder(
                    decoder_input_values=output_sequence_ttnn,
                    encoder_hidden_states=self.ttnn_encoder(ttnn_input_ids)[0],
                    speaker_embeddings=ttnn_speaker_embeddings,
                )

                # Postnet inference
                mel_before, mel_after, stop_logits = self.ttnn_postnet(decoder_hidden_states)

                # Check stopping condition
                sigmoid_logits = ttnn.sigmoid(stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG)
                sum_prob = ttnn.sum(sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                should_stop = ttnn.ge(sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
                any_stop_scalar = ttnn.sum(should_stop)
                if ttnn.to_torch(any_stop_scalar).item() > 0:
                    break

                # Extract new mel frames
                current_seq_len = output_sequence_ttnn.shape[1]
                start_idx = (current_seq_len - 1) * SpeechT5Constants.REDUCTION_FACTOR
                end_idx = start_idx + SpeechT5Constants.REDUCTION_FACTOR

                # Slice the new frames from mel_after
                new_frames_ttnn = ttnn.slice(
                    mel_after,
                    [0, start_idx, 0],  # start indices [batch, seq, mel_bins]
                    [batch_size, end_idx, num_mel_bins],  # end indices
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

                # Build spectrogram incrementally
                if spectrogram_ttnn is None:
                    spectrogram_ttnn = new_frames_ttnn
                else:
                    spectrogram_ttnn = ttnn.concat(
                        [spectrogram_ttnn, new_frames_ttnn], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
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
                    [output_sequence_ttnn, last_frame_ttnn], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
                )

                steps_completed += 1

                # Streaming: yield audio chunks periodically
                if stream and steps_completed % SpeechT5Constants.STREAMING_CHUNK_SIZE == 0:
                    partial_spectrogram = ttnn.to_torch(spectrogram_ttnn)
                    partial_audio = self.vocoder(partial_spectrogram)

                    # Convert to base64
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, partial_audio.squeeze().detach().numpy(), SpeechT5Constants.SAMPLE_RATE, format='WAV')
                    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

                    yield {
                        'type': 'streaming_chunk',
                        'chunk': PartialStreamingAudioResponse(
                            audio_chunk=audio_base64,
                            chunk_id=steps_completed // SpeechT5Constants.STREAMING_CHUNK_SIZE,
                            format="wav",
                            sample_rate=SpeechT5Constants.SAMPLE_RATE
                        ),
                        'task_id': None  # Will be set by caller
                    }

            # Transfer final spectrogram from device to host
            if spectrogram_ttnn is not None:
                final_spectrogram = ttnn.to_torch(spectrogram_ttnn)
            else:
                final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

            # Generate final audio
            final_audio = self.vocoder(final_spectrogram)

            # Convert to base64
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, final_audio.squeeze().detach().numpy(), SpeechT5Constants.SAMPLE_RATE, format='WAV')
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

            # Calculate duration
            duration = len(final_audio.squeeze()) / SpeechT5Constants.SAMPLE_RATE

            yield {
                'type': 'final_result',
                'result': TextToSpeechResponse(
                    audio=audio_base64,
                    duration=duration,
                    sample_rate=SpeechT5Constants.SAMPLE_RATE,
                    format="wav"
                ),
                'task_id': None  # Will be set by caller
            }

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Audio generation failed: {e}")
            raise AudioGenerationError(f"Audio generation failed: {str(e)}") from e

    async def _run_inference_async(self, requests: list[TextToSpeechRequest]):
        """Main inference method"""
        try:
            # Validate prerequisites
            if self.ttnn_encoder is None or self.ttnn_decoder is None or self.ttnn_postnet is None:
                raise ModelNotLoadedError("Model components not loaded. Call load_model() first.")
            if self.ttnn_device is None:
                raise DeviceInitializationError("TTNN device not initialized")

            # Process single request (batch processing not implemented yet)
            if len(requests) > 1:
                self.logger.warning(f"Device {self.device_id}: Batch processing not implemented. Processing only first of {len(requests)} requests")

            request = requests[0]
            if request is None:
                raise TextProcessingError("Request cannot be None")

            if not request.text or not request.text.strip():
                raise TextProcessingError("Text cannot be empty")

            # Prepare speaker embedding
            speaker_embedding = self._prepare_speaker_embedding(request)
            request._speaker_embedding_array = speaker_embedding.detach().numpy()

            # Return appropriate result based on streaming
            if request.stream:
                # For streaming, return the async generator
                return self._generate_audio_with_task_id(request.text, speaker_embedding, request._task_id)
            else:
                # For non-streaming, collect and return final result
                final_result = None
                async for result in self._generate_audio_sync(request.text, speaker_embedding, False):
                    result['task_id'] = request._task_id
                    final_result = result
                # Extract the actual TextToSpeechResponse from the dictionary
                if final_result and 'result' in final_result:
                    return final_result['result']
                return final_result

        except (ModelNotLoadedError, DeviceInitializationError, TextProcessingError, AudioGenerationError):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise InferenceError(f"Inference failed: {str(e)}") from e

    @log_execution_time("Run SpeechT5 inference", TelemetryEvent.MODEL_INFERENCE, os.environ.get("TT_VISIBLE_DEVICES"))
    def run_inference(self, requests: list[TextToSpeechRequest]):
        """Synchronous wrapper for async inference"""
        result = asyncio.run(self._run_inference_async(requests))
        # Wrap result in list as expected by device_worker
        return [result] if result is not None else []
