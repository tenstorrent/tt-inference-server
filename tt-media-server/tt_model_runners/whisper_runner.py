# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from config.constants import SupportedModels
from config.settings import settings
import time
import torch
from tqdm import tqdm
import os
from domain.audio_transcription_request import AudioTranscriptionRequest
import ttnn
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from utils.transcript_utils import TranscriptUtils
from domain.transcription_response import TranscriptionResponse, TranscriptionSegment, PartialStreamingTranscriptionResponse
from utils.logger import TTLogger
import numpy as np

from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache
from models.common.generation_utils import get_logits_processor

class WhisperConstants:
    TASK_TRANSCRIBE = "transcribe"
    LANGUAGE_ENGLISH = "English"
    MAX_CLEANUP_RETRIES = 3
    RETRY_DELAY_SECONDS = 1

class WhisperModelError(Exception):
    """Base exception for Whisper model errors"""
    pass

class ModelNotLoadedError(WhisperModelError):
    """Raised when attempting inference without loaded model"""
    pass

class AudioProcessingError(WhisperModelError):
    """Raised when audio data processing fails"""
    pass

class DeviceInitializationError(WhisperModelError):
    """Raised when device initialization fails"""
    pass

class InferenceError(WhisperModelError):
    """Error occurred during model inference"""
    pass

class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds timeout limit"""
    pass

class DeviceCleanupError(WhisperModelError):
    """Error occurred during device cleanup"""
    pass

class TTWhisperRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()
        self.ttnn_device = None
        self.pipeline = None
        self.ttnn_model = None
        # Limit threading for stability during inference
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'                     

    def _set_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        # for now use all available devices
        return self._mesh_device()

    def _prepare_device_params(self):
        try:
            device_params = {'l1_small_size': WHISPER_L1_SMALL_SIZE}
            return self.get_updated_device_params(device_params)
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Device parameter preparation failed: {e}")
            raise DeviceInitializationError(f"Device parameter preparation failed: {str(e)}") from e

    def _configure_fabric(self, updated_device_params):
        try:
            fabric_config = updated_device_params.pop("fabric_config", None)
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

    def _verify_device_initialization(self, mesh_device, num_devices_requested):
        try:
            actual_device_count = mesh_device.get_num_devices()
            if actual_device_count != num_devices_requested:
                self.logger.warning(
                    f"Device {self.device_id}: Requested {num_devices_requested} devices but got {actual_device_count}"
                )
        except Exception as e:
            self.logger.warning(f"Device {self.device_id}: Could not verify device count: {e}")

    def _mesh_device(self):
        try:
            # Get available devices
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise DeviceInitializationError("No TTNN devices available")
            self.logger.info(f"Device {self.device_id}: Found {len(device_ids)} available TTNN devices: {device_ids}")

            # Always fixed for whisper!
            mesh_shape = ttnn.MeshShape(settings.device_mesh_shape)
            num_devices_requested =1

            # Prepare device parameters
            updated_device_params = self._prepare_device_params()

            # Configure fabric
            fabric_config = self._configure_fabric(updated_device_params)

            # Initialize mesh device
            mesh_device = self._initialize_mesh_device(mesh_shape, updated_device_params, fabric_config)

            # Verify initialization
            self._verify_device_initialization(mesh_device, num_devices_requested)

            self.logger.info(f"Device {self.device_id}: Successfully created multidevice with {mesh_device.get_num_devices()} devices")
            return mesh_device

        except DeviceInitializationError:
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Unexpected error during device initialization: {e}")
            raise DeviceInitializationError(f"Unexpected device initialization error: {str(e)}") from e

    def close_device(self, mesh_device):
        for attempt in range(WhisperConstants.MAX_CLEANUP_RETRIES):
            try:
                self.logger.info(f"Device {self.device_id}: Closing mesh device (attempt {attempt + 1}/{WhisperConstants.MAX_CLEANUP_RETRIES})")
                if mesh_device is not None:
                    ttnn.close_mesh_device(mesh_device)
                    self.logger.info(f"Device {self.device_id}: Successfully closed mesh device")
                else:
                    self.logger.info(f"Device {self.device_id}: Device is None, no need to close")
                return  # Success, exit early

            except Exception as e:
                self.logger.warning(f"Device {self.device_id}: Attempt {attempt + 1} failed to close device: {e}")
                if attempt == WhisperConstants.MAX_CLEANUP_RETRIES - 1:  # Last attempt
                    self.logger.error(f"Device {self.device_id}: Failed to close device after {WhisperConstants.MAX_CLEANUP_RETRIES} attempts: {e}")
                    raise DeviceCleanupError(f"Device {self.device_id}: Device cleanup failed after {WhisperConstants.MAX_CLEANUP_RETRIES} attempts: {str(e)}") from e
                time.sleep(WhisperConstants.RETRY_DELAY_SECONDS)  # Brief delay before retry

    def _handle_load_failure_cleanup(self, device):
        if device is None:
            try:
                self.close_device(None)
            except Exception as cleanup_error:
                self.logger.warning(f"Device {self.device_id}: Failed to cleanup device after failure: {cleanup_error}")

    @log_execution_time("Whisper model load")
    async def load_model(self, device) -> bool:
        try:
            self.logger.info(f"Device {self.device_id}: Loading Whisper model...")

            # Initialize device
            try:
                if device is None:
                    self.ttnn_device = self._mesh_device()
                    self.mesh_device = self.ttnn_device  # Store reference for cleanup
                else:
                    self.ttnn_device = device
                    self.mesh_device = device
            except DeviceInitializationError as e:
                self.logger.error(f"Device {self.device_id}: Device initialization failed: {e}")
                raise

            # Load model components
            try:
                self.ttnn_model = ttnn_optimized_functional_whisper
                self.pipeline = await self._create_functional_whisper_for_conditional_generation_inference_pipeline()
                self.logger.info(f"Device {self.device_id}: Model pipeline created successfully")
            except Exception as e:
                self.logger.error(f"Device {self.device_id}: Model pipeline creation failed: {e}")
                self._handle_load_failure_cleanup(device)
                raise WhisperModelError(f"Device {self.device_id}: Model pipeline creation failed: {str(e)}") from e

            self.logger.info(f"Device {self.device_id}: Whisper model loaded and pipeline ready")

            # Warmup
            try:
                dummy_audio = np.zeros(settings.default_sample_rate, dtype=np.float32)
                self.logger.info(f"Device {self.device_id}: Starting model warmup with {len(dummy_audio)} samples")
                await self.pipeline(dummy_audio)
                self.logger.info(f"Device {self.device_id}: Model warmup completed successfully")
            except Exception as e:
                self.logger.error(f"Device {self.device_id}: Model warmup failed: {e}")
                self.pipeline = None
                self.ttnn_model = None
                self._handle_load_failure_cleanup(device)
                raise WhisperModelError(f"Device {self.device_id}: Model warmup failed: {str(e)}") from e

            return True

        except (DeviceInitializationError, WhisperModelError):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Model loading failed: {e}")
            raise WhisperModelError(f"Device {self.device_id}: Model loading failed: {str(e)}") from e

    async def _execute_pipeline(self, audio_data, stream, return_perf_metrics):
        """Main pipeline execution method"""
        try:
            if stream:
                # Return the async generator
                return self._execute_pipeline_streaming(audio_data, return_perf_metrics)
            else:
                # Return the single result
                return await self._execute_pipeline_non_streaming(audio_data, return_perf_metrics)
            
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Pipeline execution failed: {e}")
            raise InferenceError(f"Audio transcription failed: {str(e)}") from e

    async def _execute_pipeline_streaming(self, audio_data, return_perf_metrics):
        """Async generator for streaming results"""
        generator = await self.pipeline(
            audio_data,
            stream=True,
            return_perf_metrics=return_perf_metrics
        )
        
        for item in generator:
            yield item

    async def _execute_pipeline_non_streaming(self, audio_data, return_perf_metrics):
        """Non-streaming pipeline execution"""
        result = await self.pipeline(
            audio_data,
            stream=False,
            return_perf_metrics=return_perf_metrics
        )
        
        if result is None:
            raise InferenceError("Pipeline returned None result")
            
        return result

    @log_execution_time("Run Whisper inference")
    def run_inference(self, requests: list[AudioTranscriptionRequest]):
        """Synchronous wrapper for async inference"""
        return asyncio.run(self._run_inference_async(requests))

    async def _run_inference_async(self, requests: list[AudioTranscriptionRequest]):
        """Main inference method - validates input and routes to appropriate processing"""
        try:
            # Validate prerequisites and input
            if self.pipeline is None:
                raise ModelNotLoadedError("Model pipeline not loaded. Call load_model() first.")
            if self.ttnn_device is None:
                raise DeviceInitializationError("TTNN device not initialized")
            request = self._validate_and_extract_request(requests)

            if request._audio_segments and len(request._audio_segments) > 0:
                # Process audio with audio segments
                self.logger.info(f"Device {self.device_id}: Processing {len(request._audio_segments)} audio segments for enhanced transcription")
                
                if request.stream:
                    return self._process_segments_streaming(request)
                else:
                    return await self._process_segments_non_streaming(request)
            else:
                # Process audio without segments - direct inference on full audio
                self.logger.info(f"Device {self.device_id}: Running inference on audio data, duration: {request._duration:.2f}s, samples: {len(request._audio_array)}, stream: {request.stream}")

                result = await self._execute_pipeline(request._audio_array, request.stream, request._return_perf_metrics)

                if request.stream:
                    return self._format_streaming_result(result, request._duration, request._task_id)
                else:
                    return self._format_non_streaming_result(result, request._duration)

        except (AudioProcessingError, InferenceError, ModelNotLoadedError, DeviceInitializationError, InferenceTimeoutError):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise InferenceError(f"Inference failed: {str(e)}") from e

    def _validate_and_extract_request(self, requests: list[AudioTranscriptionRequest]) -> AudioTranscriptionRequest:
        """Validate input requests and extract the first request for processing"""
        if not requests:
            raise AudioProcessingError("Empty requests list provided")
        
        if len(requests) > 1:
            self.logger.warning(f"Device {self.device_id}: Batch processing not fully implemented. Processing only first of {len(requests)} requests")
        
        request = requests[0]
        if request is None:
            raise AudioProcessingError("Request cannot be None")

        if not hasattr(request._audio_array, 'shape'):
            raise AudioProcessingError(f"Expected numpy array with shape attribute, got {type(request._audio_array)}")

        if len(request._audio_array) == 0:
            raise AudioProcessingError("Audio data is empty")

        if not np.isfinite(request._audio_array).all():
            raise AudioProcessingError("Audio data contains non-finite values (NaN or Inf)")

        if request._duration > settings.max_audio_duration_seconds:
            self.logger.warning(f"Device {self.device_id}: Audio duration {request._duration:.2f}s exceeds recommended maximum {settings.max_audio_duration_seconds}s")

        return request

    async def _process_segments_streaming(self, request: AudioTranscriptionRequest):
        """Process segments with streaming - yields tokens immediately as they're generated"""
        segments = []
        full_text_parts = []
        speakers_set = set()
        
        for i, segment in enumerate(request._audio_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            start_sample = int(start_time * settings.default_sample_rate)
            end_sample = int(end_time * settings.default_sample_rate)
            segment_audio = request._audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                self.logger.warning(f"Device {self.device_id}: Empty audio segment {i} from {start_time:.2f}s to {end_time:.2f}s")
                continue

            self.logger.info(f"Device {self.device_id}: Processing segment {i+1}/{len(request._audio_segments)}: {start_time:.2f}s-{end_time:.2f}s, speaker: {speaker}")

            async_generator = await self._execute_pipeline(segment_audio, request.stream, request._return_perf_metrics)
            
            segment_prefix = f"[{speaker}] "
            first_token = True
            segment_text_parts = []
            chunk_count = 0
            
            async for partial_result in async_generator:
                if partial_result == "<EOS>":
                    continue
                    
                text_part = partial_result
                if request._return_perf_metrics and isinstance(partial_result, tuple):
                    text_part = partial_result[0]
                
                # Add speaker prefix to first token for streaming display
                if first_token:
                    streaming_display_text = segment_prefix + text_part
                    first_token = False
                else:
                    streaming_display_text = text_part
                
                chunk_count += 1
                
                # Yield formatted PartialStreamingTranscriptionResponse
                formatted_chunk = PartialStreamingTranscriptionResponse(
                    text=TranscriptUtils.clean_text(streaming_display_text),
                    chunk_id=chunk_count
                )
                    
                yield {
                    'type': 'streaming_chunk',
                    'chunk': formatted_chunk,
                    'segment_id': i,
                    'speaker': speaker,
                    'task_id': request._task_id
                }
                
                segment_text_parts.append(text_part)
            
            # Build segment data for final result
            segment_result = TranscriptUtils.concatenate_chunks(segment_text_parts)
            segment = TranscriptionSegment(
                id=i,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=TranscriptUtils.clean_text(segment_result)
            )
            segments.append(segment)
            full_text_parts.append(TranscriptUtils.clean_text(segment_result))
            speakers_set.add(speaker)
        
        speakers = list(speakers_set)
        
        final_result = TranscriptionResponse(
            text=TranscriptUtils.concatenate_chunks(full_text_parts),
            task=WhisperConstants.TASK_TRANSCRIBE.lower(),
            language=WhisperConstants.LANGUAGE_ENGLISH.lower(),
            duration=request._duration,
            segments=segments,
            speaker_count=len(speakers),
            speakers=speakers
        )
        
        yield {
            'type': 'final_result',
            'result': final_result,
            'task_id': request._task_id
        }

    async def _process_segments_non_streaming(self, request: AudioTranscriptionRequest):
        """Process segments without streaming - direct transcription of each segment"""
        segments = []
        full_text_parts = []
        speakers_set = set()

        for i, segment in enumerate(request._audio_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            start_sample = int(start_time * settings.default_sample_rate)
            end_sample = int(end_time * settings.default_sample_rate)
            segment_audio = request._audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                self.logger.warning(f"Device {self.device_id}: Empty audio segment {i} from {start_time:.2f}s to {end_time:.2f}s")
                continue

            self.logger.info(f"Device {self.device_id}: Processing segment {i+1}/{len(request._audio_segments)}: {start_time:.2f}s-{end_time:.2f}s, speaker: {speaker}")

            segment_result = await self._execute_pipeline(segment_audio, request.stream, request._return_perf_metrics)
            
            if request._return_perf_metrics and isinstance(segment_result, tuple):
                segment_result = segment_result[0]  # Extract text part

            if isinstance(segment_result, list) and len(segment_result) > 0:
                segment_result = segment_result[0]

            segment_result = TranscriptUtils.remove_trailing_angle_bracket(segment_result)

            segment = TranscriptionSegment(
                id=i,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=TranscriptUtils.clean_text(segment_result)
            )
            segments.append(segment)
            full_text_parts.append(TranscriptUtils.clean_text(segment_result))
            speakers_set.add(speaker)

        speakers = list(speakers_set)
        
        return [TranscriptionResponse(
            text=TranscriptUtils.concatenate_chunks(full_text_parts),
            task=WhisperConstants.TASK_TRANSCRIBE.lower(),
            language=WhisperConstants.LANGUAGE_ENGLISH.lower(),
            duration=request._duration,
            segments=segments,
            speaker_count=len(speakers),
            speakers=speakers
        )]

    async def _format_streaming_result(self, result_generator, duration, task_id):
        """Format streaming result - yield chunks immediately as they arrive"""
        streaming_chunks = []
        chunk_count = 0
        
        async for chunk in result_generator:
            if isinstance(chunk, str) and chunk != "<EOS>":
                streaming_chunks.append(chunk)
                chunk_count += 1
                
                # Yield formatted PartialStreamingTranscriptionResponse
                formatted_chunk = PartialStreamingTranscriptionResponse(
                    text=TranscriptUtils.clean_text(chunk),
                    chunk_id=chunk_count
                )
                
                yield {
                    'type': 'streaming_chunk',
                    'chunk': formatted_chunk,
                    'task_id': task_id
                }
        
        final_result = TranscriptionResponse(
            text=TranscriptUtils.concatenate_chunks(streaming_chunks),
            task=WhisperConstants.TASK_TRANSCRIBE.lower(),
            language=WhisperConstants.LANGUAGE_ENGLISH.lower(),
            duration=duration
        )
        
        yield {
            'type': 'final_result',
            'result': final_result,
            'task_id': task_id
        }

    def _format_non_streaming_result(self, result, duration):
        """Format non-streaming result"""
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        result = TranscriptUtils.remove_trailing_angle_bracket(result)
        
        final_result = TranscriptionResponse(
            text=TranscriptUtils.clean_text(result),
            task=WhisperConstants.TASK_TRANSCRIBE.lower(),
            language=WhisperConstants.LANGUAGE_ENGLISH.lower(),
            duration=duration
        )
        return [final_result]

    def _load_conditional_generation_ref_model(self):
        """Synchronous model loading - runs in thread pool"""
        try:
            model_repo = settings.model_weights_path or SupportedModels.DISTIL_WHISPER_LARGE_V3.value
            self.logger.info(f"Device {self.device_id}: Loading HuggingFace model: {model_repo}")

            hf_ref_model = WhisperForConditionalGeneration.from_pretrained(model_repo).to(torch.bfloat16).eval()
            self.logger.debug(f"Device {self.device_id}: Model loaded to bfloat16 and set to eval mode")
            processor = AutoProcessor.from_pretrained(
                model_repo, 
                language=WhisperConstants.LANGUAGE_ENGLISH, 
                task=WhisperConstants.TASK_TRANSCRIBE
            )
            self.logger.debug(f"Device {self.device_id}: Processor loaded successfully")
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_repo)
            config = hf_ref_model.config

            self.logger.info(f"Device {self.device_id}: Successfully loaded HuggingFace model components")
            return (
                hf_ref_model,
                config,
                processor,
                feature_extractor,
            )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to load HuggingFace model: {e}")
            raise WhisperModelError(f"Failed to load reference model: {str(e)}") from e
        
    async def _load_conditional_generation_ref_model_async(self):
        """Async wrapper for model loading in thread pool"""
        try:
            self.logger.info(f"Device {self.device_id}: Starting model loading in separate thread...")
            # Run the synchronous model loading in a thread pool to avoid blocking the event loop
            return await asyncio.to_thread(self._load_conditional_generation_ref_model)
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to load HuggingFace model in thread: {e}")
            raise WhisperModelError(f"Failed to load reference model: {str(e)}") from e
    
    async def _init_conditional_generation_tt_model(self, hf_ref_model, config, weights_mesh_mapper, max_seq_len=512):
        try:
            self.logger.info(f"Device {self.device_id}: Initializing TTNN model components")

            if self.ttnn_device is None:
                raise DeviceInitializationError("TTNN device not initialized")
            
            model = hf_ref_model.model
            linear_weight = hf_ref_model.proj_out.weight

            ttnn_linear_weight = ttnn.from_torch(
                linear_weight, layout=ttnn.TILE_LAYOUT, device=self.ttnn_device, dtype=ttnn.bfloat16, mesh_mapper=weights_mesh_mapper
            )
            ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
            ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)
            self.logger.info(f"Device {self.device_id}: Weights are set up")

            # Preprocess model parameters in thread pool to avoid blocking
            def _preprocess_parameters():
                # Limit threading for stability
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                
                return preprocess_model_parameters(
                    initialize_model=lambda: model,
                    convert_to_ttnn=self.ttnn_model.convert_to_ttnn,
                    custom_preprocessor=self.ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
                    device=self.ttnn_device,
                )
            parameters = await asyncio.to_thread(_preprocess_parameters)
            self.logger.info(f"Device {self.device_id}: Model parameters preprocessed")

            # Initialize KV cache in thread pool to avoid blocking
            # Note: config.max_length is typically 448 for whisper large models
            def _init_kv_cache():
                return init_kv_cache(config, self.ttnn_device, settings.max_batch_size, max_seq_len=max_seq_len, weights_mesh_mapper=weights_mesh_mapper)
            kv_cache = await asyncio.to_thread(_init_kv_cache)

            self.logger.info(f"Device {self.device_id}: Successfully initialized TTNN model components")
            return parameters, ttnn_linear_weight, kv_cache

        except DeviceInitializationError:
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to initialize TTNN model: {e}")
            raise WhisperModelError(f"TTNN model initialization failed: {str(e)}") from e
    
    def _run_generate(
        self,
        config,
        current_batch,
        feature_extractor,
        parameters,
        processor,
        ttnn_linear_weight,
        generation_config,
        input_mesh_mapper,
        output_mesh_composer,
        weights_mesh_mapper,
        kv_cache=None,
        stream_generation=False,
        return_perf_metrics=False,
    ):
        try:
            all_input_features = []
            start_encode = time.time()
            for audio_array in current_batch:
                inputs = feature_extractor(
                    audio_array,
                    sampling_rate=settings.default_sample_rate,
                    return_tensors="pt",
                )
                all_input_features.append(inputs.input_features)
            input_features = torch.cat(all_input_features, dim=0)  # [B, x, y]
            del all_input_features
            unpadded_batch_size = input_features.shape[0]

            if unpadded_batch_size != 1 * self.mesh_device.get_num_devices():
                raise AudioProcessingError(f"Only batch size (per device) 1 is supported for inference, got {unpadded_batch_size}")
            
            # Compute embeddings
            input_embeds = self.ttnn_model.preprocess_encoder_inputs(
                config,
                input_features,
                parameters=parameters.encoder,
                device=self.mesh_device,
                weights_mesh_mapper=weights_mesh_mapper,
                input_mesh_mapper=input_mesh_mapper,
            )
            # Run encoder
            encoder_hidden_states = self.ttnn_model.encoder(config, input_embeds, parameters=parameters.encoder)
            ttnn.synchronize_device(self.mesh_device)
            self.logger.info(f"Device {self.device_id}: Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

        except (AudioProcessingError, DeviceInitializationError):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed during encoding phase: {e}")
            raise InferenceError(f"Encoding failed: {str(e)}") from e

        # Run decoder
        try:
            def _run_generate():
                def pad_input_32(tensor, value):
                    len = tensor.shape[1]

                    if len % 32 == 0:
                        return tensor

                    padded_len = ((len // 32) + 1) * 32

                    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
                    tensor = torch.cat([tensor, pad_tensor], dim=1)

                    return tensor

                # Input ids
                input_ids = torch.tensor([[1]]) * config.decoder_start_token_id
                input_ids = input_ids.repeat(input_features.shape[0], 1)
                logits_processor = get_logits_processor(input_ids, config)
                if not kv_cache:
                    input_ids = pad_input_32(input_ids, config.pad_token_id).to(torch.long)
                    decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)
                # Initial decode position
                current_decode_pos = (
                    ttnn.from_torch(
                        torch.zeros(unpadded_batch_size), device=self.mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
                    )
                    if kv_cache
                    else None
                )
                MAX_GEN_LEN = config.max_length  # typically 448 for whisper large models
                print_each_iter = False
                output_ids = []
                total_decode_time = 0
                prompt_is_done = [False for _ in range(unpadded_batch_size)]

                try:
                    for i in tqdm(range(MAX_GEN_LEN), desc="Decode inference iterations"):
                        # Check timeout
                        elapsed_time = time.time() - start_encode
                        if elapsed_time > settings.default_inference_timeout_seconds:
                            raise InferenceTimeoutError(f"Inference timed out after {elapsed_time:.2f}s at decoding step {i}")

                        start_iter = time.time()
                        decoder_hidden_states, decoder_attention_mask = self.ttnn_model.preprocess_decoder_inputs(
                            config=config,
                            input_ids=input_ids,
                            attention_mask=None,
                            parameters=parameters.decoder,
                            device=self.mesh_device,
                            decode_pos=i if kv_cache else None,
                            create_attention_mask=(not kv_cache),
                            input_mesh_mapper=input_mesh_mapper,
                        )

                        output = self.ttnn_model.decoder(
                            config,
                            decoder_hidden_states,
                            decoder_attention_mask=decoder_attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            kv_cache=kv_cache,
                            current_decode_pos=current_decode_pos,
                            parameters=parameters.decoder,
                        )

                        if not kv_cache:
                            # Note: if not using a kv cache, the entire sequence is recomputed at each step
                            # Only run the lm head on the last tile to fix bad outputs and reduce redundant computation
                            last_tile_start_idx = i // 32 * 32
                            output_idx = i % 32
                            output = output[:, last_tile_start_idx : last_tile_start_idx + 32, :]
                        else:
                            output_idx = 0
                        
                        output = output @ ttnn_linear_weight
                        logits_to_torch = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
                        next_token_logits = logits_to_torch[:, output_idx, :]
                        next_tokens_scores = logits_processor(input_features, next_token_logits)
                        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                        output_ids.append(next_tokens)

                        if i == 0:
                            first_token_time = time.time()
                            ttft = first_token_time - start_encode
                        
                        # Update input_ids and current_decode_pos
                        if not kv_cache:
                            if (i + 1) % 32 == 0:
                                input_ids = torch.cat([input_ids, decoder_start_values], dim=1)
                            input_ids[:, i + 1] = next_tokens[:, None]
                        else:
                            input_ids = next_tokens[:, None]
                            ttnn.plus_one(current_decode_pos)
                        
                        total_decode_time += time.time() - start_iter
                        avg_decode_throughput = (i + 1) / total_decode_time
                        for user_id, user_decode_id in enumerate(next_tokens[:unpadded_batch_size]):
                            if user_decode_id == config.eos_token_id:
                                prompt_is_done[user_id] = True
                            if prompt_is_done[user_id]:
                                next_tokens[user_id] = config.eos_token_id
                        ttnn_transcription = processor.batch_decode(next_tokens.unsqueeze(dim=1), skip_special_tokens=True)
                        if print_each_iter:
                            self.logger.info(processor.batch_decode(torch.stack(output_ids, dim=1), skip_special_tokens=True))

                        # Convert list of strings to a single string
                        if stream_generation and isinstance(ttnn_transcription, list) and all(isinstance(t, str) for t in ttnn_transcription):
                            ttnn_transcription = "".join(ttnn_transcription)

                        if return_perf_metrics:
                            yield ttnn_transcription, ttft, avg_decode_throughput
                        else:
                            yield ttnn_transcription

                        if all(prompt_is_done):
                            break

                    # Signal end of streaming with a special marker
                    if return_perf_metrics:
                        yield "<EOS>", ttft, avg_decode_throughput
                    else:
                        yield "<EOS>"

                except InferenceTimeoutError:
                    raise
                except Exception as decode_error:
                    self.logger.error(f"Device {self.device_id}: Error during decoding iteration {i}: {decode_error}")
                    raise InferenceError(f"Decoding failed at step {i}: {str(decode_error)}") from decode_error

                total_generate_time = time.time() - start_encode
                self.logger.info(f"Device {self.device_id}: Time to first token: {(ttft*1000):.3f}ms")
                self.logger.info(f"Device {self.device_id}: Total decode time: {total_decode_time:.3f}s")
                self.logger.info(f"Device {self.device_id}: Total generate time: {total_generate_time:.3f}s")
                self.logger.info(f"Device {self.device_id}: Average decode throughput (per user): {avg_decode_throughput:.3f} t/s/u")
                self.logger.info(f"Device {self.device_id}: Average decode throughput (total batch): {(avg_decode_throughput * unpadded_batch_size):.3f} t/s")
            
            # conditionally return generator or full response
            if stream_generation:
                return _run_generate()
            else:
                output = [[] for _ in range(input_features.shape[0])]
                for x in _run_generate():
                    if return_perf_metrics:
                        out_cur, ttft, avg_decode_throughput = x
                    else:
                        out_cur = x
                    for idx in range(input_features.shape[0]):
                        output[idx].append(out_cur[idx])
                output = ["".join(tokens) for tokens in output]
                if return_perf_metrics:
                    return output, ttft, avg_decode_throughput
                else:
                    return output
        except (InferenceError, InferenceTimeoutError):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed during decoding phase: {e}")
            raise InferenceError(f"Generation failed: {str(e)}") from e

    async def _create_functional_whisper_for_conditional_generation_inference_pipeline(self):
        """
        Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
        and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
        signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
        the callable returns the full decoded output.
        """
        try:
            self.logger.info(f"Device {self.device_id}: Creating inference pipeline")

            input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(self.mesh_device)
            hf_ref_model, config, processor, feature_extractor = await self._load_conditional_generation_ref_model_async()
            parameters, ttnn_linear_weight, kv_cache = await self._init_conditional_generation_tt_model(
                hf_ref_model, config, weights_mesh_mapper
            )

            async def _model_pipeline(
                audio_data,
                stream=False,
                return_perf_metrics=False
            ):
                try:
                    # Validate pipeline inputs
                    if audio_data is None or len(audio_data) == 0:
                        raise AudioProcessingError("Audio data is empty or None")

                    if not hasattr(audio_data, 'shape'):
                        raise AudioProcessingError(f"Pipeline expected array with shape, got {type(audio_data)}")

                    if self.ttnn_device is None:
                        raise DeviceInitializationError("TTNN device not initialized")

                    # TODO: Support real batching here (currently only single-item batch)
                    current_batch = [audio_data]

                    durations = [audio_array.shape[0] / settings.default_sample_rate for audio_array in current_batch]
                    self.logger.info(
                        f"Running model on batch of {len(current_batch)} samples with durations: {['{:.3f}s'.format(d) for d in durations]}"
                    )

                    # Run inference in thread pool to avoid blocking
                    def _run_inference():
                        return self._run_generate(
                            config=config,
                            current_batch=current_batch,
                            feature_extractor=feature_extractor,
                            parameters=parameters,
                            processor=processor,
                            ttnn_linear_weight=ttnn_linear_weight,
                            generation_config=hf_ref_model.generation_config,
                            input_mesh_mapper=input_mesh_mapper,
                            output_mesh_composer=output_mesh_composer,
                            weights_mesh_mapper=weights_mesh_mapper,
                            kv_cache=kv_cache,
                            stream_generation=stream,
                            return_perf_metrics=return_perf_metrics,
                        )

                    return await asyncio.to_thread(_run_inference)
                except (AudioProcessingError, InferenceError, DeviceInitializationError, InferenceTimeoutError):
                    raise
                except Exception as e:
                    self.logger.error(f"Device {self.device_id}: Pipeline execution failed: {e}")
                    raise InferenceError(f"Pipeline execution failed: {str(e)}") from e

            self.logger.info(f"Device {self.device_id}: Successfully created inference pipeline")
            return _model_pipeline

        except (WhisperModelError, DeviceInitializationError):
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to create inference pipeline: {e}")
            raise WhisperModelError(f"Pipeline creation failed: {str(e)}") from e

