# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from typing import Optional

import numpy as np
import torch
import ttnn
from config.constants import SupportedModels
from domain.audio_processing_request import AudioProcessingRequest
from domain.audio_text_response import (
    AudioTextResponse,
    AudioTextSegment,
    PartialStreamingAudioTextResponse,
)
from model_services.device_worker import setup_cpu_threading_limits
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_L1_SMALL_SIZE,
    convert_to_ttnn,
    create_custom_mesh_preprocessor,
    init_kv_cache,
)
from models.demos.whisper.tt.whisper_generator import (
    GenerationParams,
    generate,
)
from telemetry.telemetry_client import TelemetryEvent
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from tt_model_runners.base_device_runner import BaseDeviceRunner
from ttnn.model_preprocessing import preprocess_model_parameters
from utils.helpers import log_execution_time
from utils.text_utils import TextUtils


class TTWhisperRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None
        self.ttnn_model = None
        setup_cpu_threading_limits("1")

    def get_pipeline_device_params(self):
        device_params = {"l1_small_size": WHISPER_L1_SMALL_SIZE}
        return device_params

    def _create_generation_params(
        self, request: AudioProcessingRequest
    ) -> GenerationParams:
        generation_params = GenerationParams()
        if self.settings.audio_language is not None:
            generation_params.language = self.settings.audio_language
        if self.settings.audio_task is not None:
            generation_params.task = self.settings.audio_task

        return generation_params

    @log_execution_time(
        "Whisper model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def load_model(self) -> bool:
        try:
            self.logger.info(f"Device {self.device_id}: Loading Whisper model...")

            # Load model components
            try:
                self.ttnn_model = ttnn_optimized_functional_whisper
                self.pipeline = await self._create_functional_whisper_for_conditional_generation_inference_pipeline()
                self.logger.info(
                    f"Device {self.device_id}: Model pipeline created successfully"
                )
            except Exception as e:
                self.logger.error(
                    f"Device {self.device_id}: Model pipeline creation failed: {e}"
                )
                raise RuntimeError(
                    f"Device {self.device_id}: Model pipeline creation failed: {str(e)}"
                ) from e

            self.logger.info(
                f"Device {self.device_id}: Whisper model loaded and pipeline ready"
            )

            # Warmup
            try:
                dummy_audio = np.zeros(
                    self.settings.default_sample_rate, dtype=np.float32
                )
                self.logger.info(
                    f"Device {self.device_id}: Starting model warmup with {len(dummy_audio)} samples"
                )
                await self.pipeline(dummy_audio)
                self.logger.info(
                    f"Device {self.device_id}: Model warmup completed successfully"
                )
            except Exception as e:
                self.logger.error(f"Device {self.device_id}: Model warmup failed: {e}")
                self.pipeline = None
                self.ttnn_model = None
                raise RuntimeError(
                    f"Device {self.device_id}: Model warmup failed: {str(e)}"
                ) from e

            return True
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Model loading failed: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Model loading failed: {str(e)}"
            ) from e

    async def _execute_pipeline(self, audio_data, stream, generation_params):
        """Main pipeline execution method"""
        try:
            if stream:
                # Return the async generator
                return self._execute_pipeline_streaming(audio_data, generation_params)
            else:
                # Return the single result
                return await self._execute_pipeline_non_streaming(
                    audio_data, generation_params
                )

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Pipeline execution failed: {e}"
            )
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e

    async def _execute_pipeline_streaming(self, audio_data, generation_params):
        """Async generator for streaming results"""
        generator = await self.pipeline(
            audio_data,
            stream=True,
            generation_params=generation_params,
        )

        for item in generator:
            yield item

    async def _execute_pipeline_non_streaming(self, audio_data, generation_params):
        """Non-streaming pipeline execution"""
        result = await self.pipeline(
            audio_data,
            stream=False,
            generation_params=generation_params,
        )

        if result is None:
            raise RuntimeError("Pipeline returned None result")

        return result

    @log_execution_time(
        "Run Whisper inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run_inference(self, requests: list[AudioProcessingRequest]):
        """Synchronous wrapper for async inference"""
        return asyncio.run(self._run_inference_async(requests))

    async def _run_inference_async(self, requests: list[AudioProcessingRequest]):
        """Main inference method - validates input and routes to appropriate processing"""
        try:
            # Validate prerequisites and input
            if self.pipeline is None:
                raise RuntimeError(
                    "Model pipeline not loaded. Call load_model() first."
                )
            if self.ttnn_device is None:
                raise RuntimeError("TTNN device not initialized")
            request = self._validate_and_extract_request(requests)

            if request._audio_segments and len(request._audio_segments) > 0:
                # Process audio with audio segments
                self.logger.info(
                    f"Device {self.device_id}: Processing {len(request._audio_segments)} audio segments, stream: {request.stream}"
                )

                if request.stream:
                    return self._process_segments_streaming(request)
                else:
                    return await self._process_segments_non_streaming(request)
            else:
                # Process audio without segments - direct inference on full audio
                self.logger.info(
                    f"Device {self.device_id}: Running inference on audio data, duration: {request._duration:.2f}s, samples: {len(request._audio_array)}, stream: {request.stream}"
                )

                result = await self._execute_pipeline(
                    request._audio_array,
                    request.stream,
                    self._create_generation_params(request),
                )

                if request.stream:
                    return self._format_streaming_result(
                        result, request._duration, request._task_id
                    )
                else:
                    return self._format_non_streaming_result(result, request._duration)

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    def _validate_and_extract_request(
        self, requests: list[AudioProcessingRequest]
    ) -> AudioProcessingRequest:
        """Validate input requests and extract the first request for processing"""
        if not requests:
            raise ValueError("Empty requests list provided")

        if len(requests) > 1:
            self.logger.warning(
                f"Device {self.device_id}: Batch processing not fully implemented. Processing only first of {len(requests)} requests"
            )

        request = requests[0]
        if request is None:
            raise ValueError("Request cannot be None")

        if not hasattr(request._audio_array, "shape"):
            raise ValueError(
                f"Expected numpy array with shape attribute, got {type(request._audio_array)}"
            )

        if len(request._audio_array) == 0:
            raise ValueError("Audio data is empty")

        if not np.isfinite(request._audio_array).all():
            raise ValueError("Audio data contains non-finite values (NaN or Inf)")

        if request._duration > self.settings.max_audio_duration_seconds:
            self.logger.warning(
                f"Device {self.device_id}: Audio duration {request._duration:.2f}s exceeds recommended maximum {self.settings.max_audio_duration_seconds}s"
            )

        return request

    async def _process_segments_streaming(self, request: AudioProcessingRequest):
        """Process segments with streaming - yields tokens immediately as they're generated"""
        segments = []
        full_text_parts = []
        speakers_set = set()
        chunk_count = 0

        for i, segment in enumerate(request._audio_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            start_sample = int(start_time * self.settings.default_sample_rate)
            end_sample = int(end_time * self.settings.default_sample_rate)
            segment_audio = request._audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                self.logger.warning(
                    f"Device {self.device_id}: Empty audio segment {i} from {start_time:.2f}s to {end_time:.2f}s"
                )
                continue

            self.logger.info(
                f"Device {self.device_id}: Processing segment {i + 1}/{len(request._audio_segments)}: {start_time:.2f}s-{end_time:.2f}s, speaker: {speaker}"
            )

            async_generator = await self._execute_pipeline(
                segment_audio,
                request.stream,
                self._create_generation_params(request),
            )

            segment_prefix = f"[{speaker}] "
            first_token = True
            segment_text_parts = []

            async for partial_result in async_generator:
                if partial_result == "<EOS>":
                    continue

                text_part = partial_result
                if isinstance(partial_result, tuple):
                    text_part = partial_result[0]
                    if isinstance(text_part, list) and len(text_part) > 0:
                        text_part = text_part[0]

                # Add speaker prefix to first token for streaming display
                if first_token:
                    streaming_display_text = segment_prefix + text_part
                    first_token = False
                else:
                    streaming_display_text = text_part

                # Clean text and only yield non-empty chunks
                cleaned_text = TextUtils.clean_text(streaming_display_text)
                if cleaned_text:
                    chunk_count += 1

                    formatted_chunk = PartialStreamingAudioTextResponse(
                        text=cleaned_text, chunk_id=chunk_count
                    )

                    yield {
                        "type": "streaming_chunk",
                        "chunk": formatted_chunk,
                        "segment_id": i,
                        "speaker": speaker,
                        "task_id": request._task_id,
                    }

                segment_text_parts.append(text_part)

            # Build segment data for final result
            segment_result = TextUtils.concatenate_chunks(segment_text_parts)
            segment = AudioTextSegment(
                id=i,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=TextUtils.clean_text(segment_result),
            )
            segments.append(segment)
            full_text_parts.append(TextUtils.clean_text(segment_result))
            speakers_set.add(speaker)

        # Sort speakers for consistent ordering
        speakers = sorted(list(speakers_set))

        final_result = AudioTextResponse(
            text=TextUtils.concatenate_chunks(full_text_parts),
            task=self.settings.audio_task,
            language=self.settings.audio_language,
            duration=request._duration,
            segments=segments,
            speaker_count=len(speakers),
            speakers=speakers,
        )

        yield {
            "type": "final_result",
            "result": final_result,
            "task_id": request._task_id,
        }

    async def _process_segments_non_streaming(self, request: AudioProcessingRequest):
        """Process segments without streaming - direct processing of each segment"""
        segments = []
        full_text_parts = []
        speakers_set = set()

        duration = 0.0

        for i, segment in enumerate(request._audio_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            duration += end_time - start_time
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            start_sample = int(start_time * self.settings.default_sample_rate)
            end_sample = int(end_time * self.settings.default_sample_rate)
            segment_audio = request._audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                self.logger.warning(
                    f"Device {self.device_id}: Empty audio segment {i} from {start_time:.2f}s to {end_time:.2f}s"
                )
                continue

            self.logger.info(
                f"Device {self.device_id}: Processing segment {i + 1}/{len(request._audio_segments)}: {start_time:.2f}s-{end_time:.2f}s, speaker: {speaker}"
            )

            segment_result = await self._execute_pipeline(
                segment_audio,
                request.stream,
                self._create_generation_params(request),
            )

            if isinstance(segment_result, list) and len(segment_result) > 0:
                segment_result = segment_result[0]

            segment_result = TextUtils.remove_trailing_angle_bracket(segment_result)

            segment = AudioTextSegment(
                id=i,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=TextUtils.clean_text(segment_result),
            )
            segments.append(segment)
            full_text_parts.append(TextUtils.clean_text(segment_result))
            speakers_set.add(speaker)

        # Sort speakers for consistent ordering
        speakers = sorted(list(speakers_set))

        return [
            AudioTextResponse(
                text=TextUtils.concatenate_chunks(full_text_parts),
                task=self.settings.audio_task,
                language=self.settings.audio_language,
                duration=duration,
                segments=segments,
                speaker_count=len(speakers),
                speakers=speakers,
            )
        ]

    async def _format_streaming_result(self, result_generator, duration, task_id):
        """Format streaming result - yield chunks immediately as they arrive"""
        streaming_chunks = []
        chunk_count = 0

        async for chunk in result_generator:
            text_chunk = chunk
            if isinstance(chunk, tuple):
                text_chunk = chunk[0]
                if isinstance(text_chunk, list) and len(text_chunk) > 0:
                    text_chunk = text_chunk[0]

            if isinstance(text_chunk, str) and text_chunk != "<EOS>":
                # Clean text and only yield non-empty chunks
                cleaned_text = TextUtils.clean_text(text_chunk)
                if cleaned_text:
                    streaming_chunks.append(text_chunk)
                    chunk_count += 1

                    formatted_chunk = PartialStreamingAudioTextResponse(
                        text=cleaned_text, chunk_id=chunk_count
                    )

                    yield {
                        "type": "streaming_chunk",
                        "chunk": formatted_chunk,
                        "task_id": task_id,
                    }

        final_result = AudioTextResponse(
            text=TextUtils.concatenate_chunks(streaming_chunks),
            task=self.settings.audio_task,
            language=self.settings.audio_language,
            duration=duration,
        )

        yield {"type": "final_result", "result": final_result, "task_id": task_id}

    def _format_non_streaming_result(self, result, duration):
        """Format non-streaming result"""
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        result = TextUtils.remove_trailing_angle_bracket(result)

        final_result = AudioTextResponse(
            text=TextUtils.clean_text(result),
            task=self.settings.audio_task,
            language=self.settings.audio_language,
            duration=duration,
        )
        return [final_result]

    def _load_conditional_generation_ref_model(self):
        """Synchronous model loading - runs in thread pool"""
        try:
            model_weights_path = (
                self.settings.model_weights_path
                or SupportedModels.DISTIL_WHISPER_LARGE_V3.value
            )
            self.logger.info(
                f"Device {self.device_id}: Loading HuggingFace model: {model_weights_path}"
            )

            hf_ref_model = (
                WhisperForConditionalGeneration.from_pretrained(model_weights_path)
                .to(torch.bfloat16)
                .eval()
            )
            self.logger.debug(
                f"Device {self.device_id}: Model loaded to bfloat16 and set to eval mode"
            )
            processor = AutoProcessor.from_pretrained(
                model_weights_path,
                task=self.settings.audio_task,
                language=self.settings.audio_language,
            )
            self.logger.debug(f"Device {self.device_id}: Processor loaded successfully")
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_weights_path)
            config = hf_ref_model.config

            self.logger.info(
                f"Device {self.device_id}: Successfully loaded HuggingFace model components"
            )
            return (
                hf_ref_model,
                config,
                processor,
                feature_extractor,
            )
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to load HuggingFace model: {e}"
            )
            raise RuntimeError(f"Failed to load reference model: {str(e)}") from e

    async def _load_conditional_generation_ref_model_async(self):
        """Async wrapper for model loading in thread pool"""
        try:
            self.logger.info(
                f"Device {self.device_id}: Starting model loading in separate thread..."
            )
            # Run the synchronous model loading in a thread pool to avoid blocking the event loop
            return await asyncio.to_thread(self._load_conditional_generation_ref_model)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to load HuggingFace model in thread: {e}"
            )
            raise RuntimeError(f"Failed to load reference model: {str(e)}") from e

    async def _init_conditional_generation_tt_model(
        self, hf_ref_model, config, weights_mesh_mapper, max_seq_len=512
    ):
        try:
            self.logger.info(
                f"Device {self.device_id}: Initializing TTNN model components"
            )

            if self.ttnn_device is None:
                raise RuntimeError("TTNN device not initialized")

            model = hf_ref_model.model
            linear_weight = hf_ref_model.proj_out.weight

            ttnn_linear_weight = ttnn.from_torch(
                linear_weight,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                dtype=ttnn.bfloat16,
                mesh_mapper=weights_mesh_mapper,
            )
            ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
            ttnn_linear_weight = ttnn.to_layout(
                ttnn_linear_weight, layout=ttnn.TILE_LAYOUT
            )
            self.logger.info(f"Device {self.device_id}: Weights are set up")

            # Preprocess model parameters in thread pool to avoid blocking
            def _preprocess_parameters():
                setup_cpu_threading_limits("1")

                return preprocess_model_parameters(
                    initialize_model=lambda: model,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=create_custom_mesh_preprocessor(
                        weights_mesh_mapper
                    ),
                    device=self.ttnn_device,
                )

            parameters = await asyncio.to_thread(_preprocess_parameters)
            self.logger.info(f"Device {self.device_id}: Model parameters preprocessed")

            # Initialize KV cache in thread pool to avoid blocking
            # Note: config.max_length is typically 448 for whisper large models
            def _init_kv_cache():
                return init_kv_cache(
                    config,
                    self.ttnn_device,
                    self.settings.max_batch_size,
                    max_seq_len=max_seq_len,
                    weights_mesh_mapper=weights_mesh_mapper,
                )

            kv_cache = await asyncio.to_thread(_init_kv_cache)

            self.logger.info(
                f"Device {self.device_id}: Successfully initialized TTNN model components"
            )
            return parameters, ttnn_linear_weight, kv_cache

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to initialize TTNN model: {e}"
            )
            raise RuntimeError(f"TTNN model initialization failed: {str(e)}") from e

    async def _create_functional_whisper_for_conditional_generation_inference_pipeline(
        self,
    ):
        """
        Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
        and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
        signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
        the callable returns the full decoded output.
        """
        try:
            self.logger.info(f"Device {self.device_id}: Creating inference pipeline")

            input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = (
                get_mesh_mappers(self.ttnn_device)
            )
            (
                hf_ref_model,
                config,
                processor,
                feature_extractor,
            ) = await self._load_conditional_generation_ref_model_async()
            (
                parameters,
                ttnn_linear_weight,
                kv_cache,
            ) = await self._init_conditional_generation_tt_model(
                hf_ref_model, config, weights_mesh_mapper
            )

            async def _model_pipeline(
                audio_data,
                stream=False,
                generation_params: Optional[GenerationParams] = None,
            ):
                try:
                    # Validate pipeline inputs
                    if audio_data is None or len(audio_data) == 0:
                        raise ValueError("Audio data is empty or None")

                    if not hasattr(audio_data, "shape"):
                        raise ValueError(
                            f"Pipeline expected array with shape, got {type(audio_data)}"
                        )

                    if self.ttnn_device is None:
                        raise RuntimeError("TTNN device not initialized")

                    # TODO: Support real batching here (currently only single-item batch)
                    # Format as (sampling_rate, audio_array) tuples as expected by generate()
                    current_batch = [(self.settings.default_sample_rate, audio_data)]

                    durations = [
                        audio_array.shape[0] / sampling_rate
                        for sampling_rate, audio_array in current_batch
                    ]
                    self.logger.info(
                        f"Running model on batch of {len(current_batch)} samples with durations: {['{:.3f}s'.format(d) for d in durations]}"
                    )

                    # Run inference in thread pool to avoid blocking
                    def _run_inference():
                        return generate(
                            config,
                            self.ttnn_device,
                            (input_mesh_mapper, weights_mesh_mapper),
                            current_batch,
                            feature_extractor,
                            parameters=parameters,
                            processor=processor,
                            ttnn_linear_weight=ttnn_linear_weight,
                            mesh_device=self.ttnn_device,
                            generation_config=hf_ref_model.generation_config,
                            input_mesh_mapper=input_mesh_mapper,
                            output_mesh_composer=output_mesh_composer,
                            weights_mesh_mapper=weights_mesh_mapper,
                            kv_cache=kv_cache,
                            generation_params=generation_params,
                            stream_generation=stream,
                        )

                    return await asyncio.to_thread(_run_inference)
                except Exception as e:
                    self.logger.error(
                        f"Device {self.device_id}: Pipeline execution failed: {e}"
                    )
                    raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e

            self.logger.info(
                f"Device {self.device_id}: Successfully created inference pipeline"
            )
            return _model_pipeline

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to create inference pipeline: {e}"
            )
            raise RuntimeError(f"Pipeline creation failed: {str(e)}") from e
