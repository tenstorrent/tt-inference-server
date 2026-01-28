# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from typing import Optional

import numpy as np
import torch
import ttnn
from config.constants import ResponseFormat, SupportedModels
from device_workers.worker_utils import setup_cpu_threading_limits
from domain.audio_processing_request import AudioProcessingRequest
from domain.audio_text_response import (
    AudioStreamChunk,
    AudioTextResponse,
    AudioTextSegment,
)
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_L1_SMALL_SIZE,
    WHISPER_TRACE_REGION_SIZE,
    convert_to_ttnn,
    create_custom_mesh_preprocessor,
    init_kv_cache,
)
from models.demos.audio.whisper.tt.whisper_generator import (
    GenerationParams,
    WhisperGenerator,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers
from telemetry.telemetry_client import TelemetryEvent
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from ttnn.model_preprocessing import preprocess_model_parameters
from utils.decorators import log_execution_time
from utils.text_utils import TextUtils


class TTWhisperRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.pipeline = None
        setup_cpu_threading_limits("1")

    def get_pipeline_device_params(self):
        device_params = {
            "l1_small_size": WHISPER_L1_SMALL_SIZE,
            "trace_region_size": WHISPER_TRACE_REGION_SIZE,
        }
        return device_params

    @log_execution_time(
        "Whisper model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        try:
            self.logger.info(f"Device {self.device_id}: Loading Whisper model...")

            # Load model components
            try:
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
                raise RuntimeError(
                    f"Device {self.device_id}: Model warmup failed: {str(e)}"
                ) from e

            return True
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Model loading failed: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Model loading failed: {str(e)}"
            ) from e

    @log_execution_time(
        "Run Whisper inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[AudioProcessingRequest]):
        """Synchronous wrapper for async inference"""
        return asyncio.run(self._run_async(requests))

    async def _run_async(self, requests: list[AudioProcessingRequest]):
        """Main inference method - validates input and routes to appropriate processing"""
        try:
            # Validate prerequisites and input
            if self.pipeline is None:
                raise RuntimeError("Model pipeline not loaded. Call warmup() first.")
            if self.ttnn_device is None:
                raise RuntimeError("TTNN device not initialized")
            request = self._validate_and_extract_request(requests)

            if request._segments and len(request._segments) > 0:
                # Process audio with audio segments
                self.logger.info(
                    f"Device {self.device_id}: Processing {len(request._segments)} audio segments, stream: {request.stream}"
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
                    return self._format_streaming_result(result, request)
                else:
                    return self._format_non_streaming_result(result, request._duration)

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    def _create_generation_params(
        self, request: AudioProcessingRequest
    ) -> GenerationParams:
        generation_params = GenerationParams()
        if request.temperatures is not None:
            generation_params.temperatures = request.temperatures
        if request.compression_ratio_threshold is not None:
            generation_params.compression_ratio_threshold = (
                request.compression_ratio_threshold
            )
        if request.logprob_threshold is not None:
            generation_params.logprob_threshold = request.logprob_threshold
        if request.no_speech_threshold is not None:
            generation_params.no_speech_threshold = request.no_speech_threshold
        if request.return_timestamps is not None:
            generation_params.return_timestamps = request.return_timestamps
        if request.prompt is not None:
            generation_params.prompt = request.prompt
        if self.settings.audio_language is not None:
            generation_params.language = self.settings.audio_language
        if self.settings.audio_task is not None:
            generation_params.task = self.settings.audio_task

        return generation_params

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

    async def _process_segments_streaming(self, request: AudioProcessingRequest):
        """Process segments with streaming - yields tokens immediately as they're generated"""
        segments = []
        final_text = ""
        speakers_set = set()
        chunk_count = 0

        for i, segment in enumerate(request._segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            # In streaming mode, we get the full audio array and need to slice it
            start_sample = int(start_time * self.settings.default_sample_rate)
            end_sample = int(end_time * self.settings.default_sample_rate)
            segment_audio = request._audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                self.logger.warning(
                    f"Device {self.device_id}: Empty audio segment {i} from {start_time:.2f}s to {end_time:.2f}s"
                )
                continue

            self.logger.info(
                f"Device {self.device_id}: Processing segment {i + 1}/{len(request._segments)}: {start_time:.2f}s-{end_time:.2f}s, speaker: {speaker}"
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
                text_part, start, end = TextUtils.extract_text(partial_result)
                # Check is_final flag
                if isinstance(partial_result, tuple) and len(partial_result) >= 4:
                    is_final = partial_result[3]
                    if is_final:
                        final_text = text_part
                        break

                # Add speaker prefix to first token for streaming display
                if first_token:
                    streaming_display_text = segment_prefix + text_part
                    first_token = False
                else:
                    streaming_display_text = text_part

                if streaming_display_text:
                    chunk_count += 1

                    formatted_chunk = AudioStreamChunk(
                        text=streaming_display_text, chunk_id=chunk_count
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
                text=segment_result,
            )
            segments.append(segment)
            speakers_set.add(speaker)

        # Sort speakers for consistent ordering
        speakers = sorted(list(speakers_set))

        final_result = AudioTextResponse(
            text=final_text,
            duration=request._duration,
            segments=segments,
            speaker_count=len(speakers),
            speakers=speakers,
            start=start,
            end=end,
        )

        yield {
            "type": "final_result",
            "result": final_result,
            "task_id": request._task_id,
            "return": request.response_format.lower() != ResponseFormat.TEXT.value,
        }

    async def _process_segments_non_streaming(self, request: AudioProcessingRequest):
        """Process segments without streaming - direct processing of each segment"""
        segments = []
        full_text_parts = []
        speakers_set = set()

        duration = 0.0

        for i, segment in enumerate(request._segments):
            start_time = segment["start"]
            end_time = segment["end"]
            duration += end_time - start_time
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            segment_audio = request._audio_array

            if len(segment_audio) == 0:
                self.logger.warning(
                    f"Device {self.device_id}: Empty audio segment {i} from {start_time:.2f}s to {end_time:.2f}s"
                )
                continue

            self.logger.info(
                f"Device {self.device_id}: Processing segment {i + 1}/{len(request._segments)}: {start_time:.2f}s-{end_time:.2f}s, speaker: {speaker}"
            )

            segment_result = await self._execute_pipeline(
                segment_audio,
                request.stream,
                self._create_generation_params(request),
            )

            cleaned_text, start, end = TextUtils.extract_text(segment_result)

            segment = AudioTextSegment(
                id=i,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=cleaned_text,
            )
            segments.append(segment)
            full_text_parts.append(cleaned_text)
            speakers_set.add(speaker)

        # Sort speakers for consistent ordering
        speakers = sorted(list(speakers_set))

        return [
            AudioTextResponse(
                text=TextUtils.concatenate_chunks(full_text_parts),
                duration=duration,
                segments=segments,
                speaker_count=len(speakers),
                speakers=speakers,
                start=start,
                end=end,
            )
        ]

    async def _format_streaming_result(
        self, result_generator, request: AudioProcessingRequest
    ):
        chunk_count = 0
        final_text = ""

        async for chunk in result_generator:
            cleaned_text, start, end = TextUtils.extract_text(chunk)

            # Check is_final flag
            if isinstance(chunk, tuple) and len(chunk) >= 4:
                is_final = chunk[3]
                if is_final:
                    final_text = cleaned_text
                    break

            # Yield non-empty chunks
            if not cleaned_text:
                continue

            chunk_count += 1
            formatted_chunk = AudioStreamChunk(text=cleaned_text, chunk_id=chunk_count)
            yield {
                "type": "streaming_chunk",
                "chunk": formatted_chunk,
                "task_id": request._task_id,
            }

        final_result = AudioTextResponse(
            text=final_text,
            duration=request._duration,
            start=start,
            end=end,
        )

        yield {
            "type": "final_result",
            "result": final_result,
            "task_id": request._task_id,
            "return": request.response_format.lower() != ResponseFormat.TEXT.value,
        }

    def load_weights(self):
        self._load_conditional_generation_ref_model()
        return True

    def _format_non_streaming_result(self, result, duration):
        text, start, end = TextUtils.extract_text(result)
        final_result = AudioTextResponse(
            text=text,
            duration=duration,
            start=start,
            end=end,
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

            kv_cache, cross_attn_cache = await asyncio.to_thread(_init_kv_cache)

            self.logger.info(
                f"Device {self.device_id}: Successfully initialized TTNN model components"
            )
            return parameters, ttnn_linear_weight, kv_cache, cross_attn_cache

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
                cross_attn_cache,
            ) = await self._init_conditional_generation_tt_model(
                hf_ref_model, config, weights_mesh_mapper
            )

            generator = WhisperGenerator(
                config=config,
                mesh_device=self.ttnn_device,
                parameters=parameters,
                processor=processor,
                feature_extractor=feature_extractor,
                ttnn_linear_weight=ttnn_linear_weight,
                generation_config=hf_ref_model.generation_config,
                input_mesh_mapper=input_mesh_mapper,
                output_mesh_composer=output_mesh_composer,
                weights_mesh_mapper=weights_mesh_mapper,
                kv_cache=kv_cache,
                cross_attn_cache=cross_attn_cache,
                max_batch_size=self.settings.max_batch_size,
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
                    def _run():
                        return generator.generate(
                            current_batch=current_batch,
                            generation_params=generation_params,
                            stream_generation=stream,
                            return_perf_metrics=False,
                        )

                    return await asyncio.to_thread(_run)
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
