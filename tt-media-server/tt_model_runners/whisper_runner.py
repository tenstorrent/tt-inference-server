# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import os
import time
from typing import List, Optional, Union

import numpy as np
import torch
import ttnn
from config.constants import ResponseFormat, SupportedModels
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
from telemetry.audio_metrics import (
    confidence_from_generator_output,
    record_stt_confidence,
    transcript_compression_ratio,
)
from telemetry.telemetry_client import (
    TelemetryEvent,
    audio_chunk_first_token_duration,
    audio_chunk_processing_duration,
    audio_encoder_duration,
    audio_encoder_input_seconds,
    audio_feature_extraction_duration,
    audio_feature_extraction_input_seconds,
)
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from ttnn.model_preprocessing import preprocess_model_parameters
from utils.decorators import log_execution_time
from utils.text_utils import TextUtils

# two-command-queue (2CQ) trace execution path
WHISPER_NUM_COMMAND_QUEUES = 2

# Fallback frame length when the extractor does not advertise one.
WHISPER_ENCODER_FRAME_SECONDS = 30.0

try:
    # tt-metal #53717 packs the stage timings into PerfMetrics. Probed, not
    # assumed: without it, asking for perf metrics only changes the tuple arity.
    from models.demos.audio.whisper.tt.whisper_generator import (  # noqa: F401
        PerfMetrics,
    )

    WHISPER_PERF_METRICS_SUPPORTED = True
except ImportError:
    WHISPER_PERF_METRICS_SUPPORTED = False


class TTWhisperRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None
        # Replaced with the loaded extractor's real frame length once the
        # pipeline is created.
        self._encoder_frame_seconds = WHISPER_ENCODER_FRAME_SECONDS

    def get_pipeline_device_params(self):
        device_params = {
            "l1_small_size": WHISPER_L1_SMALL_SIZE,
            "trace_region_size": WHISPER_TRACE_REGION_SIZE,
            "num_command_queues": WHISPER_NUM_COMMAND_QUEUES,
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
                # Excluded from the stage metrics: these are cold, they capture
                # the encoder traces, and their 1s of silence is not offered
                # load. The encoder side would be filtered out anyway by
                # trace_hit="false", but feature extraction has no such label,
                # so warmup would otherwise sit in its panels permanently.
                await self.pipeline(dummy_audio, record_stage_metrics=False)
                if self.settings.max_batch_size > 1:
                    warmup_batch = [dummy_audio, dummy_audio]
                    await self.pipeline(warmup_batch, record_stage_metrics=False)
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

            if requests.__len__() > 1 or self.settings.max_batch_size > 1:
                return await self._execute_pipeline_batch(
                    requests, [self._create_generation_params(req) for req in requests]
                )

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
                    prompt=request.prompt,
                    audio_profile=self._audio_profile(request),
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

    async def _execute_pipeline(
        self, audio_data, stream, generation_params, prompt=None, audio_profile=None
    ):
        """Main pipeline execution method"""
        try:
            if stream:
                # Return the async generator
                return self._execute_pipeline_streaming(
                    audio_data, generation_params, prompt, audio_profile
                )
            else:
                # Return the single result
                return await self._execute_pipeline_non_streaming(
                    audio_data, generation_params, prompt, audio_profile
                )

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Pipeline execution failed: {e}"
            )
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e

    async def _execute_pipeline_batch(self, requests, generation_params):
        """Main pipeline execution method for batch processing.

        Supports 1 or 2 requests. When 2 requests are provided, audio arrays
        are padded to the same tensor size. Generator handles both batch sizes
        natively using separate pre-warmed traces (trace_key=1 and trace_key=2).
        """
        try:
            audio_arrays = [req._audio_array for req in requests]
            durations = [req._duration for req in requests]

            if len(audio_arrays) > 2:
                raise ValueError(f"Expected 1 or 2 requests, got {len(audio_arrays)}")

            if len(audio_arrays) == 2:
                max_len = max(len(audio_arrays[0]), len(audio_arrays[1]))
                audio_data = [
                    np.pad(arr, (0, max_len - len(arr))) if len(arr) < max_len else arr
                    for arr in audio_arrays
                ]
            else:
                audio_data = audio_arrays

            # Measured before the pad above, which is silence: throughput has to
            # be credited the audio that was actually submitted, or a batch of
            # unequal lengths reports the longer item's duration twice.
            audio_durations = [
                len(arr) / self.settings.default_sample_rate for arr in audio_arrays
            ]

            # prompt is batch-homogeneous in tt-metal's WhisperGenerator: take the
            # first request's prompt and warn if requests disagree.
            prompts = {req.prompt for req in requests if req.prompt is not None}
            if len(prompts) > 1:
                self.logger.warning(
                    f"Device {self.device_id}: Batch contains differing prompts; using only the first request's prompt."
                )
            prompt = requests[0].prompt if requests else None

            result = await self.pipeline(
                audio_data,
                stream=False,
                generation_params=generation_params,
                prompt=prompt,
                audio_durations=audio_durations,
                audio_profile=self._audio_profile(requests),
            )

            responses = []
            if result and result[0]:
                for index, text in enumerate(result[0]):
                    cleaned_text, start, end = TextUtils.extract_text(text)
                    responses.append(
                        AudioTextResponse(
                            text=cleaned_text,
                            duration=durations[index],
                            start=start,
                            end=end,
                            segments=[],
                        )
                    )
                # The generator returns batch-level signal tensors, so this
                # records their mean once, with the batch's joined text.
                self._record_confidence_signals(
                    result,
                    text=" ".join(response.text for response in responses),
                )

            return responses

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Pipeline execution failed: {e}"
            )
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e

    @staticmethod
    def _resolve_encoder_frame_seconds(feature_extractor):
        """Length of the fixed frame every batch item is padded or truncated to.

        Whisper's extractor pads to `n_samples` (30s at 16kHz), so this caps the
        audio either stage can consume per item whatever was submitted.
        """
        n_samples = getattr(feature_extractor, "n_samples", None)
        sampling_rate = getattr(feature_extractor, "sampling_rate", None)
        if (
            isinstance(n_samples, (int, float))
            and isinstance(sampling_rate, (int, float))
            and sampling_rate
        ):
            return n_samples / sampling_rate
        chunk_length = getattr(feature_extractor, "chunk_length", None)
        if isinstance(chunk_length, (int, float)) and chunk_length:
            return float(chunk_length)
        return WHISPER_ENCODER_FRAME_SECONDS

    @staticmethod
    def _audio_profile(requests):
        """The submitted operating point shared by a batch of requests.

        `sample_rate` and `channels` describe what arrived, not what reaches the
        extractor: audio_manager downmixes to mono and resamples to the default
        rate before the runner sees an array, so deriving these from the array
        would report a constant. "unknown" covers a decode path that normalised
        them away (ffmpeg) and "mixed" a batch whose items disagree, so neither
        case is silently reported as a real value.
        """
        if not isinstance(requests, (list, tuple)):
            requests = [requests]

        def _shared(attr):
            values = {getattr(req, attr, None) for req in requests}
            if len(values) != 1:
                return "mixed"
            value = values.pop()
            return "unknown" if value is None else str(value)

        return {
            "sample_rate": _shared("_source_sample_rate"),
            "channels": _shared("_source_channels"),
        }

    def _audio_stage_context(
        self, current_batch, audio_durations=None, audio_profile=None
    ):
        """Label values and the useful audio seconds credited to the two stages.

        Counted here rather than read off PerfMetrics.total_audio_s, which sums
        submitted durations: an item longer than one frame is truncated to it.
        Reachable here — chunking never splits a single long VAD segment, so
        uninterrupted speech becomes one long chunk.

        `audio_durations` overrides the duration derived from each array's
        length. Batched calls zero-pad the shorter item up to the longer one
        before submitting, and that pad is silence no stage turns into useful
        features — measuring the submitted array would credit it as real audio.
        """
        frame_seconds = self._encoder_frame_seconds
        audio_seconds = 0.0
        for index, (rate, audio_array) in enumerate(current_batch):
            if rate:
                if audio_durations is not None and index < len(audio_durations):
                    duration = audio_durations[index]
                else:
                    duration = audio_array.shape[0] / rate
                audio_seconds += min(duration, frame_seconds)

        profile = audio_profile or {}
        sample_rate = profile.get("sample_rate") or str(
            self.settings.default_sample_rate
        )
        channels = profile.get("channels") or "1"

        model_name = (
            os.path.basename((self.settings.model_weights_path or "").rstrip("/"))
            or "unknown"
        )
        device_id = str(self.device_id) if self.device_id is not None else "unknown"
        return {
            "audio_seconds": audio_seconds,
            "feature_labels": {
                "model_type": self.settings.model_runner,
                "device_id": device_id,
                "sample_rate": sample_rate,
                "channels": channels,
                "batch": str(len(current_batch)),
            },
            "encoder_labels": {
                "model_type": self.settings.model_runner,
                "device_id": device_id,
                "model_name": model_name,
                "language": self.settings.audio_language or "unknown",
                "batch": str(len(current_batch)),
            },
        }

    @staticmethod
    def _find_perf_metrics(result):
        """Pull the PerfMetrics out of a returned or yielded tuple.

        Matched on shape, not type, so its position in the tuple can move.
        """
        if not isinstance(result, tuple):
            return None
        for item in result:
            if hasattr(item, "feature_extract_s") and hasattr(item, "encoder_s"):
                return item
        return None

    def _record_audio_stage_throughput(self, result, context):
        """Record feature-extraction and encoder throughput for one batch.

        Returns whether anything was recorded, so the streaming wrapper knows
        when to stop looking. Each stage is gated on a positive duration:
        generate()'s no-valid-output paths return zeroed timings, and crediting
        audio against a zero-second stage reports unbounded throughput.
        """
        perf = self._find_perf_metrics(result)
        if perf is None:
            return False

        audio_seconds = context["audio_seconds"]
        recorded = False

        feature_extract_s = getattr(perf, "feature_extract_s", 0.0) or 0.0
        if feature_extract_s > 0:
            labels = context["feature_labels"]
            audio_feature_extraction_input_seconds.labels(**labels).inc(audio_seconds)
            audio_feature_extraction_duration.labels(**labels).observe(
                feature_extract_s
            )
            recorded = True

        encoder_s = getattr(perf, "encoder_s", 0.0) or 0.0
        if encoder_s > 0:
            # The capture call runs the encoder twice; labelled, not dropped,
            # so the cost stays visible.
            trace_hit = bool(getattr(perf, "encoder_trace_hit", True))
            labels = dict(
                context["encoder_labels"], trace_hit="true" if trace_hit else "false"
            )
            audio_encoder_input_seconds.labels(**labels).inc(audio_seconds)
            audio_encoder_duration.labels(**labels).observe(encoder_s)
            recorded = True

        return recorded

    def _stream_with_stage_metrics(self, generator, context):
        """Record the stage timings once, off the first yield that carries them.

        Both stages run once per generate() call, ahead of the decode loop, and
        every yield repeats the same timings — recording per item would multiply
        one batch by its token count.
        """
        recorded = False
        for item in generator:
            if not recorded:
                recorded = self._record_audio_stage_throughput(item, context)
            yield item

    def _record_confidence_signals(self, item, text=None):
        """Export Whisper's own quality signals for one generation.

        WER proxies: true WER needs reference transcripts (offline evals in
        test_module/eval_tests/ are the source of truth); these histograms
        are the live drift detectors. ``item`` is a raw generator tuple —
        avg_logprob at index 1, no_speech_prob at index 2 on both the
        streaming final marker and the non-streaming return — and the
        compression ratio is recomputed here from the transcript with the
        generator's own formula, since the generator does not return it.
        """
        avg_logprob, no_speech_prob = confidence_from_generator_output(item)
        record_stt_confidence(
            model_type=self.settings.model_runner,
            language=self.settings.audio_language or "unknown",
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=transcript_compression_ratio(text),
        )

    @staticmethod
    def _is_final_result(item):
        """Whether a streamed item is the final marker for its chunk.

        The flag is always last but its index shifts with return_perf_metrics,
        so key off the trailing bool. Non-streaming tuples end in a tensor or a
        PerfMetrics and never match.
        """
        return (
            isinstance(item, tuple)
            and len(item) >= 4
            and isinstance(item[-1], bool)
            and item[-1]
        )

    async def _execute_pipeline_streaming(
        self, audio_data, generation_params, prompt=None, audio_profile=None
    ):
        """Async generator for streaming results"""
        generator = await self.pipeline(
            audio_data,
            stream=True,
            generation_params=generation_params,
            prompt=prompt,
            audio_profile=audio_profile,
        )

        for item in generator:
            yield item

    async def _execute_pipeline_non_streaming(
        self, audio_data, generation_params, prompt=None, audio_profile=None
    ):
        """Non-streaming pipeline execution"""
        result = await self.pipeline(
            audio_data,
            stream=False,
            generation_params=generation_params,
            prompt=prompt,
            audio_profile=audio_profile,
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

            # `_execute_pipeline` only builds the async generator; the log-mel
            # extraction and the device passes are deferred to the first pull
            # below, so the clock has to start here and stop inside the loop.
            chunk_start = time.perf_counter()
            async_generator = await self._execute_pipeline(
                segment_audio,
                request.stream,
                self._create_generation_params(request),
                prompt=request.prompt,
                audio_profile=self._audio_profile(request),
            )

            segment_prefix = f"[{speaker}] "
            first_token = True
            first_item_observed = False
            segment_text_parts = []

            async for partial_result in async_generator:
                # Ahead of the is_final check: a chunk whose only item is the
                # final marker still paid the extraction and encode cost.
                if not first_item_observed:
                    audio_chunk_first_token_duration.labels(
                        model_type=self.settings.model_runner
                    ).observe(time.perf_counter() - chunk_start)
                    first_item_observed = True

                text_part, start, end = TextUtils.extract_text(partial_result)
                if self._is_final_result(partial_result):
                    final_text = text_part
                    self._record_confidence_signals(partial_result, text=text_part)
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

            # Reached on both generator exhaustion and the is_final `break`. The
            # span covers the `yield`s above, so it is wall time per chunk and
            # includes any consumer backpressure, not compute alone.
            audio_chunk_processing_duration.labels(
                model_type=self.settings.model_runner
            ).observe(time.perf_counter() - chunk_start)

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
                prompt=request.prompt,
                audio_profile=self._audio_profile(request),
            )

            cleaned_text, start, end = TextUtils.extract_text(segment_result)
            self._record_confidence_signals(segment_result, text=cleaned_text)

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

            if self._is_final_result(chunk):
                final_text = cleaned_text
                self._record_confidence_signals(chunk, text=final_text)
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
        self._record_confidence_signals(result, text=text)
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
                    max_seq_len=max_seq_len,
                    weights_mesh_mapper=weights_mesh_mapper,
                )

            (
                kv_cache_per_batch_size,
                cross_attn_cache_per_batch_size,
            ) = await asyncio.to_thread(_init_kv_cache)

            self.logger.info(
                f"Device {self.device_id}: Successfully initialized TTNN model components"
            )
            return (
                parameters,
                ttnn_linear_weight,
                kv_cache_per_batch_size,
                cross_attn_cache_per_batch_size,
            )

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
                kv_cache_per_batch_size,
                cross_attn_cache_per_batch_size,
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
                kv_cache_per_batch_size=kv_cache_per_batch_size,
                cross_attn_cache_per_batch_size=cross_attn_cache_per_batch_size,
                max_batch_size=self.settings.max_batch_size,
            )

            self._encoder_frame_seconds = self._resolve_encoder_frame_seconds(
                feature_extractor
            )

            async def _model_pipeline(
                audio_data,
                stream=False,
                generation_params: Optional[
                    Union[GenerationParams, List[GenerationParams]]
                ] = None,
                prompt: Optional[str] = None,
                audio_durations: Optional[List[float]] = None,
                record_stage_metrics: bool = True,
                audio_profile: Optional[dict] = None,
            ):
                try:
                    # Validate pipeline inputs
                    if audio_data is None:
                        raise ValueError("Audio data is empty or None")

                    if self.ttnn_device is None:
                        raise RuntimeError("TTNN device not initialized")

                    # Handle both single audio array and batch (list of arrays)
                    if isinstance(audio_data, list):
                        # Batch mode: list of audio arrays
                        current_batch = [
                            (self.settings.default_sample_rate, arr)
                            for arr in audio_data
                        ]
                    else:
                        # Single audio mode
                        if not hasattr(audio_data, "shape"):
                            raise ValueError(
                                f"Pipeline expected array with shape, got {type(audio_data)}"
                            )
                        if len(audio_data) == 0:
                            raise ValueError("Audio data is empty")
                        current_batch = [
                            (self.settings.default_sample_rate, audio_data)
                        ]

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
                            language=self.settings.audio_language,
                            task=self.settings.audio_task,
                            prompt=prompt or None,
                            stream_generation=stream,
                            return_perf_metrics=WHISPER_PERF_METRICS_SUPPORTED,
                        )

                    result = await asyncio.to_thread(_run)

                    if not WHISPER_PERF_METRICS_SUPPORTED or not record_stage_metrics:
                        return result

                    # Built here: only this scope knows what was submitted.
                    stage_context = self._audio_stage_context(
                        current_batch,
                        audio_durations=audio_durations,
                        audio_profile=audio_profile,
                    )
                    if stream:
                        # Streaming defers every stage to the first pull, so the
                        # timings do not exist yet.
                        return self._stream_with_stage_metrics(result, stage_context)

                    self._record_audio_stage_throughput(result, stage_context)
                    return result
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
