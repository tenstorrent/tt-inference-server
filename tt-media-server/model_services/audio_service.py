# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from typing import List

from config.settings import settings
from domain.audio_processing_request import AudioProcessingRequest
from domain.audio_text_response import AudioTextResponse, AudioTextSegment
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from telemetry.telemetry_client import TelemetryEvent
from utils.helpers import log_execution_time


def create_audio_worker_context():
    from utils.audio_manager import AudioManager

    return AudioManager()


def audio_worker_function(
    audio_manager, audio_file_data, is_preprocessing_enabled, perform_diarization=False
):
    """Process audio data using the initialized AudioManager"""
    from config.settings import settings

    should_preprocess = settings.allow_audio_preprocessing and is_preprocessing_enabled

    # Process audio
    audio_array, duration = audio_manager.to_audio_array(
        audio_file_data, should_preprocess
    )
    audio_segments = (
        audio_manager.apply_diarization_with_vad(audio_array, perform_diarization)
        if should_preprocess
        else None
    )

    return (audio_array, duration, audio_segments)


class AudioService(BaseService):
    def __init__(self):
        super().__init__()

        from static.data.audio import DUMMY_WAV_BASE64

        warmup_task_data = (DUMMY_WAV_BASE64, True)
        self._cpu_workload_handler = CpuWorkloadHandler(
            name="AudioPreprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=audio_worker_function,
            worker_context_setup=create_audio_worker_context,
            warmup_task_data=warmup_task_data,
        )

    @log_execution_time("Audio preprocessing", TelemetryEvent.PRE_PROCESSING, None)
    async def pre_process(self, request: AudioProcessingRequest):
        """Asynchronous preprocessing using queue-based workers"""
        try:
            if request.file is None:
                raise ValueError("No audio data provided")

            (
                audio_array,
                duration,
                audio_segments,
            ) = await self._cpu_workload_handler.execute_task(
                request.file,
                request.is_preprocessing_enabled,
                request.perform_diarization,
            )

            request._audio_array = audio_array
            request._duration = duration
            request._audio_segments = audio_segments

            if audio_segments:
                self.logger.info(
                    f"WhisperX preprocessing completed. Found {len(audio_segments)} speech segments"
                )
            else:
                if not settings.allow_audio_preprocessing:
                    self.logger.info(
                        "WhisperX preprocessing not allowed, skipping VAD and diarization"
                    )
                elif not request.is_preprocessing_enabled:
                    self.logger.info(
                        "WhisperX preprocessing disabled for this request, skipping VAD and diarization"
                    )
                else:
                    self.logger.info("WhisperX preprocessing skipped")

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise

        return request

    @log_execution_time("Process audio request", TelemetryEvent.TOTAL_PROCESSING, None)
    async def process_request(self, request: AudioProcessingRequest):
        request = await self.pre_process(request)

        # If no audio segments, process the entire audio as one segment
        if not request._audio_segments:
            return await super().process_request(request)

        # Create individual requests maintaining the temporal order
        individual_requests = []
        for i, audio_segment in enumerate(request._audio_segments):
            self.logger.debug(
                f"Audio segment {i}: start={audio_segment['start']}, end={audio_segment['end']}, speaker={audio_segment.get('speaker_id', 'N/A')}"
            )
            field_values = request.model_dump()
            new_request = type(request)(**field_values)
            new_request.is_preprocessing_enabled = False  # Skip double preprocessing
            new_request._audio_segments = [audio_segment]  # Single segment
            new_request._audio_array = (
                request._audio_array
            )  # Keep audio array for processing
            new_request.file = None  # Clear file data to save memory
            individual_requests.append(new_request)

        # Create tasks maintaining order - asyncio.gather preserves order
        tasks = []
        for req in individual_requests:
            tasks.append(super().process(req))

        # Gather results in order (asyncio.gather maintains the order of inputs)
        results = await asyncio.gather(*tasks)

        # Combine all AudioTextResponse objects into one (preserving order)
        combined_response = self._combine_transcription_responses(results)
        return combined_response

    def _combine_transcription_responses(
        self, responses: List[AudioTextResponse]
    ) -> AudioTextResponse:
        """Combine multiple AudioTextResponse objects into a single response.

        Args:
            responses: List of AudioTextResponse objects to combine

        Returns:
            AudioTextResponse: Combined response with summed duration and merged content
        """
        if not responses:
            # Return empty response if no responses provided
            raise ValueError("No transcription responses to combine")

        if len(responses) == 1:
            # Return single response as-is
            return responses[0]

        # Combine text from all responses
        combined_text = " ".join(
            response.text.strip() for response in responses if response.text.strip()
        )

        # Sum up all durations
        total_duration = sum(response.duration for response in responses)

        # Use first response's task and language as defaults
        first_response = responses[0]
        combined_task = first_response.task
        combined_language = first_response.language

        # Combine segments if available
        combined_segments = []
        segment_id_counter = 1
        all_speakers = set()

        for response in responses:
            if response.segments:
                for segment in response.segments:
                    # Create new segment with updated ID to maintain sequence
                    combined_segment = AudioTextSegment(
                        id=segment_id_counter,
                        speaker=segment.speaker,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=segment.text,
                    )
                    combined_segments.append(combined_segment)
                    all_speakers.add(segment.speaker)
                    segment_id_counter += 1

            # Also collect speakers from response-level speaker info
            if response.speakers:
                all_speakers.update(response.speakers)

        # Combine speaker information
        combined_speakers = sorted(list(all_speakers)) if all_speakers else None
        combined_speaker_count = len(all_speakers) if all_speakers else None

        # Create combined response
        combined_response = AudioTextResponse(
            text=combined_text,
            task=combined_task,
            language=combined_language,
            duration=total_duration,
            segments=combined_segments if combined_segments else None,
            speaker_count=combined_speaker_count,
            speakers=combined_speakers,
        )

        self.logger.info(
            f"Combined {len(responses)} transcription responses into one: "
            f"total_duration={total_duration:.2f}s, "
            f"total_segments={len(combined_segments)}, "
            f"speaker_count={combined_speaker_count}"
        )

        return combined_response

    def stop_workers(self):
        self.logger.info("Shutting down audio preprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
