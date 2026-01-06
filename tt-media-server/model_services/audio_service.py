# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.settings import settings
from domain.audio_processing_request import AudioProcessingRequest
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from telemetry.telemetry_client import TelemetryEvent
from utils.audio_manager import combine_transcription_responses
from utils.decorators import log_execution_time


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
    segments = (
        audio_manager.apply_diarization_with_vad(audio_array, perform_diarization)
        if should_preprocess
        else None
    )

    return (audio_array, duration, segments)


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
                segments,
            ) = await self._cpu_workload_handler.execute_task(
                request.file,
                request.is_preprocessing_enabled,
                request.perform_diarization,
            )

            request._audio_array = audio_array
            request._duration = duration
            request._segments = segments

            if segments:
                self.logger.info(
                    f"WhisperX preprocessing completed. Found {len(segments)} speech segments"
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

    def create_segment_request(
        self, original_request: AudioProcessingRequest, segment, segment_index: int
    ) -> AudioProcessingRequest:
        """Create a request for processing a single audio segment"""
        self.logger.debug(
            f"Audio segment {segment_index}: start={segment['start']}, "
            f"end={segment['end']}, speaker={segment.get('speaker_id', 'N/A')}"
        )

        field_values = original_request.model_dump()
        new_request = type(original_request)(**field_values)
        new_request.is_preprocessing_enabled = False  # Skip double preprocessing
        new_request._segments = [segment]  # Single segment

        # Chop audio array immediately to avoid memory leak from dragging full array
        start_sample = int(segment["start"] * settings.default_sample_rate)
        end_sample = int(segment["end"] * settings.default_sample_rate)
        new_request._audio_array = original_request._audio_array[
            start_sample:end_sample
        ]

        new_request._duration = segment["end"] - segment["start"]
        new_request.file = None  # Clear file data to save memory

        return new_request

    def combine_results(self, results):
        return combine_transcription_responses(results)

    def stop_workers(self):
        self.logger.info("Shutting down audio preprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
