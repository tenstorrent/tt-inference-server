# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
    audio_manager,
    audio_file_data,
    is_preprocessing_enabled,
    perform_diarization=False,
    chunk_duration_seconds=None,
):
    """Process audio data using the initialized AudioManager"""
    from config.settings import settings

    from config.constants import ModelRunners

    should_preprocess = settings.allow_audio_preprocessing and is_preprocessing_enabled

    # Process audio
    prepared = audio_manager.to_audio_array(audio_file_data, should_preprocess)
    audio_array = prepared.audio_array

    # Qwen3-ASR-only duration fan-out: split a long clip into contiguous windows so
    # it occupies many device runners (linear RTR speedup on Galaxy DP=32). Short
    # clips already fit one runner, so splitting them only adds boundary errors
    # (measured +2.4 WER at 3s on librispeech) -> keep them whole. This path is
    # deliberately NOT taken for Whisper/other ASR: their VAD chunking below is
    # left exactly as in production. Diarization requests also use the shared path.
    is_qwen3_asr = settings.model_runner == ModelRunners.TT_QWEN3_ASR.value
    if is_qwen3_asr and not perform_diarization:
        requested = chunk_duration_seconds or settings.audio_chunk_duration_seconds
        if prepared.duration > settings.audio_min_split_duration_seconds:
            segments = audio_manager.chunk_audio_by_duration(
                prepared.duration, requested
            )
        else:
            # Keep whole: one runner transcribes the clip (validated WER 2.63%).
            segments = None
    elif should_preprocess:
        segments = audio_manager.apply_diarization_with_vad(
            audio_array, perform_diarization
        )
    else:
        segments = None

    return (
        audio_array,
        prepared.duration,
        segments,
        prepared.source_sample_rate,
        prepared.source_channels,
    )


class AudioService(BaseService):
    def __init__(self):
        super().__init__()

        from static.data.audio import DUMMY_WAV_BASE64

        warmup_task_data = (DUMMY_WAV_BASE64, True, True)
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
                source_sample_rate,
                source_channels,
            ) = await self._cpu_workload_handler.execute_task(
                request.file,
                request.is_preprocessing_enabled,
                request.perform_diarization,
                request.chunk_duration_seconds,
            )

            request._audio_array = audio_array
            request._duration = duration
            request._segments = segments
            # The submitted operating point, kept for the stage-throughput
            # labels: the array itself is already mono at the default rate.
            request._source_sample_rate = source_sample_rate
            request._source_channels = source_channels

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

        from config.constants import ModelRunners

        # Qwen3-ASR only: skip copying the encoded audio payload. It is cleared
        # below anyway, but model_dump() would deep-copy the full base64 clip
        # once per segment first (~13MB x 32 segments for a 320s request), which
        # is serial work on the event loop and made fan-out cost scale with
        # duration x segment count (320s measured 103x -> 306x once removed).
        # Whisper keeps the original dump so its production path is untouched.
        if settings.model_runner == ModelRunners.TT_QWEN3_ASR.value:
            field_values = original_request.model_dump(exclude={"file"})
            field_values["file"] = ""  # placeholder; required field, cleared below
        else:
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
        # A segment inherits the parent's operating point; it is the same audio.
        new_request._source_sample_rate = getattr(
            original_request, "_source_sample_rate", None
        )
        new_request._source_channels = getattr(
            original_request, "_source_channels", None
        )
        new_request.file = None  # Clear file data to save memory

        return new_request

    def combine_results(self, results):
        return combine_transcription_responses(results)

    def stop_workers(self):
        self.logger.info("Shutting down audio preprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
