# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.audio_transcription_request import AudioTranscriptionRequest
from model_services.base_service import BaseService
from config.settings import settings
from model_services.process_queue_handler import ProcessQueueHandler

def create_audio_worker_context():
    from utils.audio_manager import AudioManager
    return AudioManager()

def audio_worker_function(audio_manager, audio_file_data, is_preprocessing_enabled):
    """Process audio data using the initialized AudioManager"""
    from config.settings import settings

    should_preprocess = (
        settings.allow_audio_preprocessing and
        is_preprocessing_enabled
    )

    # Process audio
    audio_array, duration = audio_manager.to_audio_array(audio_file_data, should_preprocess)
    audio_segments = audio_manager.apply_diarization_with_vad(audio_array) if should_preprocess else None

    return (audio_array, duration, audio_segments)

class AudioService(BaseService):
    def __init__(self):
        super().__init__()

        from static.data.audio import DUMMY_WAV_BASE64
        warmup_task_data = (DUMMY_WAV_BASE64, True)
        self._process_queue_handler = ProcessQueueHandler(
            name="AudioPreprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=audio_worker_function,
            worker_context_setup=create_audio_worker_context,
            warmup_task_data=warmup_task_data
        )

    async def pre_process(self, request: AudioTranscriptionRequest):
        """Asynchronous preprocessing using queue-based workers"""
        try:
            if request.file is None:
                raise ValueError("No audio data provided")

            audio_array, duration, audio_segments = await self._process_queue_handler.submit_task(
                request.file,
                request.is_preprocessing_enabled
            )

            request._audio_array = audio_array
            request._duration = duration
            request._audio_segments = audio_segments

            if audio_segments:
                self.logger.info(f"WhisperX preprocessing completed. Found {len(audio_segments)} speech segments")
            else:
                if not settings.allow_audio_preprocessing:
                    self.logger.info("WhisperX preprocessing not allowed, skipping VAD and diarization")
                elif not request.is_preprocessing_enabled:
                    self.logger.info("WhisperX preprocessing disabled for this request, skipping VAD and diarization")
                else:
                    self.logger.info("WhisperX preprocessing skipped")

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise

        return request

    def stop_workers(self):
        self.logger.info("Shutting down audio preprocessing workers")
        self._process_queue_handler.stop_workers()

        return super().stop_workers()
