# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import math
from concurrent.futures import ProcessPoolExecutor

from domain.audio_transcription_request import AudioTranscriptionRequest
from model_services.base_service import BaseService
from config.settings import settings
from utils.audio_manager import AudioManager

# Global variable to hold AudioManager in worker processes
_worker_audio_manager = None

def _init_worker():
    """Initialize AudioManager once per worker process"""
    global _worker_audio_manager
    try:
        _worker_audio_manager = AudioManager()
    except Exception as e:
        # Log error but don't crash the worker process
        # AudioManager initialization failures will be handled when methods are called
        import logging
        logging.error(f"Error initializing AudioManager in worker process: {e}")
        _worker_audio_manager = None

def _process_audio_in_worker(audio_file_data, is_preprocessing_enabled) -> tuple[list, float, list, str]:
    """Worker function that runs in separate process"""
    try:
        global _worker_audio_manager
        
        if _worker_audio_manager is None:
            return None, 0.0, None, "AudioManager failed to initialize in worker process"

        should_preprocess = (
            settings.allow_audio_preprocessing and
            is_preprocessing_enabled 
        )

        audio_array, duration = _worker_audio_manager.to_audio_array(audio_file_data, should_preprocess)
        audio_segments = _worker_audio_manager.apply_diarization_with_vad(audio_array) if should_preprocess else None
        
        return audio_array, duration, audio_segments, None
        
    except Exception as e:
        return None, 0.0, None, str(e)


class AudioService(BaseService):

    def __init__(self):
        super().__init__()
        # Create process pool with max workers = number of scheduler workers
        # Use initializer to create AudioManager once per worker process
        self._process_pool = ProcessPoolExecutor(
            max_workers=math.ceil(self.scheduler.get_worker_count() * 1.5),
            initializer=_init_worker
        )
        # Warm up the process pool by submitting a dummy audio job.
        # This ensures that worker process is started and the AudioManager is initialized,
        # reducing latency for the first real audio request.
        from static.data.audio import DUMMY_WAV_BASE64
        self._process_pool.submit(_process_audio_in_worker, DUMMY_WAV_BASE64, True)

    async def pre_process(self, request: AudioTranscriptionRequest):
        """Asynchronous preprocessing using process pool"""
        try:
            if request.file is None:
                raise ValueError("No audio data provided")
            
            loop = asyncio.get_event_loop()
            audio_array, duration, audio_segments, error = await loop.run_in_executor(
                self._process_pool,
                _process_audio_in_worker,
                request.file,
                request.is_preprocessing_enabled
            )
            
            if error:
                raise Exception(error)
            
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
        self.logger.info("Shutting down audio processing pool")
        if hasattr(self, '_process_pool'):
            self._process_pool.shutdown(wait=True)
            
        return super().stop_workers()
