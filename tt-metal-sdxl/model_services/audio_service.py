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
from utils.transcript_utils import TranscriptUtils
from domain.transcription_response import TranscriptionResponse, PartialStreamingTranscriptionResponse

# Global variable to hold AudioManager in worker processes
_worker_audio_manager = None

def _init_worker():
    """Initialize AudioManager once per worker process"""
    global _worker_audio_manager
    _worker_audio_manager = AudioManager()

def _process_audio_in_worker(audio_file_data) -> tuple[list, float, list, str]:
    """Worker function that runs in separate process"""
    try:
        global _worker_audio_manager

        audio_array, duration = _worker_audio_manager.to_audio_array(audio_file_data)
        audio_segments = _worker_audio_manager.apply_diarization_with_vad(audio_array) if settings.enable_audio_preprocessing else None
        
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
        self._process_pool.submit(_process_audio_in_worker, DUMMY_WAV_BASE64)

    async def pre_process(self, request: AudioTranscriptionRequest):
        """Asynchronous preprocessing using process pool"""
        try:
            if request.file is None:
                raise ValueError("No audio data provided")

            loop = asyncio.get_event_loop()
            audio_array, duration, audio_segments, error = await loop.run_in_executor(
                self._process_pool,
                _process_audio_in_worker,
                request.file
            )
            
            if error:
                raise Exception(error)
            
            request._audio_array = audio_array
            request._duration = duration
            request._audio_segments = audio_segments
            
            if audio_segments:
                self.logger.info(f"WhisperX preprocessing completed. Found {len(audio_segments)} speech segments")
            else:
                self.logger.info("WhisperX preprocessing disabled, skipping VAD and diarization")
                
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise
            
        return request

    def stop_workers(self):
        self.logger.info("Shutting down audio processing pool")
        if hasattr(self, '_process_pool'):
            self._process_pool.shutdown(wait=True)
            
        return super().stop_workers()

    def post_process(self, result):
        if isinstance(result, str):
            return [{
                "text": TranscriptUtils.clean_text(result)
            }]
        
        return super().post_process(result)

    async def process(self, request: AudioTranscriptionRequest):
        if request.stream:
            return self._process_model_streaming_via_scheduler(request)
        
        return await super().process(request)
    
    async def _process_model_streaming_via_scheduler(self, request: AudioTranscriptionRequest):
        """Handle model-level streaming through the scheduler/device worker"""
        if request._audio_array is None or len(request._audio_array) == 0:
            raise ValueError("No audio data available for streaming")
        
        self.logger.info("Starting model-level streaming transcription via scheduler")

        self.scheduler.process_request(request)
        future = asyncio.get_running_loop().create_future()
        self.scheduler.result_futures[request._task_id] = future
        
        try:
            result = await asyncio.wait_for(future, timeout=60.0)
            
            if not isinstance(result, dict) or result.get('type') != 'streaming_result':
                raise Exception(f"Unexpected result type for streaming request: {type(result)} - {result}")
            
            chunks = result.get('chunks', [])
            for i, chunk in enumerate(chunks):
                formatted_chunk = PartialStreamingTranscriptionResponse(
                    text=TranscriptUtils.clean_text(chunk),
                    chunk_id=i
                )
                if formatted_chunk.text:
                    yield formatted_chunk
            
            final_result_generator = self._yield_final_streaming_result(result, request)
            for final_chunk in final_result_generator:
                yield final_chunk
            
        except asyncio.TimeoutError:
            self.logger.error("Model-level streaming timed out")
            raise Exception("Streaming transcription timed out")
        except Exception as e:
            self.logger.error(f"Model-level streaming failed: {e}")
            raise
        finally:
            self.scheduler.result_futures.pop(request._task_id, None)
    
    def _yield_final_streaming_result(self, result: dict, request: AudioTranscriptionRequest):
        """Helper method to yield the final result from streaming response"""
        if 'final_result' in result:
            final_result_data = result['final_result']
            
            if isinstance(final_result_data, TranscriptionResponse):
                yield final_result_data
            else:
                yield TranscriptionResponse.from_dict(final_result_data)
        else:
            raise Exception(f"Streaming result missing 'final_result': {result}")
