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
    _worker_audio_manager = AudioManager()

def _process_audio_in_worker(audio_file_data) -> tuple[list, list, str]:
    """Worker function that runs in separate process"""
    try:
        global _worker_audio_manager

        audio_array = _worker_audio_manager.to_audio_array(audio_file_data)
        audio_segments = _worker_audio_manager.apply_diarization_with_vad(audio_array) if settings.enable_audio_preprocessing else None
        
        return audio_array, audio_segments, None
        
    except Exception as e:
        return None, None, str(e)


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
            audio_array, audio_segments, error = await loop.run_in_executor(
                self._process_pool,
                _process_audio_in_worker,
                request.file
            )
            
            if error:
                raise Exception(error)
            
            request._audio_array = audio_array
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
        
        # Clean up streaming runner
        if self._streaming_runner is not None:
            try:
                if hasattr(self._streaming_runner, 'close_device'):
                    device = self._streaming_runner.get_device()
                    self._streaming_runner.close_device(device)
                self._streaming_runner = None
                self.logger.info("Streaming runner cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up streaming runner: {e}")
            
        return super().stop_workers()

    async def process(self, request: AudioTranscriptionRequest, stream: bool = False):
        if stream:
            # Return the async generator for streaming
            return await self._process_streaming(request)
        else:
            return await super().process(request)
        
    async def _process_streaming(self, request: AudioTranscriptionRequest):
        """Handle streaming audio transcription requests"""
        try:
            # The cleanest approach: use the same device architecture but bypass multiprocessing
            # for streaming since generators can't be pickled through queues
            
            from tt_model_runners.runner_fabric import get_device_runner
            
            # Use device 0 for streaming - this doesn't conflict with multiprocessing workers
            # because TTNN devices can be shared safely between processes/threads
            device_id = "0"
            runner = get_device_runner(device_id)
            
            runner.set_streaming_mode(True)
            
            # Get device and load model (should reuse existing device context if available)
            device = runner.get_device()
            await runner.load_model(device)
            
            # Run streaming inference - this returns the async generator
            result_generator = await runner._run_inference_async([request])
            
            # Yield results from the generator
            if hasattr(result_generator, '__aiter__'):
                # Async generator
                async for partial_result in result_generator:
                    # Check for end-of-stream signal
                    if partial_result == "<EOS>" or (isinstance(partial_result, tuple) and partial_result[0] == "<EOS>"):
                        # Send final completion signal to client
                        yield {"type": "transcription_complete", "message": "Streaming finished"}
                        break
                    # Check for final result in segment processing
                    elif isinstance(partial_result, dict) and partial_result.get("is_final", False):
                        yield self.post_process(partial_result)
                        # Wait for explicit EOS signal
                        continue
                    else:
                        yield self.post_process(partial_result)
            elif hasattr(result_generator, '__iter__'):
                # Regular generator
                for partial_result in result_generator:
                    # Check for end-of-stream signal
                    if partial_result == "<EOS>" or (isinstance(partial_result, tuple) and partial_result[0] == "<EOS>"):
                        # Send final completion signal to client
                        yield {"type": "transcription_complete", "message": "Streaming finished"}
                        break
                    # Check for final result in segment processing
                    elif isinstance(partial_result, dict) and partial_result.get("is_final", False):
                        yield self.post_process(partial_result)
                        # Wait for explicit EOS signal
                        continue
                    else:
                        yield self.post_process(partial_result)
            else:
                # Single result (fallback)
                yield self.post_process(result_generator)
                yield {"type": "transcription_complete", "message": "Streaming finished"}
                
        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {e}")
            raise
    
