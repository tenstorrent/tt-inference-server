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
            
        return super().stop_workers()

    async def process(self, request: AudioTranscriptionRequest, stream: bool = False):
        if stream:
            return self._process_streaming(request)
        else:
            return await super().process(request)
        
    async def _process_streaming(self, request: AudioTranscriptionRequest):
        """Handle streaming audio transcription requests"""
        try:
            # We process audio in chunks for real-time results
            
            if request._audio_array is None or len(request._audio_array) == 0:
                raise ValueError("No audio data available for streaming")
            
            # Real streaming approach: process audio in chunks
            sample_rate = settings.default_sample_rate
            chunk_duration = settings.streaming_chunk_duration_seconds
            chunk_size = int(chunk_duration * sample_rate)
            overlap_duration = settings.streaming_overlap_seconds
            overlap_size = int(overlap_duration * sample_rate)
            
            audio_array = request._audio_array
            total_duration = len(audio_array) / sample_rate
            chunk_id = 0
            partial_transcript = ""
            max_chunks = 50  # Safety limit to prevent infinite loops
            
            self.logger.info(f"Starting real streaming transcription: {total_duration:.2f}s audio, {chunk_duration}s chunks")
            
            # Process audio in overlapping chunks for real-time streaming
            start_sample = 0
            while start_sample < len(audio_array) and chunk_id < max_chunks:
                end_sample = min(start_sample + chunk_size, len(audio_array))
                chunk_audio = audio_array[start_sample:end_sample]
                
                # Check if chunk is too small - break if so
                min_chunk_samples = int(0.5 * sample_rate)  # 0.5 seconds minimum
                if len(chunk_audio) < min_chunk_samples:
                    self.logger.info(f"Chunk {chunk_id} too small ({len(chunk_audio)} samples), ending streaming")
                    break
                
                chunk_start_time = start_sample / sample_rate
                chunk_end_time = end_sample / sample_rate
                
                self.logger.info(f"Processing chunk {chunk_id}: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
                
                # Create a chunk request
                chunk_request = AudioTranscriptionRequest(
                    file="",  # Placeholder - we'll set the array directly
                    model=request.model,
                    language=request.language,
                    prompt=request.prompt,
                    response_format=request.response_format,
                    temperature=request.temperature,
                    timestamp_granularities=request.timestamp_granularities,
                    speaker_diarization=False  # Disable diarization for chunks to speed up processing
                )
                chunk_request._audio_array = chunk_audio
                chunk_request._audio_segments = None  # No diarization for chunks
                chunk_request._task_id = f"{request._task_id}_chunk_{chunk_id}"
                
                # Process chunk through scheduler
                try:
                    self.scheduler.process_request(chunk_request)
                    future = asyncio.get_running_loop().create_future()
                    self.scheduler.result_futures[chunk_request._task_id] = future
                    
                    # Add timeout to prevent hanging
                    chunk_result = await asyncio.wait_for(future, timeout=30.0)

                    if chunk_result:
                        processed_chunk = self.post_process(chunk_result)

                        if isinstance(processed_chunk, list) and len(processed_chunk) > 0:
                            chunk_text = processed_chunk[0].get("text", "").strip()
                            if chunk_text:
                                # Update running transcript
                                if chunk_id == 0:
                                    partial_transcript = chunk_text
                                else:
                                    partial_transcript += " " + chunk_text
                                
                                # Yield partial result for this chunk
                                yield {
                                    "type": "chunk_result",
                                    "chunk_id": chunk_id,
                                    "start_time": chunk_start_time,
                                    "end_time": chunk_end_time,
                                    "text": chunk_text,
                                    "partial_transcript": partial_transcript,
                                    "is_partial": True
                                }
                
                except asyncio.TimeoutError:
                    self.logger.warning(f"Chunk {chunk_id} processing timed out")
                    break  # Stop on timeout
                except Exception as e:
                    self.logger.warning(f"Chunk {chunk_id} processing failed: {e}")
                    # If model is not ready, stop immediately
                    if "Model is not ready" in str(e) or "405" in str(e):
                        self.logger.warning("Model not ready, stopping chunk processing")
                        break
                
                finally:
                    # Clean up the future
                    self.scheduler.result_futures.pop(chunk_request._task_id, None)
                
                chunk_id += 1
                
                # Ensure we always advance - fix infinite loop
                next_start = end_sample - overlap_size
                if next_start <= start_sample:
                    # Force advance if overlap would cause us to not move forward
                    next_start = start_sample + chunk_size // 2
                
                start_sample = next_start
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Safety check
            if chunk_id >= max_chunks:
                self.logger.warning(f"Reached maximum chunk limit ({max_chunks}), stopping")
            
            # Yield final result
            yield {
                "type": "final_result",
                "task": "transcribe",
                "language": "english",
                "duration": total_duration, 
                "text": partial_transcript or "No transcription available",
                "segments": [],
                "speaker_count": 0,
                "speakers": [],
                "is_final": True
            }
            
            # Signal end of streaming
            yield {"type": "transcription_complete", "message": "Real streaming finished"}
                
        except Exception as e:
            self.logger.error(f"Real streaming transcription failed: {e}")
            raise
    
