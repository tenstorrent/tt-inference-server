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

    def post_process(self, result):
        if isinstance(result, str):
            clean_text = result.replace("<EOS>", "").strip()
            return [{
                "text": clean_text
            }]
        
        return super().post_process(result)

    async def process(self, request: AudioTranscriptionRequest, stream: bool = False):
        if stream and settings.enable_service_level_streaming:
            return self._process_streaming(request)
        
        return await super().process(request)
    
    async def _process_audio_chunk(self, audio_chunk, task_id, chunk_info):
        """Process a single audio chunk (segment or time-based chunk) and return the result"""
        chunk_request = AudioTranscriptionRequest(
            file="",  # Placeholder - we'll set the array directly
            stream=True  # Use model-level streaming for faster time-to-first-token
        )
        chunk_request._audio_array = audio_chunk
        chunk_request._audio_segments = None  # No further diarization needed
        chunk_request._task_id = task_id
        
        try:
            self.scheduler.process_request(chunk_request)
            future = asyncio.get_running_loop().create_future()
            self.scheduler.result_futures[chunk_request._task_id] = future
            
            # Add timeout to prevent hanging
            chunk_result = await asyncio.wait_for(future, timeout=30.0)
            
            if chunk_result:
                chunk_text = ""
                if isinstance(chunk_result, str):
                    chunk_text = chunk_result.replace("<EOS>", "").strip()
                elif isinstance(chunk_result, list) and len(chunk_result) > 0:
                    first_result = chunk_result[0]
                    if isinstance(first_result, str):
                        chunk_text = first_result.replace("<EOS>", "").strip()
                    elif isinstance(first_result, dict):
                        chunk_text = first_result.get("text", "").strip()
                
                return chunk_text
            else:
                self.logger.warning(f"{chunk_info['type']} {chunk_info['id']} returned None result")
                return None
        
        except asyncio.TimeoutError:
            self.logger.warning(f"{chunk_info['type']} {chunk_info['id']} processing timed out")
            return None
        except Exception as e:
            self.logger.warning(f"{chunk_info['type']} {chunk_info['id']} processing failed: {e}")
            return None
        
        finally:
            self.scheduler.result_futures.pop(chunk_request._task_id, None)
        
    async def _process_streaming(self, request: AudioTranscriptionRequest):
        """Handle streaming audio transcription requests"""
        try:
            # We process audio in chunks for real-time results
            if request._audio_array is None or len(request._audio_array) == 0:
                raise ValueError("No audio data available for streaming")
            
            total_duration = len(request._audio_array) / settings.default_sample_rate
            partial_transcript = ""
            speakers_info = []
            unique_speakers = set()
            
            self.logger.info(f"Starting streaming transcription: {total_duration:.2f}s audio")
            
            if request._audio_segments and len(request._audio_segments) > 0:
                # Use diarization segments for streaming
                self.logger.info(f"Using {len(request._audio_segments)} diarization segments for streaming")
                
                for segment_id, segment in enumerate(request._audio_segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    speaker = segment.get('speaker', f"SPEAKER_{segment_id:02d}")
                    
                    # Extract audio segment
                    start_sample = int(start_time * settings.default_sample_rate)
                    end_sample = int(end_time * settings.default_sample_rate)
                    segment_audio = request._audio_array[start_sample:end_sample]
                    
                    if len(segment_audio) == 0:
                        self.logger.warning(f"Empty audio segment {segment_id} from {start_time:.2f}s to {end_time:.2f}s")
                        continue
                    
                    self.logger.info(f"Processing segment {segment_id}: {start_time:.2f}s - {end_time:.2f}s, speaker: {speaker}")
                    
                    # Process segment using helper function
                    task_id = f"{request._task_id}_segment_{segment_id}"
                    chunk_info = {"type": "Segment", "id": segment_id}
                    
                    segment_text = await self._process_audio_chunk(segment_audio, task_id, chunk_info)
                    
                    if segment_text:
                        partial_transcript = segment_text if segment_id == 0 else partial_transcript + " " + segment_text
                        
                        # Collect speaker information
                        unique_speakers.add(speaker)
                        speakers_info.append({
                            "speaker": speaker,
                            "start_time": start_time,
                            "end_time": end_time,
                            "text": segment_text
                        })
                        
                        self.logger.info(f"STREAMING: About to yield segment {segment_id} with text: '{segment_text}'")
                        
                        yield {
                            "type": "segment_result",
                            "segment_id": segment_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "text": segment_text,
                            "speaker": speaker,
                            "partial_transcript": partial_transcript,
                            "is_partial": True
                        }
                        
                        self.logger.info(f"STREAMING: Successfully yielded segment {segment_id}")
                    else:
                        self.logger.warning(f"Segment {segment_id} produced empty text after processing")
                    
                    # Small delay between segments
                    await asyncio.sleep(0.05)
            
            else:
                # No diarization segments - use time-based chunking as fallback
                self.logger.info("No diarization segments available, using time-based chunking")
                
                # Streaming approach: process audio in chunks
                chunk_size = int(settings.streaming_chunk_duration_seconds * settings.default_sample_rate)
                overlap_size = int(settings.streaming_overlap_seconds * settings.default_sample_rate)
                
                chunk_id = 0
                max_chunks = 50  # Safety limit to prevent infinite loops
                
                self.logger.info(f"Using {settings.streaming_chunk_duration_seconds}s chunks with {settings.streaming_overlap_seconds}s overlap")
                
                # Process audio in overlapping chunks for real-time streaming
                start_sample = 0
                while start_sample < len(request._audio_array) and chunk_id < max_chunks:
                    end_sample = min(start_sample + chunk_size, len(request._audio_array))
                    chunk_audio = request._audio_array[start_sample:end_sample]
                    
                    # Check if chunk is too small - break if so
                    min_chunk_samples = int(0.5 * settings.default_sample_rate)  # 0.5 seconds minimum
                    if len(chunk_audio) < min_chunk_samples:
                        self.logger.info(f"Chunk {chunk_id} too small ({len(chunk_audio)} samples), ending streaming")
                        break
                    
                    chunk_start_time = start_sample / settings.default_sample_rate
                    chunk_end_time = end_sample / settings.default_sample_rate
                    
                    self.logger.info(f"Processing chunk {chunk_id}: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
                    
                    # Process chunk using helper function
                    task_id = f"{request._task_id}_chunk_{chunk_id}"
                    chunk_info = {"type": "Chunk", "id": chunk_id}
                    
                    chunk_text = await self._process_audio_chunk(chunk_audio, task_id, chunk_info)
                    
                    self.logger.info(f"Chunk {chunk_id} extracted text: '{chunk_text}' (length: {len(chunk_text) if chunk_text else 0})")
                    
                    if chunk_text:
                        partial_transcript = chunk_text if chunk_id == 0 else partial_transcript + " " + chunk_text
                        
                        self.logger.info(f"Chunk {chunk_id} yielding result with text: '{chunk_text}'")
                        
                        yield {
                            "type": "chunk_result",
                            "chunk_id": chunk_id,
                            "start_time": chunk_start_time,
                            "end_time": chunk_end_time,
                            "text": chunk_text,
                            "partial_transcript": partial_transcript,
                            "is_partial": True
                        }
                    else:
                        self.logger.warning(f"Chunk {chunk_id} produced empty text after processing")
                    
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
            
            yield {
                "type": "final_result",
                "duration": total_duration, 
                "text": partial_transcript or "No transcription available",
                "segments": speakers_info,
                "speaker_count": len(unique_speakers),
                "speakers": list(unique_speakers),
                "is_final": True
            }
                
        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {e}")
            raise
    
