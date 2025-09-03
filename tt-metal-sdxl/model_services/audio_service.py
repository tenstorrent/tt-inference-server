# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.audio_transcription_request import AudioTranscriptionRequest
from model_services.base_service import BaseService
from config.settings import settings
from utils.audio_manager import AudioManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.logger import TTLogger

audio_manager = AudioManager()

def diarize_chunk(audio_chunk, chunk_start):
    segments = audio_manager.apply_diarization_with_vad(audio_chunk)
    for seg in segments:
        seg["start"] += chunk_start
        seg["end"] += chunk_start
    return segments

class AudioService(BaseService):

    def __init__(self):
        super().__init__()
        self._logger = TTLogger()

    def _multiprocess_diarization(self, audio):
        duration = len(audio) / settings.default_sample_rate
        num_chunks = min(4, int(duration // 10) + 1)  # e.g., 4 processes or 10s per chunk
        chunk_size = len(audio) // num_chunks

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else len(audio)
            chunk_audio = audio[start:end]
            chunk_start_time = start / settings.default_sample_rate
            chunks.append((chunk_audio, chunk_start_time))

        segments = []
        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            futures = [
                executor.submit(diarize_chunk, chunk_audio, chunk_start)
                for chunk_audio, chunk_start in chunks
            ]
            for future in as_completed(futures):
                segments.extend(future.result())

        segments.sort(key=lambda s: s["start"])
        return segments

    def pre_process(self, request: AudioTranscriptionRequest):
        try:
            request._audio_array = audio_manager.to_audio_array(request.file)

            if settings.enable_audio_preprocessing:
                request._audio_segments = self._multiprocess_diarization(request._audio_array)
                self._logger.info(f"WhisperX multiprocessing completed. Found {len(request._audio_segments)} speech segments")
            else:
                self._logger.info("WhisperX preprocessing disabled, skipping VAD and diarization")
        except Exception as e:
            self._logger.error(f"Audio preprocessing failed: {e}")
        return request