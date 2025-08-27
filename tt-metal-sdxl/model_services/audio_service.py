# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.audio_transcription_request import AudioTranscriptionRequest
from model_services.base_service import BaseService
from config.settings import settings
from utils.audio_manager import AudioManager
from utils.logger import TTLogger

class AudioService(BaseService):

    def __init__(self):
        super().__init__()
        self._logger = TTLogger()
        self._audio_manager = AudioManager()

    def pre_process(self, request: AudioTranscriptionRequest):
        try:
            request._audio_array = self._audio_manager.to_audio_array(request.file)

            if settings.enable_whisperx_preprocessing:
                # Apply VAD to detect speech segments
                segments = self._audio_manager.apply_vad(request._audio_array)

                # Apply speaker diarization if enabled
                if settings.enable_speaker_diarization:
                    segments = self._audio_manager.apply_diarization(request._audio_array, segments)

                self._logger.info(f"WhisperX preprocessing completed. Found {len(segments)} speech segments")
                request._whisperx_segments = segments
            else:
                self._logger.info("WhisperX preprocessing disabled, skipping VAD and diarization")
        
        except Exception as e:
            self._logger.error(f"Audio preprocessing failed: {e}")
        return request