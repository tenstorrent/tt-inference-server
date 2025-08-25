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
        """
        Apply WhisperX preprocessing to audio data.
        
        WhisperX preprocessing includes:
        1. Voice Activity Detection (VAD)
        2. Audio segmentation 
        3. Speaker diarization (if enabled)
        
        Args:
            request: AudioTranscriptionRequest with audio data
            
        Returns:
            Preprocessed request with enhanced audio segments
        """
        try:
            # Get audio data from request
            request._audio_array = self._audio_manager.to_audio_array(request.file)

            # # Apply VAD to detect speech segments
            # segments = self._audio_manager.apply_vad(audio_array)

            # # Optionally apply speaker diarization
            # if getattr(settings, 'enable_speaker_diarization', False):
            #     segments = self._audio_manager.apply_diarization(audio_array, segments)

            # # Update request with preprocessed segments
            # # For now, we'll keep the original audio but could modify to use segments
            # self._logger.info(f"WhisperX preprocessing completed. Found {len(segments)} speech segments")
            
            # # Store segments in request for potential use in post-processing
            # if hasattr(request, '_whisperx_segments'):
            #     request._whisperx_segments = segments
            
            return request
            
        except Exception as e:
            self._logger.error(f"WhisperX preprocessing failed: {e}")
            # Fallback to original request if preprocessing fails
            return request