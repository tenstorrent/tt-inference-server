# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from domain.base_request import BaseRequest
from utils.logger import TTLogger
from utils.audio_manager import AudioManager

class AudioTranscriptionRequest(BaseRequest):
    file: str  # Base64-encoded audio file

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = TTLogger()
        self._audio_manager = AudioManager()

    def get_model_input(self):
        """Convert base64-encoded audio file to numpy array for audio model inference."""
        try:
            audio_bytes = base64.b64decode(self.file)
            self._audio_manager.validate_file_size(audio_bytes)
            audio_array = self._audio_manager.convert_to_audio_array(audio_bytes)
            return self._audio_manager.validate_and_truncate_duration(audio_array)
        except Exception as e:
            self._logger.error(f"Failed to decode audio data: {e}")
            raise ValueError(f"Failed to process audio data: {str(e)}")