# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import numpy as np
from config.settings import settings
from domain.base_request import BaseRequest
from utils.logger import TTLogger

class AudioTranscriptionRequest(BaseRequest):
    file: str  # Base64-encoded audio file

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = TTLogger()

    def get_model_input(self):
        """Convert base64-encoded audio file to numpy array for audio model inference."""
        try:
            audio_bytes = base64.b64decode(self.file)
            self._validate_file_size(audio_bytes)
            audio_array = self._convert_to_audio_array(audio_bytes)
            return self._validate_and_truncate_duration(audio_array)
        except Exception as e:
            self._logger.error(f"Failed to decode audio data: {e}")
            raise ValueError(f"Failed to process audio data: {str(e)}")

    def _validate_file_size(self, audio_bytes):
        if len(audio_bytes) > settings.max_audio_size_bytes:
            raise ValueError(f"Audio file too large: {len(audio_bytes)} bytes. Maximum allowed: {settings.max_audio_size_bytes} bytes")

    def _convert_to_audio_array(self, audio_bytes):
        # Try float32 first, then int16
        for dtype, scale in [(np.float32, 1.0), (np.int16, 1/32768.0)]:
            try:
                array = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32) * scale
                # If values are in reasonable range, accept this format
                if len(array) > 0 and np.abs(array).max() <= 1.0:
                    return array
            except (ValueError, TypeError) as e:
                self._logger.debug(f"Failed to decode as {dtype}: {e}")
                continue
        raise ValueError("Could not decode audio data as float32 or int16")

    def _validate_and_truncate_duration(self, audio_array):
        duration_seconds = len(audio_array) / settings.default_sample_rate
        if duration_seconds > settings.max_audio_duration_seconds:
            max_samples = int(settings.max_audio_duration_seconds * settings.default_sample_rate)
            self._logger.warning(f"Audio truncated from {duration_seconds:.2f}s to {settings.max_audio_duration_seconds}s")
            return audio_array[:max_samples]
        return audio_array