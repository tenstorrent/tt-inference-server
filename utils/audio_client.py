# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import numpy as np
import requests
import struct
import time as time_module
from typing import Tuple, Optional


class AudioClient:
    def __init__(self, base_url=None, service_port=None, jwt_secret=None):
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:" + str(service_port)
        self.jwt_secret = jwt_secret or "your-secret-key"

    def get_health(self, attempt_number=1) -> Tuple[bool, Optional[str]]:
        """Check if the audio service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/tt-liveness", timeout=30)
            if response.status_code != 200:
                if attempt_number < 20:
                    print(f"Health check failed with status code: {response.status_code}. Retrying...")
                    time_module.sleep(15)
                    return self.get_health(attempt_number + 1)
                else:
                    raise Exception(f"Health check failed with status code: {response.status_code}")
            runner_in_use = response.json().get("runner_in_use", None)
            return True, runner_in_use
        except Exception as e:
            print(f"Health check error: {e}")
            return False, None

    def generate_random_audio(self, duration_seconds: float = 5.0, sample_rate: int = 16000) -> str:
        """Generate random audio data and return as base64-encoded WAV."""
        # Generate random audio samples (speech-like noise)
        num_samples = int(duration_seconds * sample_rate)
        
        # Generate more realistic audio by combining frequencies
        t = np.linspace(0, duration_seconds, num_samples)
        audio_data = (
            0.1 * np.sin(2 * np.pi * 440 * t) +  # 440Hz tone
            0.05 * np.sin(2 * np.pi * 880 * t) + # 880Hz harmonic
            0.02 * np.random.normal(0, 0.1, num_samples)  # Small amount of noise
        )
        
        # Ensure audio is in valid range
        audio_data = np.clip(audio_data, -0.8, 0.8).astype(np.float32)
        
        # Convert to 16-bit PCM
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_data = self._create_wav_bytes(audio_data_int16, sample_rate)
        
        # Encode to base64
        return base64.b64encode(wav_data).decode('utf-8')

    def _create_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Create WAV file bytes from audio data."""
        # WAV file header parameters
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_data) * 2  # 2 bytes per sample for 16-bit
        file_size = 36 + data_size
        
        # Build WAV file header
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', file_size, b'WAVE', b'fmt ', 16,
            1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
            b'data', data_size
        )
        
        return wav_header + audio_data.tobytes()

    def transcribe_audio(self, audio_base64: str) -> requests.Response:
        """Send audio for transcription."""
        # Use OPENAI_API_KEY if available (which contains the properly encoded JWT),
        # otherwise fall back to the provided jwt_secret
        import os
        api_key = os.environ.get("OPENAI_API_KEY", self.jwt_secret)
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audio_base64
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                json=payload,
                headers=headers,
                timeout=90
            )
            return response
        except requests.exceptions.RequestException as e:
            print(f"Audio transcription request failed: {e}")
            raise
