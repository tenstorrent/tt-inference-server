# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import numpy as np
import struct
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
        """Convert WAV file bytes to numpy array."""
        
        # Verify this is a WAV file (starts with RIFF header)
        if len(audio_bytes) < 12 or audio_bytes[:4] != b'RIFF' or audio_bytes[8:12] != b'WAVE':
            raise ValueError("Expected WAV file format (RIFF/WAVE headers not found)")
        
        self._logger.info("Processing WAV file format")
        return self._decode_wav_file(audio_bytes)

    def _decode_wav_file(self, audio_bytes):
        try:
            # Parse WAV file manually
            if len(audio_bytes) < 44:
                raise ValueError("WAV file too short")
                
            # Read WAV header
            riff = audio_bytes[0:4]
            file_size = struct.unpack('<I', audio_bytes[4:8])[0]
            wave = audio_bytes[8:12]
            fmt = audio_bytes[12:16]
            
            if riff != b'RIFF' or wave != b'WAVE' or fmt != b'fmt ':
                raise ValueError("Invalid WAV file format")
            
            # Parse format chunk
            fmt_size = struct.unpack('<I', audio_bytes[16:20])[0]
            audio_format = struct.unpack('<H', audio_bytes[20:22])[0]
            num_channels = struct.unpack('<H', audio_bytes[22:24])[0]
            sample_rate = struct.unpack('<I', audio_bytes[24:28])[0]
            bits_per_sample = struct.unpack('<H', audio_bytes[34:36])[0]
            
            self._logger.info(f"WAV format: {num_channels} channels, {sample_rate} Hz, {bits_per_sample} bits")
            
            # Find data chunk
            pos = 20 + fmt_size
            while pos < len(audio_bytes) - 8:
                chunk_id = audio_bytes[pos:pos+4]
                chunk_size = struct.unpack('<I', audio_bytes[pos+4:pos+8])[0]
                
                if chunk_id == b'data':
                    # Found audio data
                    audio_data = audio_bytes[pos+8:pos+8+chunk_size]
                    break
                    
                pos += 8 + chunk_size
            else:
                raise ValueError("No audio data found in WAV file")
            
            # Convert audio data to numpy array
            if bits_per_sample == 16:
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif bits_per_sample == 24:
                # Handle 24-bit audio
                audio_ints = []
                for i in range(0, len(audio_data), 3):
                    if i + 2 < len(audio_data):
                        # Convert 3 bytes to int24
                        byte_data = audio_data[i:i+3] + b'\x00'  # Pad to 4 bytes
                        val = struct.unpack('<i', byte_data)[0] >> 8  # Shift back
                        audio_ints.append(val)
                audio_array = np.array(audio_ints, dtype=np.float32) / 8388608.0  # 2^23
            elif bits_per_sample == 32:
                if audio_format == 3:  # IEEE float
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                else:  # 32-bit PCM
                    audio_array = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported bit depth: {bits_per_sample}")
            
            # Convert stereo to mono if needed
            if num_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)
            elif num_channels > 2:
                audio_array = audio_array.reshape(-1, num_channels).mean(axis=1)
            
            # Resample to 16kHz if needed (simple decimation/interpolation)
            if sample_rate != 16000:
                target_length = int(len(audio_array) * 16000 / sample_rate)
                indices = np.linspace(0, len(audio_array) - 1, target_length)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
            
            self._logger.info(f"Loaded WAV: {len(audio_array)} samples, duration: {len(audio_array)/16000:.2f}s")
            return audio_array.astype(np.float32)
            
        except Exception as e:
            self._logger.error(f"Failed to decode WAV file: {e}")
            raise ValueError(f"Could not decode WAV file: {str(e)}")

    def _validate_and_truncate_duration(self, audio_array):
        duration_seconds = len(audio_array) / settings.default_sample_rate
        if duration_seconds > settings.max_audio_duration_seconds:
            max_samples = int(settings.max_audio_duration_seconds * settings.default_sample_rate)
            self._logger.warning(f"Audio truncated from {duration_seconds:.2f}s to {settings.max_audio_duration_seconds}s")
            return audio_array[:max_samples]
        return audio_array