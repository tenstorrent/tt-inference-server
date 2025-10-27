# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import os
import struct

import numpy as np
from config.constants import ModelServices, SupportedModels
from config.settings import settings
from utils.logger import TTLogger

if settings.model_service == ModelServices.AUDIO.value:
    from whisperx.diarize import DiarizationPipeline


class AudioManager:
    _whisperx_device: str = "cpu"

    def __init__(self):
        self._logger = TTLogger()
        self._diarization_model = None
        
        if settings.allow_audio_preprocessing:
            self._initialize_diarization_model()
        else:
            self._logger.info("Audio preprocessing disabled - only basic transcription available")

    def to_audio_array(self, file, should_preprocess):
        """Convert base64-encoded audio file to numpy array for audio model inference."""
        try:
            audio_bytes = base64.b64decode(file)
            self._validate_file_size(audio_bytes)
            audio_array = self._convert_to_audio_array(audio_bytes)
            return self._validate_and_truncate_duration(audio_array, should_preprocess)
        except Exception as e:
            self._logger.error(f"Failed to decode audio data: {e}")
            raise ValueError(f"Failed to process audio data: {str(e)}")

    def apply_diarization_with_vad(self, audio_array):
        """Apply speaker diarization (includes VAD), then create speaker-aware chunks for Whisper processing."""  
        if self._diarization_model is None:
            raise RuntimeError("Speaker diarization model not available - cannot perform diarization")
        
        self._logger.info("Performing speaker diarization...")
        diarization_result = self._diarization_model(audio_array)
        
        # Extract VAD segments (speech regions) with speaker info
        vad_segments = []
        for _, row in diarization_result.iterrows():
            vad_segments.append({
                "start": row.get('start', 0),
                "end": row.get('end', 0), 
                "text": "",  # TT-Metal will fill this
                "speaker": row.get('speaker', 'SPEAKER_00')
            })
        
        if not vad_segments:
            # Fallback: create single segment for entire audio
            vad_segments = [{
                "start": 0.0,
                "end": len(audio_array) / settings.default_sample_rate,
                "text": "",
                "speaker": "SPEAKER_00"
            }]
        
        whisper_chunks = self._merge_vad_segments_by_speaker_and_duration(vad_segments)
        self._logger.info(f"Diarization detected {len(vad_segments)} VAD segments, created {len(whisper_chunks)} speaker-aware chunks for Whisper")

        return whisper_chunks

    def _merge_vad_segments_by_speaker_and_duration(self, vad_segments, target_chunk_duration=30.0):
        """
        Create speaker-aware chunks for Whisper processing that balance speaker boundaries with optimal chunk sizes.
        Respects speaker boundaries while ensuring reasonable chunk durations for Whisper performance.
        """
        if not vad_segments:
            return []
        
        chunks = []
        current_chunk_start = vad_segments[0]["start"]
        current_chunk_end = vad_segments[0]["end"]
        current_speaker = vad_segments[0]["speaker"]
        
        for segment in vad_segments[1:]:
            potential_end = segment["end"]
            potential_duration = potential_end - current_chunk_start
            
            # Finalize chunk if:
            # 1. Speaker changes (always respect speaker boundaries), OR
            # 2. Would exceed target duration
            should_finalize = (
                segment["speaker"] != current_speaker or
                potential_duration > target_chunk_duration
            )
            
            if should_finalize:
                chunks.append({
                    "start": current_chunk_start,
                    "end": current_chunk_end,
                    "text": "",
                    "speaker": current_speaker
                })
                
                # Start new chunk
                current_chunk_start = segment["start"]
                current_chunk_end = segment["end"]
                current_speaker = segment["speaker"]
            else:
                # Add segment to current chunk (same speaker only)
                current_chunk_end = segment["end"]
        
        # Add final chunk
        if current_chunk_start < current_chunk_end:
            chunks.append({
                "start": current_chunk_start,
                "end": current_chunk_end,
                "text": "",
                "speaker": current_speaker
            })
        
        self._logger.info(f"Created {len(chunks)} Whisper chunks")
        return chunks

    def _initialize_diarization_model(self):
        """Initialize diarization model."""
        try:
            if not os.getenv("HF_TOKEN", None):
                self._logger.warning("HF_TOKEN environment variable not set.")

            self._logger.info("Loading speaker diarization model...")
            self._diarization_model = DiarizationPipeline(
                model_name=settings.preprocessing_model_weights_path or SupportedModels.PYANNOTE_SPEAKER_DIARIZATION.value,
                use_auth_token=os.getenv("HF_TOKEN", None),
                device=self._whisperx_device
            )
            self._logger.info("Speaker diarization model loaded successfully")
        except Exception as e:
            self._logger.warning(f"Failed to load diarization model: {e}. Continuing without audio preprocessing")
            self._diarization_model = None
            
            # Provide actionable next steps
            self._logger.info("To enable audio preprocessing:")
            self._logger.info("1. Ensure HF_TOKEN is set: export HF_TOKEN=your_huggingface_token")
            self._logger.info("2. Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.0 and https://hf.co/pyannote/segmentation-3.0")
            self._logger.info("3. Restart the service")

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
                # WAV files are typically little-endian
                endian = '<'
                for i in range(0, len(audio_data), 3):
                    if i + 3 <= len(audio_data):  # Ensure all three bytes are available
                        # Convert 3 bytes to int24 with proper endianness
                        if endian == '<':
                            byte_data = audio_data[i:i+3] + b'\x00'  # Pad to 4 bytes (LSB)
                        else:
                            byte_data = b'\x00' + audio_data[i:i+3]  # Pad to 4 bytes (MSB)
                        val = struct.unpack(endian + 'i', byte_data)[0] >> 8  # Shift back
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

            # Resample to default sample rate if needed (simple linear interpolation)
            if sample_rate != settings.default_sample_rate:
                target_length = int(len(audio_array) * settings.default_sample_rate / sample_rate)
                indices = np.linspace(0, len(audio_array) - 1, target_length)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

            self._logger.info(f"Loaded WAV: {len(audio_array)} samples, duration: {len(audio_array)/settings.default_sample_rate:.2f}s")
            return audio_array.astype(np.float32)

        except Exception as e:
            self._logger.error(f"Failed to decode WAV file: {e}")
            raise ValueError(f"Could not decode WAV file: {str(e)}")

    def _validate_and_truncate_duration(self, audio_array, should_preprocess):
        duration_seconds = len(audio_array) / settings.default_sample_rate
        
        # Use extended duration limit when preprocessing is allowed and requested
        max_duration = (
            settings.max_audio_duration_seconds 
            if should_preprocess and self._diarization_model is not None
            else settings.max_audio_duration_seconds
        )
        
        if duration_seconds > max_duration:
            max_samples = int(max_duration * settings.default_sample_rate)
            self._logger.warning(f"Audio truncated from {duration_seconds:.2f}s to {max_duration}s")
            return audio_array[:max_samples], max_duration
        return audio_array, duration_seconds
