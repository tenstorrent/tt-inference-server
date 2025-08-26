# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import struct

import numpy as np
from config.settings import settings
from utils.logger import TTLogger
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


class AudioManager:
    _whisperx_device: str = "cpu"

    def __init__(self):
        super().__init__()
        self._logger = TTLogger()

        self._logger.info("Loading WhisperX model for VAD...")
        self._vad_model = whisperx.load_model(
            "base",
            device=self._whisperx_device,
            compute_type="int8"
        )
        
        try:
            self._logger.info("Loading speaker diarization model...")
            self._diarization_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.0",
                use_auth_token=settings.huggingface_token,
                device=self._whisperx_device
            )
            self._logger.info("Speaker diarization model loaded successfully")
        except Exception as e:
            self._logger.warning(f"Failed to load diarization model: {e}")
            self._diarization_model = None

        try:
            self._logger.info("Loading WhisperX alignment model...")
            self._align_model, self._align_model_metadata = whisperx.load_align_model(
                language_code="en", 
                device=self._whisperx_device
            )
            self._logger.info("WhisperX alignment model loaded successfully")
        except Exception as e:
            self._logger.warning(f"Failed to load alignment model: {e}")
            self._align_model = None
            self._align_model_metadata = {
                "language": "en",
                "dictionary": {},
                "type": "huggingface"
                }

    def to_audio_array(self, file):
        """Convert base64-encoded audio file to numpy array for audio model inference."""
        try:
            audio_bytes = base64.b64decode(file)
            self._validate_file_size(audio_bytes)
            audio_array = self._convert_to_audio_array(audio_bytes)
            return self._validate_and_truncate_duration(audio_array)
        except Exception as e:
            self._logger.error(f"Failed to decode audio data: {e}")
            raise ValueError(f"Failed to process audio data: {str(e)}")

    def apply_vad(self, audio_array):
        """Apply Voice Activity Detection to find speech segments."""
        try:
            # Use WhisperX model to transcribe and get word-level timestamps
            # This gives us natural speech segmentation with VAD built-in
            self._logger.info("Using WhisperX model for VAD preprocessing")
            
            # WhisperX expects audio at 16kHz
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Run WhisperX transcription to get segments with timestamps
            result = self._vad_model.transcribe(audio_array, batch_size=16)
            
            # Extract segments from WhisperX result
            segments = []
            if result and "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        "start": segment.get("start", 0),
                        "end": segment.get("end", len(audio_array) / settings.default_sample_rate),
                        "text": segment.get("text", "").strip()
                    })
                    
            # If no segments found, fall back to simple VAD
            if not segments:
                self._logger.warning("WhisperX found no segments, falling back to simple VAD")
                return self._simple_vad(audio_array)
                
            self._logger.info(f"WhisperX VAD detected {len(segments)} speech segments")
            return segments

        except Exception as e:
            self._logger.error(f"WhisperX VAD processing failed: {e}")
            return self._simple_vad(audio_array)

    def _simple_vad(self, audio_array):
        """Simple energy-based Voice Activity Detection."""
        try:
            # Calculate frame energy
            frame_length = int(0.025 * settings.default_sample_rate)  # 25ms frames
            hop_length = int(0.010 * settings.default_sample_rate)    # 10ms hop
            
            # Calculate RMS energy for each frame
            energy = []
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frame = audio_array[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                energy.append(rms)
            
            energy = np.array(energy)
            
            # Simple threshold-based VAD
            threshold = np.mean(energy) * 0.3  # Adjust threshold as needed
            speech_frames = energy > threshold
            
            # Find continuous speech segments
            segments = []
            in_speech = False
            start_time = 0
            
            for i, is_speech in enumerate(speech_frames):
                time = i * hop_length / settings.default_sample_rate
                
                if is_speech and not in_speech:
                    # Start of speech
                    start_time = time
                    in_speech = True
                elif not is_speech and in_speech:
                    # End of speech
                    if time - start_time > 0.1:  # Minimum 100ms segments
                        segments.append({"start": start_time, "end": time})
                    in_speech = False
            
            # Handle case where audio ends during speech
            if in_speech:
                end_time = len(audio_array) / settings.default_sample_rate
                if end_time - start_time > 0.1:
                    segments.append({"start": start_time, "end": end_time})
            
            # If no segments found, return the full audio
            if not segments:
                duration = len(audio_array) / settings.default_sample_rate
                segments = [{"start": 0, "end": duration}]
                
            self._logger.info(f"Simple VAD detected {len(segments)} speech segments")
            return segments
            
        except Exception as e:
            self._logger.error(f"Simple VAD failed: {e}")
            duration = len(audio_array) / settings.default_sample_rate

    def apply_diarization(self, audio_array, segments):
        """Apply speaker diarization to separate different speakers."""
        try:
            # Try WhisperX alignment if available
            if hasattr(whisperx, 'assign_word_speakers') and hasattr(whisperx, 'load_align_model'):
                has_text_segments = any(segment.get("text", "").strip() for segment in segments)
                if self._align_model and has_text_segments:
                    aligned_segments = self._perform_whisperx_alignment(audio_array, segments)
                    if aligned_segments:
                        diarized_segments = self._perform_speaker_diarization(audio_array, aligned_segments)
                        if diarized_segments:
                            return diarized_segments
                        return aligned_segments
        except Exception as e:
            self._logger.warning(f"Speaker diarization failed: {e}")

        # Fallback to simple speaker assignment
        self._logger.info("Default speaker assignment")
        return [dict(segment, speaker="SPEAKER_00") for segment in segments]
    
    def _perform_whisperx_alignment(self, audio_array, segments):
        try:
            # Format segments for WhisperX
            formatted_segments = []
            for segment in segments:
                text = segment.get("text", "").strip()
                if text:
                    formatted_segments.append({
                        "start": float(segment.get("start", 0)),
                        "end": float(segment.get("end", 0)),
                        "text": text
                    })
            if not formatted_segments:
                self._logger.warning("No valid segments for alignment")
                return None
            
            # Perform alignment
            self._logger.info("Starting WhisperX alignment")
            aligned_result = whisperx.align(
                formatted_segments,
                self._align_model,
                self._align_model_metadata,
                audio_array,
                self._whisperx_device
            )

            enhanced_segments = aligned_result.get("segments", segments)
            
            self._logger.info(f"WhisperX alignment completed for {len(enhanced_segments)} segments")
            return enhanced_segments
            
        except Exception as e:
            self._logger.warning(f"WhisperX alignment failed: {e}")
            return None

    def _perform_speaker_diarization(self, audio_array, aligned_segments):
        try:
            # Perform diarization on the audio
            self._logger.info("Performing speaker diarization...")
            diarization_result = self._diarization_model(audio_array)
            
            # Create transcript result in the format expected by assign_word_speakers
            transcript_result = {"segments": aligned_segments}
            
            # Assign speakers to segments using WhisperX
            diarized_result = assign_word_speakers(
                diarization_result, 
                transcript_result
            )
            
            diarized_segments = diarized_result["segments"]
            self._logger.info(f"Speaker diarization completed for {len(diarized_segments)} segments")
            
            # Count unique speakers
            speakers = set()
            for segment in diarized_segments:
                if "speaker" in segment:
                    speakers.add(segment["speaker"])
            
            self._logger.info(f"Detected {len(speakers)} unique speakers: {sorted(speakers)}")
            return diarized_segments
            
        except ImportError as e:
            self._logger.warning(f"WhisperX diarization not available: {e}")
            return None
        except Exception as e:
            self._logger.warning(f"Speaker diarization failed: {e}")
            return None

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

    def _validate_and_truncate_duration(self, audio_array):
        duration_seconds = len(audio_array) / settings.default_sample_rate
        if duration_seconds > settings.max_audio_duration_seconds:
            max_samples = int(settings.max_audio_duration_seconds * settings.default_sample_rate)
            self._logger.warning(f"Audio truncated from {duration_seconds:.2f}s to {settings.max_audio_duration_seconds}s")
            return audio_array[:max_samples]
        return audio_array
