# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional
from pydantic import BaseModel

class PartialStreamingAudioResponse(BaseModel):
    """Response for streaming audio chunks"""
    audio_chunk: str  # Base64-encoded audio chunk
    chunk_id: int  # Sequential chunk identifier
    format: str = "wav"  # Audio format (wav, mp3, etc.)
    sample_rate: int = 16000  # Sample rate in Hz

    def to_dict(self):
        return {
            "audio_chunk": self.audio_chunk,
            "chunk_id": self.chunk_id,
            "format": self.format,
            "sample_rate": self.sample_rate
        }

class TextToSpeechResponse(BaseModel):
    """Complete response for text-to-speech generation"""
    audio: str  # Base64-encoded complete audio
    duration: float  # Audio duration in seconds
    sample_rate: int = 16000  # Sample rate in Hz
    format: str = "wav"  # Audio format
    speaker_id: Optional[str] = None  # Speaker ID used (if any)

    def to_dict(self):
        return {
            "audio": self.audio,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "speaker_id": self.speaker_id
        }

