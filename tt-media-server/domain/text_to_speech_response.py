# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from pydantic import BaseModel


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
            "speaker_id": self.speaker_id,
        }
