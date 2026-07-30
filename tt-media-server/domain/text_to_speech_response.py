# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class TextToSpeechResponse(BaseModel):
    """Complete response for text-to-speech generation"""

    audio: str  # Base64-encoded complete audio
    duration: float  # Audio duration in seconds
    sample_rate: int = 16000  # Sample rate in Hz
    format: str = "wav"  # Audio format
    speaker_id: Optional[str] = None  # Speaker ID used (if any)

    # Binary response body: set by post_process (WAV/MP3/OGG per response_format).
    output_bytes: Optional[bytes] = None

    def to_dict(self):
        return {
            "audio": self.audio,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "speaker_id": self.speaker_id,
        }


class TextToSpeechChunkResult(BaseModel):
    """One progressively-decoded audio chunk (or the terminal empty-audio
    marker) from streaming TTS synthesis (Inworld TTS runner only, currently).
    Mirrors ``CompletionResult``'s streaming-chunk/final-result role for text,
    but each chunk carries a small self-contained WAV (base64-encoded) instead
    of a token of text -- streaming synthesis emits one such chunk per
    ``synthesize_tp8_streaming`` chunk (see tt_modeling.py), not per token.
    """

    audio_base64: str = ""  # Base64-encoded WAV for this chunk; "" on the final marker.
    chunk_index: int
    is_final: bool = False
    sample_rate: int = 16000
    format: str = "wav"
    duration: Optional[float] = None  # This chunk's audio duration in seconds.
    speaker_id: Optional[str] = None

    def to_dict(self):
        return {
            "audio": self.audio_base64,
            "chunk_index": self.chunk_index,
            "is_final": self.is_final,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "duration": self.duration,
            "speaker_id": self.speaker_id,
        }


class TextToSpeechChunkOutput(TypedDict):
    """Output from the runner's streaming TTS path -- used the same way
    ``CompletionOutput`` is used for streaming LLM text (device_worker.py's
    generic streaming dispatch and base_service.py's process_streaming both
    key off ``type``/``data`` regardless of model domain)."""

    type: Literal["streaming_chunk", "final_result"]
    data: TextToSpeechChunkResult
