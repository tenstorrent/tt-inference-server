# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional, Union

import numpy as np
from config.constants import ResponseFormat
from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import PrivateAttr, field_validator

# Load max text length once at module import time (avoids runtime call in validator)
_settings = get_settings()
MAX_TTS_TEXT_LENGTH = _settings.max_tts_text_length


class TextToSpeechRequest(BaseRequest):
    # Required fields
    text: str  # Input text to convert to speech

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, text):
        if text is None:
            raise ValueError("Text cannot be None")
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if len(text) > MAX_TTS_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {MAX_TTS_TEXT_LENGTH} characters. "
                f"Received {len(text)} characters."
            )
        return text

    # Optional fields for speaker embedding
    speaker_embedding: Optional[Union[str, bytes]] = (
        None  # Base64-encoded or raw bytes of speaker embedding
    )
    speaker_id: Optional[str] = None  # ID for pre-configured speaker embeddings

    # Response format
    response_format: str = ResponseFormat.AUDIO.value  # ResponseFormat.AUDIO for WAV bytes, ResponseFormat.VERBOSE_JSON or ResponseFormat.JSON for JSON

    # Private fields for internal processing
    _speaker_embedding_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _estimated_duration: Optional[float] = PrivateAttr(default=None)
