# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional, Union

import numpy as np
from config.constants import TTS_RESPONSE_FORMATS
from domain.base_request import BaseRequest
from pydantic import PrivateAttr, field_validator

# Default max text length (runner handles chunking internally)
DEFAULT_MAX_TTS_TEXT_LENGTH = 20000


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
        if len(text) > DEFAULT_MAX_TTS_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {DEFAULT_MAX_TTS_TEXT_LENGTH} characters. "
                f"Received {len(text)} characters."
            )
        return text

    # Optional fields for speaker embedding
    speaker_embedding: Optional[Union[str, bytes]] = (
        None  # Base64-encoded or raw bytes of speaker embedding
    )
    speaker_id: Optional[str] = None  # ID for pre-configured speaker embeddings

    # Response format: wav (default), mp3, ogg, json, or verbose_json
    response_format: str = "wav"

    @field_validator("response_format", mode="before")
    @classmethod
    def validate_response_format(cls, v):
        normalized = str(v).strip().lower() if v is not None else "wav"
        if normalized not in TTS_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {sorted(TTS_RESPONSE_FORMATS)}"
            )
        return normalized

    # Private fields for internal processing
    _speaker_embedding_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _estimated_duration: Optional[float] = PrivateAttr(default=None)
