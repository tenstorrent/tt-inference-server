# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional, Union

import numpy as np
from config.constants import (
    DEFAULT_TTS_LANGUAGE,
    TTS_RESPONSE_FORMATS,
    XTTS_SUPPORTED_LANGUAGES,
)
from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr, field_validator

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

    # Voice cloning: base64-encoded reference AUDIO FILE (any soundfile-readable
    # format) whose voice the synthesis should clone.
    reference_audio: Optional[str] = None

    @field_validator("reference_audio", mode="before")
    @classmethod
    def validate_reference_audio(cls, v):
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("reference_audio must be a base64-encoded audio file")
        # Cap the base64 string at 16 MB (~12 MB of audio file bytes after decoding).
        # Deliberately generous: a 30 s WAV is ~5 MB.
        if len(v) > 16 * 1024 * 1024:
            raise ValueError("reference_audio exceeds the 16 MB base64 limit")
        return v

    # Synthesis language. Validated here so an unsupported code raises HTTP 422 early
    # instead of raising inside a device worker. Region variants normalize to their
    # base code ("pt-br" -> "pt", "zh-cn" -> "zh").
    language: str = DEFAULT_TTS_LANGUAGE

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v):
        if v is None:
            return DEFAULT_TTS_LANGUAGE
        base = str(v).strip().lower().split("-")[0]
        if base not in XTTS_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {v!r}; supported: {sorted(XTTS_SUPPORTED_LANGUAGES)} "
                "(region variants like 'pt-br' are accepted)"
            )
        return base

    # Optional sampling seed for stochastic TTS models: fixing it makes
    # identical text reproduce identical audio. None lets the model draw randomly.
    seed: Optional[int] = Field(default=None, ge=0, lt=2**31)

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
