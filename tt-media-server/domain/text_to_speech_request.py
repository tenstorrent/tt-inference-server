# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional, Union
import numpy as np
from pydantic import ConfigDict, PrivateAttr, field_validator
from config.constants import AudioResponseFormat
from domain.base_request import BaseRequest


class TextToSpeechRequest(BaseRequest):
    # Required for np.ndarray type in private fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        return text

    # Optional fields for speaker embedding
    speaker_embedding: Optional[Union[str, bytes]] = (
        None  # Base64-encoded or raw bytes of speaker embedding
    )
    speaker_id: Optional[str] = None  # ID for pre-configured speaker embeddings

    # Custom fields for our implementation
    response_format: str = AudioResponseFormat.VERBOSE_JSON.value
    stream: bool = False  # Whether to stream audio generation

    # Private fields for internal processing
    _speaker_embedding_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _estimated_duration: Optional[float] = PrivateAttr(default=None)
