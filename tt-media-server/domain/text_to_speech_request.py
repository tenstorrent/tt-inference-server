# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional, Union
import numpy as np
from domain.base_request import BaseRequest


class TextToSpeechRequest(BaseRequest):
    # Required fields
    text: str  # Input text to convert to speech

    # Optional fields for speaker embedding
    speaker_embedding: Optional[Union[str, bytes]] = (
        None  # Base64-encoded or raw bytes of speaker embedding
    )
    speaker_id: Optional[str] = None  # ID for pre-configured speaker embeddings

    # Custom fields for our implementation
    response_format: str = "verbose_json"  # API response format: "verbose_json" (full response) or "text" (audio only)
    stream: bool = False  # Whether to stream audio generation

    # Private fields for internal processing
    _speaker_embedding_array: Optional[np.ndarray] = (
        None  # Processed speaker embedding as numpy array
    )
    _estimated_duration: Optional[float] = None  # Estimated audio duration in seconds

    class Config:
        # Required for np.ndarray type in _speaker_embedding_array field
        arbitrary_types_allowed = True
