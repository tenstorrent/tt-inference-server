# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from config.constants import ResponseFormat
from domain.base_request import BaseRequest
from pydantic import PrivateAttr, field_validator


class AudioProcessingRequest(BaseRequest):
    # Required fields
    file: Union[str, bytes]  # Base64-encoded audio string OR raw audio bytes

    # Custom fields for our implementation
    stream: bool = False
    response_format: str = ResponseFormat.VERBOSE_JSON.value
    is_preprocessing_enabled: bool = (
        True  # Enable VAD and diarization for specific request
    )
    perform_diarization: bool = (
        False  # Whether to perform speaker diarization during preprocessing
    )

    temperatures: Optional[Union[float, Tuple[float, ...]]] = 0.0
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -2.0
    no_speech_threshold: Optional[float] = 0.6
    return_timestamps: Optional[bool] = False
    prompt: Optional[str] = ""

    # Private fields for internal processing
    _audio_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _segments: Optional[List[Dict[str, Union[float, str]]]] = PrivateAttr(default=None)
    _duration: float = PrivateAttr(default=0.0)

    @field_validator("temperatures", mode="before")
    @classmethod
    def parse_and_validate_temperatures(cls, temperatures):
        if temperatures is None or temperatures == "":
            return None

        if isinstance(temperatures, (float, int)):
            return float(temperatures)

        if isinstance(temperatures, (tuple, list)) and temperatures:
            try:
                # Validate all elements are numeric
                temp_values = tuple(float(t) for t in temperatures)
                return temp_values if len(temp_values) > 1 else temp_values[0]
            except (ValueError, TypeError):
                pass

        if isinstance(temperatures, str) and temperatures.strip():
            try:
                temp_values = [float(t.strip()) for t in temperatures.split(",")]
                if temp_values:
                    return (
                        tuple(temp_values) if len(temp_values) > 1 else temp_values[0]
                    )
            except (ValueError, AttributeError):
                pass

        raise ValueError(
            "Invalid temperatures format. Use comma-separated floats.",
        )
