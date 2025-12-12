# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from config.constants import AudioResponseFormat
from domain.base_request import BaseRequest
from pydantic import PrivateAttr, field_validator


class AudioProcessingRequest(BaseRequest):
    # Required fields
    file: Union[str, bytes]  # Base64-encoded audio string OR raw audio bytes

    # Custom fields for our implementation
    stream: bool = False
    response_format: str = AudioResponseFormat.VERBOSE_JSON.value
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
    def parse_temperatures(cls, v):
        """
        Parse temperatures from various formats into float or tuple of floats.
        """
        if v is None:
            return 0.0

        # If it's a string, parse it
        if isinstance(v, str):
            try:
                # Try to parse as comma-separated values
                values = [float(x.strip()) for x in v.split(",")]
                return tuple(values) if len(values) > 1 else values[0]
            except (ValueError, AttributeError):
                # If parsing fails, try to convert to single float
                try:
                    return float(v)
                except ValueError:
                    raise ValueError(
                        f"Invalid temperatures format: {v}. "
                        f"Expected float or comma-separated floats like '0.0,0.2,0.4'"
                    )

        # If it's a list, convert to tuple
        if isinstance(v, list):
            return tuple(float(x) for x in v)

        # If it's already a float or tuple, return as-is
        if isinstance(v, (float, int, tuple)):
            return v

        raise ValueError(
            f"Invalid temperatures type: {type(v)}. "
            f"Expected float, tuple, list, or comma-separated string"
        )
