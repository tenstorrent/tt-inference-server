# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Union

import numpy as np
from config.constants import AudioResponseFormat
from domain.base_request import BaseRequest
from pydantic import PrivateAttr


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

    # Private fields for internal processing
    _audio_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _segments: Optional[List[Dict[str, Union[float, str]]]] = PrivateAttr(default=None)
    _duration: float = 0.0
