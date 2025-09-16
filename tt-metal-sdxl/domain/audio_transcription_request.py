# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Union
import numpy as np
from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    # Required fields
    file: str  # Base64-encoded audio file
    
    # Optional fields
    language: Optional[str] = None
    temperature: Optional[float] = 0.0
    
    # Custom fields for our implementation
    speaker_diarization: bool = False
    stream: bool = False

    # Private fields for internal processing
    _audio_array: Optional[np.ndarray] = None
    _return_perf_metrics: bool = False
    _audio_segments: Optional[List[Dict[str, Union[float, str]]]] = None