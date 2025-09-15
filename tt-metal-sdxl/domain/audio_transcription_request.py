# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Union
import numpy as np
from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    # Required fields
    file: str  # Base64-encoded audio file
    
    # Optional OpenAI API compatible fields
    model: Optional[str] = "whisper-1"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0
    timestamp_granularities: Optional[List[str]] = None
    
    # Custom fields for our implementation
    speaker_diarization: bool = False
    stream: bool = False

    # Private fields for internal processing
    _audio_array: Optional[np.ndarray] = None
    _return_perf_metrics: bool = False
    _audio_segments: Optional[List[Dict[str, Union[float, str]]]] = None