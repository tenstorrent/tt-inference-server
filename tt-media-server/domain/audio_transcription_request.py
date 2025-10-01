# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Union
import numpy as np
from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    # Required fields
    file: str  # Base64-encoded audio file
    
    # Custom fields for our implementation
    stream: bool = False
    return_perf_metrics: bool = False

    # Private fields for internal processing
    _audio_array: Optional[np.ndarray] = None
    _audio_segments: Optional[List[Dict[str, Union[float, str]]]] = None
    _duration: float = 0.0