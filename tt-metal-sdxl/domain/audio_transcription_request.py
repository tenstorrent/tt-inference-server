# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, List, Optional, Union
import numpy as np
from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    file: str  # Base64-encoded audio file

    _audio_array: Optional[np.ndarray] = None
    _return_perf_metrics: bool = False
    _whisperx_segments: Optional[List[Dict[str, Union[float, str]]]] = None