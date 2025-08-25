# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional
import numpy as np
from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    file: str  # Base64-encoded audio file

    _audio_array: Optional[np.ndarray] = None
    stream: bool = False
    return_perf_metrics: bool = False
    timeout_seconds: Optional[int] = None