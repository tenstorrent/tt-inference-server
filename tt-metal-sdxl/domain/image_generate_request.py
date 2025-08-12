# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional
from domain.base_request import BaseRequest

class ImageGenerateRequest(BaseRequest):
    prompt: str
    # negative_prompt: Optional[str] = None
    # output_format: OutputFormat
    # num_inference_step: int = Field(..., ge=1, le=50)