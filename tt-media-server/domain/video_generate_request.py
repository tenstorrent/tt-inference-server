# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from domain.base_request import BaseRequest
from pydantic import Field


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None
