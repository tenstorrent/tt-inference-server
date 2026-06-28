# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from domain.base_request import BaseRequest
from pydantic import Field


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Negative prompt. Not applied by LTX-2.3 distilled (guidance-free, no CFG).",
    )
    num_inference_steps: Optional[int] = Field(
        default=20,
        ge=12,
        le=50,
        description="Denoise steps. Ignored by LTX-2.3 distilled (fixed 8+3 sigma schedule).",
    )
    seed: Optional[int] = None
