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
    # When unset, sp_runner / runners use service defaults (e.g. 480×832×81).
    height: Optional[int] = Field(default=None, ge=16, le=4096)
    width: Optional[int] = Field(default=None, ge=16, le=4096)
    num_frames: Optional[int] = Field(default=None, ge=1, le=2048)
