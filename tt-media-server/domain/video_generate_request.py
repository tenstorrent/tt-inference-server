# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from config.constants import (
    DEFAULT_VIDEO_INFERENCE_STEPS,
    MAX_VIDEO_INFERENCE_STEPS,
    MIN_VIDEO_INFERENCE_STEPS,
)
from domain.base_request import BaseRequest
from pydantic import Field


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(
        default=DEFAULT_VIDEO_INFERENCE_STEPS,
        ge=MIN_VIDEO_INFERENCE_STEPS,
        le=MAX_VIDEO_INFERENCE_STEPS,
    )
    seed: Optional[int] = None
    # Optional output geometry. Additive/backward-compatible: models that fix their own
    # resolution/duration (Wan, Mochi) ignore these; the MiniMax-H3 runner honors them (clamped to
    # the model's envelope). width/height are pixels (snapped /32); num_frames is on the 17n+5 grid.
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    num_frames: Optional[int] = Field(default=None, gt=0)
