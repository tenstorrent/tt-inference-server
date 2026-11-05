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
from pydantic import Field, field_validator, model_validator


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    # Optional fields
    negative_prompt: Optional[str] = None
    # None = the default; resolved in _default_steps.
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=MIN_VIDEO_INFERENCE_STEPS,
        le=MAX_VIDEO_INFERENCE_STEPS,
    )
    seed: Optional[int] = None

    # Shape. None = the model's default.
    height: Optional[int] = Field(default=None, gt=0)
    width: Optional[int] = Field(default=None, gt=0)
    fps: Optional[float] = Field(default=None, gt=0)
    duration: Optional[float] = Field(default=None, gt=0)
    num_frames: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _default_steps(self):
        if self.num_inference_steps is None:
            self.num_inference_steps = DEFAULT_VIDEO_INFERENCE_STEPS
        return self
