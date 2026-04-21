# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import List, Optional, Tuple, Union

from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr, field_validator

# Flux models support fewer inference steps than other image models
_FLUX_RUNNERS = {"tt-flux.1-dev", "tt-flux.1-schnell"}
_FLUX_MIN_INFERENCE_STEPS = 4
_DEFAULT_MIN_INFERENCE_STEPS = 12
_SKIP_STEP_VALIDATION_RUNNERS = {"tt-z-image-turbo"}


class BaseImageRequest(BaseRequest):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=4, le=50)
    seed: Optional[int] = None
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)

    # Image output settings
    image_return_format: Optional[str] = Field(default="JPEG")
    image_quality: Optional[int] = Field(default=85, ge=50, le=100)

    # Private fields for internal processing
    _segments: Optional[List[int]] = PrivateAttr(default=None)

    @field_validator("num_inference_steps")
    @classmethod
    def validate_num_inference_steps(cls, v):
        if v is None:
            return v
        model_runner = get_settings().model_runner
        if model_runner in _SKIP_STEP_VALIDATION_RUNNERS:
            return v
        min_steps = (
            _FLUX_MIN_INFERENCE_STEPS
            if model_runner in _FLUX_RUNNERS
            else _DEFAULT_MIN_INFERENCE_STEPS
        )
        if v < min_steps:
            raise ValueError(
                f"num_inference_steps must be >= {min_steps} for {model_runner}, got {v}"
            )
        return v

    @field_validator("image_return_format")
    @classmethod
    def validate_image_return_format(cls, v):
        if v is not None and v not in ["JPEG", "PNG"]:
            raise ValueError("image_return_format must be 'JPEG' or 'PNG'")
        return v

    def update_object(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ImageGenerateRequest(BaseImageRequest):
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    crop_coords_top_left: Optional[Tuple[int, float]] = Field(default=(0, 0))
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timesteps: Optional[List[Union[int, float]]] = None
    sigmas: Optional[List[Union[int, float]]] = None

    # LoRA adapter settings
    lora_path: Optional[str] = Field(default=None)
    lora_scale: Optional[float] = Field(default=0.5, ge=0.0, le=2.0)
