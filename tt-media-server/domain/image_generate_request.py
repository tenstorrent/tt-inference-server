# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List, Optional, Tuple, Union

from config.constants import SDXL_VALID_IMAGE_RESOLUTIONS
from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr, field_validator, model_validator

# Flux models support fewer inference steps than other image models
_FLUX_RUNNERS = {"tt-flux.1-dev", "tt-flux.1-schnell"}
_FLUX_MIN_INFERENCE_STEPS = 4
_DEFAULT_MIN_INFERENCE_STEPS = 12

_VALID_SIZE_STRINGS = {f"{w}x{h}" for w, h in SDXL_VALID_IMAGE_RESOLUTIONS}


class ImageGenerateRequest(BaseRequest):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=4, le=50)
    seed: Optional[int] = None
    guidance_scale: float = Field(..., ge=1.0, le=20.0)
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    crop_coords_top_left: Optional[Tuple[int, float]] = Field(default=(0, 0))
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timesteps: Optional[List[Union[int, float]]] = None
    sigmas: Optional[List[Union[int, float]]] = None

    size: Optional[str] = None

    # Image output settings
    image_return_format: Optional[str] = Field(default="JPEG")
    image_quality: Optional[int] = Field(default=85, ge=50, le=100)

    # Private fields for internal processing
    _segments: Optional[List[int]] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_against_settings(self):
        settings = get_settings()
        self._validate_num_inference_steps(settings.model_runner)
        self._validate_size(settings.sdxl_image_resolution)
        return self

    def _validate_num_inference_steps(self, model_runner: str):
        if self.num_inference_steps is None:
            return
        min_steps = (
            _FLUX_MIN_INFERENCE_STEPS
            if model_runner in _FLUX_RUNNERS
            else _DEFAULT_MIN_INFERENCE_STEPS
        )
        if self.num_inference_steps < min_steps:
            raise ValueError(
                f"num_inference_steps must be >= {min_steps} for {model_runner}, "
                f"got {self.num_inference_steps}"
            )

    def _validate_size(self, configured_resolution: tuple):
        if self.size is None:
            return
        if self.size not in _VALID_SIZE_STRINGS:
            raise ValueError(
                f"Invalid size '{self.size}', must be one of {sorted(_VALID_SIZE_STRINGS)}"
            )
        w, h = self.size.split("x")
        requested = (int(w), int(h))
        if requested != configured_resolution:
            raise ValueError(
                f"Requested size '{self.size}' does not match server-configured "
                f"resolution {configured_resolution[0]}x{configured_resolution[1]}"
            )

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
