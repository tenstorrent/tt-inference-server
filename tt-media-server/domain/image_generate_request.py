# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List, Optional, Tuple, Union

from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr, field_validator


class ImageGenerateRequest(BaseRequest):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None
    guidance_scale: float = Field(..., ge=1.0, le=20.0)
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    crop_coords_top_left: Optional[Tuple[int, float]] = Field(default=(0, 0))
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timesteps: Optional[List[Union[int, float]]] = None
    sigmas: Optional[List[Union[int, float]]] = None

    # Image output settings
    image_return_format: Optional[str] = Field(default="JPEG")
    image_quality: Optional[int] = Field(default=85, ge=50, le=100)

    # Private fields for internal processing
    _segments: Optional[List[int]] = PrivateAttr(default=None)

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
