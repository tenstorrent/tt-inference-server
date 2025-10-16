# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List, Optional, Tuple, Union

from pydantic import Field
from domain.base_request import BaseRequest

class ImageGenerateRequest(BaseRequest):
    prompt: str
    prompt_2: Optional[str] = None
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None
    timesteps: Optional[List[Union[int, float]]] = None
    sigmas: Optional[List[Union[int, float]]] = None
    guidance_scale: float = Field(..., ge=1.0, le=20.0)
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    crop_coords_top_left: Optional[Tuple[int, float]] = Field(default=(0, 0))

    def update_object(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)