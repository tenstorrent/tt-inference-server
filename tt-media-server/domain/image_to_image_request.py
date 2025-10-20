# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from domain.base_request import BaseRequest
from pydantic import Field
from typing import Optional, Union, List, Tuple

class ImageToImageRequest(BaseRequest):
    prompt: str
    image: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None
    guidance_scale: float = Field(..., ge=1.0, le=20.0)
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    strength: Optional[float] = 0.3
    aesthetic_score: Optional[float] = 6.0
    negative_aesthetic_score: Optional[float] = 2.5


    def update_object(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)