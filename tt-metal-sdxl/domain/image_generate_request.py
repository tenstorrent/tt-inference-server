# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from pydantic import Field
from domain.base_request import BaseRequest

class ImageGenerateRequest(BaseRequest):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None
    # guidance_scale: float = Field(..., ge=1.0, le=20.0)
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)

    def get_model_input(self):
        return self