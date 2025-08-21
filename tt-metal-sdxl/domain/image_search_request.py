# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pydantic import field_validator
import base64
from io import BytesIO
from PIL import Image
from domain.base_request import BaseRequest


class ImageSearchRequest(BaseRequest):
    # Base64-encoded image
    prompt: str
    
    @field_validator("prompt")
    @classmethod
    def validate_image(cls, v):
        try:
            # Just validate it's decodable
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")
    
    def get_pil_image(self) -> Image.Image:
        """Get the PIL image from the base64 data."""
        prompt = base64.b64decode(self.prompt)
        return Image.open(BytesIO(prompt))
    
    def get_model_input(self):
        return self.prompt 