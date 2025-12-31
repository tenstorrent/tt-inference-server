# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
from typing import Union

from config.constants import ResponseFormat
from domain.base_request import BaseRequest
from PIL import Image
from pydantic import Field, field_validator


class ImageSearchRequest(BaseRequest):
    # Base64-encoded image
    prompt: Union[str, bytes]

    # Number of top predictions to return
    top_k: int = Field(default=3, ge=1)

    # Response format: "json" or "verbose"
    response_format: str = ResponseFormat.JSON.value

    # Minimum confidence threshold
    min_confidence: float = Field(default=70.0, ge=0.0, le=100.0)

    @field_validator("prompt")
    @classmethod
    def validate_image(cls, v):
        if isinstance(v, bytes):
            return v
        try:
            # Just validate it's decodable
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        if v not in ["json", "verbose_json"]:
            raise ValueError("response_format must be 'json' or 'verbose_json'")
        return v

    def get_pil_image(self) -> Image.Image:
        """Get the PIL image from the base64 data."""
        if isinstance(self.prompt, bytes):
            return Image.open(BytesIO(self.prompt))
        prompt = base64.b64decode(self.prompt)
        return Image.open(BytesIO(prompt))
