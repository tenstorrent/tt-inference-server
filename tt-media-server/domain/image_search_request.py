# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
from typing import Union

from domain.base_request import BaseRequest
from PIL import Image
from pydantic import field_validator


class ImageSearchRequest(BaseRequest):
    # Base64-encoded image
    prompt: Union[str, bytes]

    # Number of top predictions to return
    top_k: int = 3

    # Response format: "json" or "verbose"
    response_format: str = "json"

    # Minimum confidence threshold
    min_confidence: float = 70.0

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

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        if v not in ["json", "verbose_json"]:
            raise ValueError("response_format must be 'json' or 'verbose_json'")
        return v

    @field_validator("min_confidence")
    @classmethod
    def validate_min_confidence(cls, v):
        if v < 0.0 or v > 100.0:
            raise ValueError("min_confidence must be between 0.0 and 100.0")
        return v

    def get_pil_image(self) -> Image.Image:
        """Get the PIL image from the base64 data."""
        if isinstance(self.prompt, bytes):
            return Image.open(BytesIO(self.prompt))
        prompt = base64.b64decode(self.prompt)
        return Image.open(BytesIO(prompt))
