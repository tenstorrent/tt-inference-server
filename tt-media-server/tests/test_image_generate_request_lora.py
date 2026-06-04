# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import pytest
from domain.image_generate_request import ImageGenerateRequest
from pydantic import ValidationError


class TestImageGenerateRequestLoraFields:
    def test_lora_path_defaults_to_none(self):
        req = ImageGenerateRequest(prompt="a cat", guidance_scale=5.0)
        assert req.lora_path is None

    def test_lora_scale_defaults_to_half(self):
        req = ImageGenerateRequest(prompt="a cat", guidance_scale=5.0)
        assert req.lora_scale == 0.5

    def test_accepts_lora_fields(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            guidance_scale=5.0,
            lora_path="/some/path.safetensors",
            lora_scale=0.8,
        )
        assert req.lora_path == "/some/path.safetensors"
        assert req.lora_scale == 0.8

    def test_lora_scale_lower_bound(self):
        req = ImageGenerateRequest(prompt="a cat", guidance_scale=5.0, lora_scale=0.0)
        assert req.lora_scale == 0.0

    def test_lora_scale_upper_bound(self):
        req = ImageGenerateRequest(prompt="a cat", guidance_scale=5.0, lora_scale=2.0)
        assert req.lora_scale == 2.0

    def test_lora_scale_below_range_raises(self):
        with pytest.raises(ValidationError):
            ImageGenerateRequest(prompt="a cat", guidance_scale=5.0, lora_scale=-0.1)

    def test_lora_scale_above_range_raises(self):
        with pytest.raises(ValidationError):
            ImageGenerateRequest(prompt="a cat", guidance_scale=5.0, lora_scale=2.5)
