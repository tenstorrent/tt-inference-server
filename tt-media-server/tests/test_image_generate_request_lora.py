# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import pytest
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
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


class TestImageToImageRequestLoraFields:
    def test_inherits_lora_fields(self):
        req = ImageToImageRequest(
            prompt="a cat",
            guidance_scale=5.0,
            image="base64data",
            lora_path="/adapter.safetensors",
            lora_scale=1.2,
        )
        assert req.lora_path == "/adapter.safetensors"
        assert req.lora_scale == 1.2

    def test_defaults(self):
        req = ImageToImageRequest(
            prompt="a cat", guidance_scale=5.0, image="base64data"
        )
        assert req.lora_path is None
        assert req.lora_scale == 0.5


class TestImageEditRequestLoraFields:
    def test_inherits_lora_fields(self):
        req = ImageEditRequest(
            prompt="a cat",
            guidance_scale=5.0,
            image="base64data",
            mask="maskdata",
            lora_path="/adapter.safetensors",
            lora_scale=0.5,
        )
        assert req.lora_path == "/adapter.safetensors"
        assert req.lora_scale == 0.5

    def test_defaults(self):
        req = ImageEditRequest(
            prompt="a cat",
            guidance_scale=5.0,
            image="base64data",
            mask="maskdata",
        )
        assert req.lora_path is None
        assert req.lora_scale == 0.5
