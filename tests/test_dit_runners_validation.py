# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for MIN_INFERENCE_STEPS validation in DiT runners.

Tests the _validate_inference_steps logic without importing the actual
runner classes (which require TT hardware dependencies).
"""

import sys
from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class MockRequest:
    """Minimal mock of ImageGenerateRequest for validation testing."""

    prompt: str = "test"
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = 20
    seed: int = 0
    guidance_scale: float = 8.0


def _make_validate_fn(min_steps: int):
    """Create a validate function matching TTDiTRunner._validate_inference_steps."""

    def validate(request):
        if request.num_inference_steps < min_steps:
            raise ValueError(
                f"num_inference_steps must be >= {min_steps}, "
                f"got {request.num_inference_steps}"
            )

    return validate


class TestFlux1InferenceStepsValidation:
    """Flux.1 (dev and schnell) should accept num_inference_steps >= 4."""

    validate = staticmethod(_make_validate_fn(4))

    def test_accepts_4_steps(self):
        self.validate(MockRequest(num_inference_steps=4))

    def test_accepts_20_steps(self):
        self.validate(MockRequest(num_inference_steps=20))

    def test_accepts_50_steps(self):
        self.validate(MockRequest(num_inference_steps=50))

    def test_rejects_3_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps must be >= 4"):
            self.validate(MockRequest(num_inference_steps=3))

    def test_rejects_1_step(self):
        with pytest.raises(ValueError, match="num_inference_steps must be >= 4"):
            self.validate(MockRequest(num_inference_steps=1))


class TestSD35InferenceStepsValidation:
    """SD3.5 should reject num_inference_steps < 12."""

    validate = staticmethod(_make_validate_fn(12))

    def test_accepts_12_steps(self):
        self.validate(MockRequest(num_inference_steps=12))

    def test_accepts_20_steps(self):
        self.validate(MockRequest(num_inference_steps=20))

    def test_rejects_4_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps must be >= 12"):
            self.validate(MockRequest(num_inference_steps=4))

    def test_rejects_11_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps must be >= 12"):
            self.validate(MockRequest(num_inference_steps=11))


class TestMotifInferenceStepsValidation:
    """Motif should reject num_inference_steps < 12 (same as base default)."""

    validate = staticmethod(_make_validate_fn(12))

    def test_accepts_12_steps(self):
        self.validate(MockRequest(num_inference_steps=12))

    def test_rejects_4_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps must be >= 12"):
            self.validate(MockRequest(num_inference_steps=4))


class TestImageGenerateRequestPydanticValidation:
    """Test that Pydantic allows num_inference_steps=4 after ge=4 change."""

    def test_pydantic_accepts_4(self):
        sys.path.insert(0, "tt-media-server")
        from domain.image_generate_request import ImageGenerateRequest

        req = ImageGenerateRequest(
            prompt="test", guidance_scale=8.0, num_inference_steps=4
        )
        assert req.num_inference_steps == 4

    def test_pydantic_rejects_3(self):
        sys.path.insert(0, "tt-media-server")
        from domain.image_generate_request import ImageGenerateRequest

        with pytest.raises(Exception):  # Pydantic ValidationError
            ImageGenerateRequest(
                prompt="test", guidance_scale=8.0, num_inference_steps=3
            )

    def test_pydantic_accepts_12(self):
        sys.path.insert(0, "tt-media-server")
        from domain.image_generate_request import ImageGenerateRequest

        req = ImageGenerateRequest(
            prompt="test", guidance_scale=8.0, num_inference_steps=12
        )
        assert req.num_inference_steps == 12

    def test_pydantic_accepts_50(self):
        sys.path.insert(0, "tt-media-server")
        from domain.image_generate_request import ImageGenerateRequest

        req = ImageGenerateRequest(
            prompt="test", guidance_scale=8.0, num_inference_steps=50
        )
        assert req.num_inference_steps == 50

    def test_pydantic_rejects_51(self):
        sys.path.insert(0, "tt-media-server")
        from domain.image_generate_request import ImageGenerateRequest

        with pytest.raises(Exception):  # Pydantic ValidationError
            ImageGenerateRequest(
                prompt="test", guidance_scale=8.0, num_inference_steps=51
            )
