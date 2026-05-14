# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for num_inference_steps validation in ImageGenerateRequest.

Validates that the field_validator enforces per-model minimum inference steps:
- Flux models (tt-flux.1-dev, tt-flux.1-schnell): min 4 steps
- All other image models (SDXL, SD3.5, Motif, etc.): min 12 steps
"""

import sys
from unittest.mock import MagicMock

import pytest

# Mock config.settings before importing ImageGenerateRequest,
# since config.settings depends on pydantic_settings which is not
# available in the local test environment.
_mock_settings_module = MagicMock()
_mock_settings_instance = MagicMock()
_mock_settings_instance.model_runner = "tt-sdxl-trace"  # default
_mock_settings_module.get_settings.return_value = _mock_settings_instance
sys.modules.setdefault("config", MagicMock())
sys.modules["config.settings"] = _mock_settings_module

# Now safe to add tt-media-server to path and import
sys.path.insert(0, "tt-media-server")
from domain.image_generate_request import ImageGenerateRequest  # noqa: E402


def _set_model_runner(runner: str):
    """Set the mocked model_runner for subsequent requests."""
    _mock_settings_instance.model_runner = runner


def _make_request(num_inference_steps: int):
    """Create an ImageGenerateRequest with given inference steps."""
    return ImageGenerateRequest(
        prompt="test",
        guidance_scale=8.0,
        num_inference_steps=num_inference_steps,
    )


class TestFluxInferenceStepsValidation:
    """Flux models should accept num_inference_steps >= 4."""

    def setup_method(self):
        _set_model_runner("tt-flux.1-dev")

    def test_accepts_4_steps(self):
        req = _make_request(4)
        assert req.num_inference_steps == 4

    def test_accepts_4_steps_schnell(self):
        _set_model_runner("tt-flux.1-schnell")
        req = _make_request(4)
        assert req.num_inference_steps == 4

    def test_accepts_20_steps(self):
        req = _make_request(20)
        assert req.num_inference_steps == 20

    def test_accepts_50_steps(self):
        req = _make_request(50)
        assert req.num_inference_steps == 50

    def test_rejects_3_steps(self):
        """Rejected by Pydantic ge=4 before field_validator runs."""
        with pytest.raises(Exception):
            _make_request(3)

    def test_rejects_1_step(self):
        """Rejected by Pydantic ge=4 before field_validator runs."""
        _set_model_runner("tt-flux.1-schnell")
        with pytest.raises(Exception):
            _make_request(1)


class TestSD35InferenceStepsValidation:
    """SD3.5 should reject num_inference_steps < 12."""

    def setup_method(self):
        _set_model_runner("tt-sd3.5")

    def test_accepts_12_steps(self):
        req = _make_request(12)
        assert req.num_inference_steps == 12

    def test_accepts_20_steps(self):
        req = _make_request(20)
        assert req.num_inference_steps == 20

    def test_rejects_4_steps(self):
        with pytest.raises(Exception, match="num_inference_steps must be >= 12"):
            _make_request(4)

    def test_rejects_11_steps(self):
        with pytest.raises(Exception, match="num_inference_steps must be >= 12"):
            _make_request(11)


class TestMotifInferenceStepsValidation:
    """Motif should reject num_inference_steps < 12."""

    def setup_method(self):
        _set_model_runner("tt-motif-image-6b-preview")

    def test_accepts_12_steps(self):
        req = _make_request(12)
        assert req.num_inference_steps == 12

    def test_rejects_4_steps(self):
        with pytest.raises(Exception, match="num_inference_steps must be >= 12"):
            _make_request(4)


class TestSDXLInferenceStepsValidation:
    """SDXL should reject num_inference_steps < 12."""

    def setup_method(self):
        _set_model_runner("tt-sdxl-trace")

    def test_accepts_12_steps(self):
        req = _make_request(12)
        assert req.num_inference_steps == 12

    def test_accepts_20_steps(self):
        req = _make_request(20)
        assert req.num_inference_steps == 20

    def test_rejects_4_steps(self):
        with pytest.raises(Exception, match="num_inference_steps must be >= 12"):
            _make_request(4)

    def test_rejects_11_steps(self):
        with pytest.raises(Exception, match="num_inference_steps must be >= 12"):
            _make_request(11)


class TestPydanticBoundaryValidation:
    """Test absolute Pydantic boundaries (ge=4, le=50)."""

    def setup_method(self):
        _set_model_runner("tt-flux.1-dev")

    def test_pydantic_rejects_3_even_for_flux(self):
        """ge=4 in Field rejects 3 before field_validator even runs."""
        with pytest.raises(Exception):
            _make_request(3)

    def test_pydantic_rejects_51(self):
        """le=50 in Field rejects 51."""
        with pytest.raises(Exception):
            _make_request(51)

    def test_pydantic_accepts_50(self):
        req = _make_request(50)
        assert req.num_inference_steps == 50
