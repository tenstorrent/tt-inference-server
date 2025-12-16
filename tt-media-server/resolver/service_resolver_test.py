# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import Mock
import sys

# Mock ALL problematic modules BEFORE any imports
sys.modules["ttnn"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_unet"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_embedding"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.sdxl_utility"] = Mock()
sys.modules["tt_model_runners.sdxl_runner"] = Mock()
sys.modules["model_services.scheduler"] = Mock()


# Create a mock ImageService class
class MockImageService:
    pass


# Mock the image_service module
mock_image_service = Mock()
mock_image_service.ImageService = MockImageService
sys.modules["model_services.image_service"] = mock_image_service

# Now we can safely import
from . import service_resolver
from model_services.base_service import BaseService


def setup_module(module):
    # Reset singletons before each test module
    service_resolver._service_holders = {}


def teardown_module(module):
    # Reset singletons after each test module
    service_resolver._service_holders = {}


def test_service_resolver_returns_image_service(monkeypatch):
    # Mock the settings directly instead of environment variables
    monkeypatch.setattr("resolver.service_resolver.settings.model_service", "image")
    # Reset singleton to ensure clean test
    service_resolver._service_holders = {}

    model = service_resolver.service_resolver()
    assert isinstance(model, MockImageService)
    # Should return the same instance (singleton)
    model2 = service_resolver.service_resolver()
    assert model is model2


def test_service_resolver_returns_base_service(monkeypatch):
    # Mock the settings directly instead of environment variables
    monkeypatch.setattr("resolver.service_resolver.settings.model_service", "OTHER")
    # Reset singleton to ensure clean test
    service_resolver._service_holders = {}

    model = service_resolver.service_resolver()
    assert isinstance(model, BaseService)
    # Should not be MockImageService
    assert not isinstance(model, MockImageService)
