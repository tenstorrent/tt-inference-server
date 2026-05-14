# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import sys
from unittest.mock import Mock

from config.constants import ModelServices

# Mock ALL problematic modules BEFORE any imports
sys.modules["ttnn"] = Mock()
sys.modules["models.demos.stable_diffusion_xl_base.tt.tt_unet"] = Mock()
sys.modules["models.demos.stable_diffusion_xl_base.tt.tt_embedding"] = Mock()
sys.modules["models.demos.stable_diffusion_xl_base.tt.sdxl_utility"] = Mock()
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
from model_services.base_service import BaseService
from resolver import service_resolver


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
    class MockLLMService(BaseService):
        def __init__(self):
            pass

    monkeypatch.setattr("resolver.service_resolver.settings.model_service", "llm")
    monkeypatch.setitem(
        service_resolver._SUPPORTED_MODEL_SERVICES,
        ModelServices.LLM,
        lambda: MockLLMService(),
    )
    # Reset singleton to ensure clean test
    service_resolver._service_holders = {}

    model = service_resolver.service_resolver()
    assert isinstance(model, BaseService)
    # Should not be MockImageService
    assert not isinstance(model, MockImageService)
