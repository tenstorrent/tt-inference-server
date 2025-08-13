# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import Mock, patch
import sys

# Mock ALL problematic modules BEFORE any imports
sys.modules['ttnn'] = Mock()
sys.modules['models.experimental.stable_diffusion_xl_base.tt.tt_unet'] = Mock()
sys.modules['models.experimental.stable_diffusion_xl_base.tt.tt_embedding'] = Mock()
sys.modules['models.experimental.stable_diffusion_xl_base.tt.sdxl_utility'] = Mock()
sys.modules['tt_model_runners.sdxl_runner'] = Mock()
sys.modules['model_services.scheduler'] = Mock()

# Create a mock ImageService class
class MockImageService:
    pass

# Mock the image_service module
mock_image_service = Mock()
mock_image_service.ImageService = MockImageService
sys.modules['model_services.image_service'] = mock_image_service

# Now we can safely import
from . import model_resolver
from model_services.base_model import BaseModel

def setup_module(module):
    # Reset singletons before each test module
    model_resolver.current_model_holder = None

def teardown_module(module):
    # Reset singletons after each test module
    model_resolver.current_model_holder = None

def test_model_resolver_returns_image_service(monkeypatch):
    # Mock the settings directly instead of environment variables
    monkeypatch.setattr('resolver.model_resolver.settings.model_service', 'image')
    # Reset singleton to ensure clean test
    model_resolver.current_model_holder = None
    
    model = model_resolver.model_resolver()
    assert isinstance(model, MockImageService)
    # Should return the same instance (singleton)
    model2 = model_resolver.model_resolver()
    assert model is model2

def test_model_resolver_returns_base_model(monkeypatch):
    # Mock the settings directly instead of environment variables
    monkeypatch.setattr('resolver.model_resolver.settings.model_service', 'OTHER')
    # Reset singleton to ensure clean test
    model_resolver.current_model_holder = None
    
    model = model_resolver.model_resolver()
    assert isinstance(model, BaseModel)
    # Should not be MockImageService
    assert not isinstance(model, MockImageService)