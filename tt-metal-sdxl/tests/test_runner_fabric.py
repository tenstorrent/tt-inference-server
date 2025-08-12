import pytest
import ttnn  # Ensure this module is imported to be patched
from config.settings import settings
from tt_model_runners.runner_fabric import get_device_runner

# Automatically patch ttnn.get_arch_name to avoid errors during test collection.
@pytest.fixture(autouse=True)
def patch_ttnn_get_arch_name(monkeypatch):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "default_arch")

def test_tt_sdxl_runner(monkeypatch):
    monkeypatch.setattr(settings, "model_runner", "tt-sdxl")
    runner = get_device_runner("test_worker")
    assert "TTSDXLRunner" in type(runner).__name__

def test_tt_sd35_runner(monkeypatch):
    monkeypatch.setattr(settings, "model_runner", "tt-sd3.5")
    runner = get_device_runner("test_worker")
    assert "TTSD35Runner" in type(runner).__name__

def test_forge_runner(monkeypatch):
    monkeypatch.setattr(settings, "model_runner", "forge")
    runner = get_device_runner("test_worker")
    assert "ForgeRunner" in type(runner).__name__
    
def test_mock_runner(monkeypatch):
    monkeypatch.setattr(settings, "model_runner", "mock")
    runner = get_device_runner("test_worker")
    assert "MockRunner" in type(runner).__name__

def test_invalid_runner(monkeypatch):
    monkeypatch.setattr(settings, "model_runner", "invalid")
    with pytest.raises(ValueError):
        get_device_runner("test_worker")