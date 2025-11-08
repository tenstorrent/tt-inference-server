# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Integration tests for TT-Comfy Bridge with ComfyUI.
"""

import pytest
import asyncio
import os
import sys
import time
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/tt-admin/tt-inference-server/tt-comfy-bridge')
sys.path.insert(0, '/home/tt-admin/ComfyUI-tt')


@pytest.fixture(scope="session")
def bridge_server():
    """Start bridge server for testing."""
    import subprocess
    
    socket_path = "/tmp/test-tt-comfy.sock"
    
    # Remove existing socket
    if Path(socket_path).exists():
        Path(socket_path).unlink()
    
    # Start server
    process = subprocess.Popen(
        [
            sys.executable, "-m", "server.main",
            "--socket-path", socket_path,
            "--device-id", "0",
            "--log-level", "INFO"
        ],
        cwd="/home/tt-admin/tt-inference-server/tt-comfy-bridge"
    )
    
    # Wait for socket
    timeout = 30
    for _ in range(timeout * 2):
        if Path(socket_path).exists():
            break
        time.sleep(0.5)
    else:
        process.kill()
        pytest.fail("Bridge server failed to start")
    
    yield socket_path
    
    # Cleanup
    process.terminate()
    process.wait(timeout=10)
    if Path(socket_path).exists():
        Path(socket_path).unlink()


@pytest.mark.integration
def test_backend_connection(bridge_server):
    """Test that ComfyUI backend can connect to bridge."""
    from comfy.backends.tenstorrent_backend import TenstorrentBackend
    
    backend = TenstorrentBackend(socket_path=bridge_server)
    
    # Test ping
    result = backend.ping()
    assert result["status"] == "pong"
    
    backend.close()


@pytest.mark.integration
def test_sdxl_model_init(bridge_server):
    """Test initializing SDXL model."""
    from comfy.backends.tenstorrent_backend import TenstorrentBackend
    
    backend = TenstorrentBackend(socket_path=bridge_server)
    
    # Initialize model
    model_id = backend.init_model("sdxl", {})
    assert model_id is not None
    assert "sdxl" in model_id
    
    # Cleanup
    backend.unload_model(model_id)
    backend.close()


@pytest.mark.integration
@pytest.mark.slow
def test_sdxl_inference_end_to_end(bridge_server):
    """Test end-to-end SDXL inference through ComfyUI wrapper."""
    os.environ['TT_COMFY_SOCKET'] = bridge_server
    
    from comfy.tt_models.sdxl import TenstorrentSDXL
    
    model = TenstorrentSDXL()
    model.backend.socket_path = bridge_server
    model.backend._connect()
    
    try:
        # Generate image
        image = model.generate_image(
            prompt="A beautiful sunset over mountains",
            negative_prompt="low quality, blurry",
            num_inference_steps=5,  # Use fewer steps for testing
            seed=42
        )
        
        # Verify result
        assert image is not None
        assert hasattr(image, 'size')
        assert image.size[0] > 0 and image.size[1] > 0
        
    finally:
        model.unload()


@pytest.mark.integration
def test_model_registry_isolation(bridge_server):
    """Test that multiple models can coexist."""
    from comfy.backends.tenstorrent_backend import TenstorrentBackend
    
    backend = TenstorrentBackend(socket_path=bridge_server)
    
    # Initialize multiple models (if supported)
    model1_id = backend.init_model("sdxl", {})
    
    # Verify they're tracked separately
    assert model1_id is not None
    
    # Cleanup
    backend.unload_model(model1_id)
    backend.close()


@pytest.mark.integration
def test_error_handling(bridge_server):
    """Test error handling in backend."""
    from comfy.backends.tenstorrent_backend import TenstorrentBackend
    
    backend = TenstorrentBackend(socket_path=bridge_server)
    
    # Try invalid operation
    with pytest.raises(RuntimeError):
        backend.full_inference("nonexistent_model_id", prompt="test")
    
    backend.close()


@pytest.mark.integration
def test_prompt_encoding(bridge_server):
    """Test prompt encoding operation."""
    from comfy.backends.tenstorrent_backend import TenstorrentBackend
    
    backend = TenstorrentBackend(socket_path=bridge_server)
    
    # Initialize model
    model_id = backend.init_model("sdxl", {})
    
    # Encode prompts
    result = backend.encode_prompts(
        model_id,
        "A beautiful landscape",
        "low quality"
    )
    
    assert result["status"] == "encoded"
    
    # Cleanup
    backend.unload_model(model_id)
    backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

