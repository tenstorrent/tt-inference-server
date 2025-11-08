# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Tests for TT-Comfy Bridge Server.
"""

import pytest
import asyncio
import socket as sock
import struct
import msgpack
from pathlib import Path


@pytest.fixture
async def test_socket_path(tmp_path):
    """Create a temporary socket path for testing."""
    socket_path = tmp_path / "test-tt-comfy.sock"
    yield str(socket_path)
    # Cleanup
    if socket_path.exists():
        socket_path.unlink()


@pytest.mark.asyncio
async def test_server_starts_and_stops():
    """Test that server can start and stop cleanly."""
    from server.main import TTComfyBridgeServer
    from server.config import BridgeConfig
    
    config = BridgeConfig(socket_path="/tmp/test-tt-comfy.sock")
    server = TTComfyBridgeServer(config)
    
    # Start server in background
    server_task = asyncio.create_task(server.start())
    
    # Wait a bit for server to initialize
    await asyncio.sleep(0.5)
    
    # Check socket exists
    assert Path(config.socket_path).exists()
    
    # Shutdown
    await server.shutdown()
    server_task.cancel()
    
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_ping_operation():
    """Test ping operation for health check."""
    from server.model_registry import ModelRegistry
    from server.operations import OperationHandler
    from protocol.messages import OperationType
    
    registry = ModelRegistry()
    handler = OperationHandler(registry)
    
    response = await handler.handle_operation(
        operation=OperationType.PING.value,
        data={},
        request_id="test-ping-1"
    )
    
    assert response.status == "success"
    assert response.data["status"] == "pong"
    assert response.request_id == "test-ping-1"


def test_model_registry():
    """Test model registry functionality."""
    from server.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    
    # Test registration
    mock_model = {"name": "test_model"}
    registry.register("test-model-1", mock_model)
    
    assert registry.exists("test-model-1")
    assert registry.get("test-model-1") == mock_model
    assert "test-model-1" in registry.list_models()
    
    # Test unregistration
    registry.unregister("test-model-1")
    assert not registry.exists("test-model-1")
    assert registry.get("test-model-1") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

