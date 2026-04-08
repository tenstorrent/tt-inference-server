# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api.tt_maintenance_api import router
from resolver.service_resolver import service_resolver


@pytest.fixture
def mock_service():
    """Create a mock service for testing"""
    service = Mock()
    service.check_is_model_ready = Mock(
        return_value={
            "model_ready": True,
            "queue_size": 2,
            "max_queue_size": 10,
            "device_mesh_shape": "(1,1)",
            "device": "metal",
            "worker_info": {"worker_0": "ready"},
            "runner_in_use": "vllm",
        }
    )
    service.deep_reset = AsyncMock(return_value=True)
    service.device_reset = AsyncMock(return_value=True)
    return service


@pytest.fixture
def test_client(mock_service):
    """Create a test client with mocked service dependency"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[service_resolver] = lambda: mock_service
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for the /health endpoint"""

    def test_health_returns_empty_dict_when_model_ready(
        self, test_client, mock_service
    ):
        """Health endpoint returns empty dict with 200 when model is ready"""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {}
        mock_service.check_is_model_ready.assert_called_once()

    def test_health_returns_503_when_model_not_ready(self, test_client, mock_service):
        """Health endpoint returns 503 when model is not ready"""
        mock_service.check_is_model_ready.return_value = {"model_ready": False}

        response = test_client.get("/health")

        assert response.status_code == 503
        assert response.json()["detail"] == "Model not ready"

    def test_health_returns_503_when_model_ready_key_missing(
        self, test_client, mock_service
    ):
        """Health endpoint returns 503 when model_ready key is missing"""
        mock_service.check_is_model_ready.return_value = {}

        response = test_client.get("/health")

        assert response.status_code == 503
        assert response.json()["detail"] == "Model not ready"

    def test_health_returns_500_on_exception(self, test_client, mock_service):
        """Health endpoint returns 500 when check_is_model_ready raises exception"""
        mock_service.check_is_model_ready.side_effect = RuntimeError(
            "Scheduler not available"
        )

        response = test_client.get("/health")

        assert response.status_code == 500
        assert "Health check failed" in response.json()["detail"]
        assert "Scheduler not available" in response.json()["detail"]


class TestLivenessEndpoint:
    """Test cases for the /tt-liveness endpoint"""

    def test_liveness_returns_status_with_model_info(self, test_client, mock_service):
        """Liveness endpoint returns alive status with model info"""
        response = test_client.get("/tt-liveness")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["model_ready"] is True
        assert data["queue_size"] == 2
        assert data["device"] == "metal"

    def test_liveness_returns_500_on_exception(self, test_client, mock_service):
        """Liveness endpoint returns 500 when service check fails"""
        mock_service.check_is_model_ready.side_effect = RuntimeError("Service error")

        response = test_client.get("/tt-liveness")

        assert response.status_code == 500
        assert "Liveness check failed" in response.json()["detail"]


class TestDeepResetEndpoint:
    """Test cases for the /tt-deep-reset endpoint"""

    def test_deep_reset_schedules_reset(self, test_client, mock_service):
        """Deep reset endpoint schedules reset and returns status"""
        response = test_client.post("/tt-deep-reset")

        assert response.status_code == 200
        assert response.json() == {"status": "Reset scheduled"}
        mock_service.deep_reset.assert_called_once()

    def test_deep_reset_returns_500_on_exception(self, test_client, mock_service):
        """Deep reset endpoint returns 500 when reset fails"""
        mock_service.deep_reset.side_effect = RuntimeError("Reset failed")

        response = test_client.post("/tt-deep-reset")

        assert response.status_code == 500
        assert "Reset failed" in response.json()["detail"]


class TestResetDeviceEndpoint:
    """Test cases for the /tt-reset-device endpoint"""

    def test_reset_device_schedules_reset(self, test_client, mock_service):
        """Reset device endpoint schedules device reset"""
        response = test_client.post("/tt-reset-device?device_id=0")

        assert response.status_code == 200
        assert response.json() == {"status": "Reset of device 0 scheduled"}
        mock_service.device_reset.assert_called_once_with("0")

    def test_reset_device_with_different_device_id(self, test_client, mock_service):
        """Reset device endpoint handles different device IDs"""
        response = test_client.post("/tt-reset-device?device_id=device_3")

        assert response.status_code == 200
        assert response.json() == {"status": "Reset of device device_3 scheduled"}
        mock_service.device_reset.assert_called_once_with("device_3")

    def test_reset_device_returns_500_on_exception(self, test_client, mock_service):
        """Reset device endpoint returns 500 when reset fails"""
        mock_service.device_reset.side_effect = RuntimeError("Device reset failed")

        response = test_client.post("/tt-reset-device?device_id=0")

        assert response.status_code == 500
        assert "Reset failed" in response.json()["detail"]

    def test_reset_device_requires_device_id(self, test_client, mock_service):
        """Reset device endpoint requires device_id parameter"""
        response = test_client.post("/tt-reset-device")

        assert response.status_code == 422  # Unprocessable Entity
