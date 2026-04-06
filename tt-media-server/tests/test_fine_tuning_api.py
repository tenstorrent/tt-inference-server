# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api.fine_tuning import router
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key


@pytest.fixture
def mock_service():
    return MagicMock()


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[service_resolver] = lambda: mock_service
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    return TestClient(app)


class TestGetCatalog:
    def test_returns_catalog_json(self, client):
        mock_settings = MagicMock()
        mock_settings.model_runner = "training-gemma-lora"
        with patch("open_ai_api.fine_tuning.get_settings", return_value=mock_settings):
            response = client.get("/catalog")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "datasets" in data
        assert "trainers" in data
        assert "optimizers" in data
        assert "clusters" in data
        assert "supported" in data
