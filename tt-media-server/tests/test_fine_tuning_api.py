# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.constants import DatasetLoaders
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api.fine_tuning import router
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from security.org_id_checker import get_org_id


@pytest.fixture
def mock_service():
    return MagicMock()


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[service_resolver] = lambda: mock_service
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    app.dependency_overrides[get_org_id] = lambda: "test-org"
    return TestClient(app)


class TestSubmitCustomDatasetJob:
    @pytest.fixture
    def submit(self, client, mock_service):
        settings = MagicMock()
        settings.device = "p150"
        mock_service.create_job = AsyncMock(return_value={"id": "job-1"})

        def _submit(**overrides):
            body = {
                "device_type": "p150",
                "dataset_loader": DatasetLoaders.CUSTOM.value,
            }
            body.update(overrides)
            with patch("open_ai_api.fine_tuning.get_settings", return_value=settings):
                return client.post("/jobs", json=body)

        return _submit

    def test_accepts_a_custom_dataset_path_as_given(self, submit, mock_service):
        response = submit(
            train_dataset_path="/datasets/train.json",
            file_type="json",
        )

        assert response.status_code == 201
        request = mock_service.create_job.call_args.args[1]
        assert request.train_dataset_path == "/datasets/train.json"
        assert request.file_type == "json"

    def test_rejects_a_custom_dataset_without_a_path(self, submit):
        assert submit(file_type="json").status_code == 422

    def test_rejects_a_custom_dataset_without_a_file_type(self, submit):
        assert submit(train_dataset_path="/datasets/train.json").status_code == 422


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
