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
    """The dataset path is client input, so submission is where it gets vetted."""

    @pytest.fixture
    def submit(self, client, mock_service, tmp_path):
        settings = MagicMock()
        settings.device = "p150"
        settings.training_datasets_dir = str(tmp_path)
        mock_service.create_job = AsyncMock(return_value={"id": "job-1"})

        def _submit(**overrides):
            body = {
                "device_type": "p150",
                "dataset_loader": DatasetLoaders.CUSTOM.value,
            }
            body.update(overrides)
            with patch(
                "open_ai_api.fine_tuning.get_settings", return_value=settings
            ), patch(
                "utils.dataset_path_resolver.get_settings", return_value=settings
            ):
                return client.post("/jobs", json=body)

        return _submit

    def test_accepts_a_path_inside_the_dataset_directory(self, submit, tmp_path):
        (tmp_path / "train.json").write_text("{}")

        response = submit(train_dataset_path="train.json")

        assert response.status_code == 201

    def test_stores_the_resolved_path(self, submit, mock_service, tmp_path):
        (tmp_path / "train.json").write_text("{}")
        submit(train_dataset_path="train.json")

        # The worker opens this, so it must be the resolved path rather than
        # whatever relative string the client sent.
        request = mock_service.create_job.call_args.args[1]
        assert request.train_dataset_path == str(tmp_path / "train.json")

    def test_rejects_a_path_outside_the_dataset_directory(self, submit, tmp_path):
        (tmp_path.parent / "secret.json").write_text("not yours")

        response = submit(train_dataset_path="../secret.json")

        assert response.status_code == 400
        assert "resolves outside" in response.json()["detail"]

    def test_rejects_a_missing_file(self, submit):
        response = submit(train_dataset_path="absent.json")

        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    def test_rejects_a_custom_dataset_without_a_path(self, submit):
        assert submit().status_code == 422


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
