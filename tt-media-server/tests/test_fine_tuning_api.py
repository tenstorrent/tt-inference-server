# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.constants import DatasetLoaders, JobTypes
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
            template="alpaca",
        )

        assert response.status_code == 201
        request = mock_service.create_job.call_args.args[1]
        assert request.train_dataset_path == "/datasets/train.json"
        assert request.file_type == "json"
        assert request.template == "alpaca"

    def test_rejects_a_custom_dataset_without_a_path(self, submit):
        assert submit(file_type="json", template="alpaca").status_code == 422

    def test_rejects_a_custom_dataset_without_a_file_type(self, submit):
        assert (
            submit(
                train_dataset_path="/datasets/train.json", template="alpaca"
            ).status_code
            == 422
        )

    def test_rejects_a_custom_dataset_without_a_template(self, submit):
        assert (
            submit(
                train_dataset_path="/datasets/train.json", file_type="json"
            ).status_code
            == 422
        )


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


class TestJobRetention:
    def test_retain_job(self, client, mock_service):
        mock_service.get_job_metadata.return_value = {
            "id": "job-1",
            "job_type": JobTypes.TRAINING.value,
        }
        mock_service.set_job_retained.return_value = {
            "id": "job-1",
            "retained": True,
        }

        response = client.put("/jobs/job-1/retention", json={"retained": True})

        assert response.status_code == 200
        assert response.json()["retained"] is True
        mock_service.set_job_retained.assert_called_once_with(
            "job-1",
            True,
            org_id="test-org",
        )

    def test_retain_missing_job_returns_404(self, client, mock_service):
        mock_service.get_job_metadata.return_value = None
        mock_service.set_job_retained.return_value = None

        response = client.put("/jobs/missing/retention", json={"retained": True})

        assert response.status_code == 404

    def test_retain_adapter_merge_job(self, client, mock_service):
        mock_service.get_job_metadata.return_value = {
            "id": "merge-1",
            "job_type": JobTypes.ADAPTER_MERGE.value,
        }
        mock_service.set_job_retained.return_value = {
            "id": "merge-1",
            "retained": True,
        }

        response = client.put("/jobs/merge-1/retention", json={"retained": True})

        assert response.status_code == 200
        assert response.json()["retained"] is True
        mock_service.set_job_retained.assert_called_once_with(
            "merge-1",
            True,
            org_id="test-org",
        )


class TestDeleteJob:
    def test_delete_terminal_job(self, client, mock_service):
        mock_service.get_job_metadata.return_value = {
            "id": "job-1",
            "job_type": JobTypes.TRAINING.value,
        }
        mock_service.delete_job.return_value = True

        response = client.delete("/jobs/job-1")

        assert response.status_code == 204
        mock_service.delete_job.assert_called_once_with("job-1", org_id="test-org")

    def test_delete_active_job_returns_conflict(self, client, mock_service):
        mock_service.get_job_metadata.return_value = {
            "id": "job-1",
            "job_type": JobTypes.TRAINING.value,
        }
        mock_service.delete_job.side_effect = ValueError(
            "Only terminal jobs can be deleted"
        )

        response = client.delete("/jobs/job-1")

        assert response.status_code == 409

    def test_delete_missing_job_returns_404(self, client, mock_service):
        mock_service.get_job_metadata.return_value = None
        mock_service.delete_job.return_value = False

        response = client.delete("/jobs/missing")

        assert response.status_code == 404

    def test_delete_adapter_merge_job(self, client, mock_service):
        mock_service.get_job_metadata.return_value = {
            "id": "merge-1",
            "job_type": JobTypes.ADAPTER_MERGE.value,
        }
        mock_service.delete_job.return_value = True

        response = client.delete("/jobs/merge-1")

        assert response.status_code == 204
        mock_service.delete_job.assert_called_once_with("merge-1", org_id="test-org")
