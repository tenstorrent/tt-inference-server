# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config.constants import JobTypes


class TestTrainingServiceCreateJob:
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with dataset configuration"""
        settings = MagicMock()
        settings.model_runner = "training-gemma-lora"
        settings.device = "p150"
        settings.download_weights_from_service = False
        return settings

    @pytest.fixture
    def mock_request(self):
        """Create mock training request"""
        request = MagicMock()
        request._task_id = "unique_task_123"
        request._output_model_path = None
        return request

    @pytest.mark.asyncio
    async def test_create_job_sets_dataset_from_settings(
        self, mock_settings, mock_request
    ):
        """Test TrainingService.create_job sets correct dataset and model path"""
        with patch(
            "model_services.training_service.get_settings", return_value=mock_settings
        ), patch("model_services.base_service.get_scheduler"), patch(
            "model_services.base_service.settings", mock_settings
        ), patch("model_services.base_job_service.settings", mock_settings), patch(
            "model_services.base_job_service.get_job_manager"
        ) as mock_jm, patch("model_services.base_service.TTLogger"), patch(
            "model_services.base_service.HuggingFaceUtils"
        ), patch("model_services.training_service.os.makedirs"):
            mock_jm.return_value.create_job = AsyncMock(return_value={"job_id": "test"})

            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            assert mock_request._output_model_path == "models_save/unique_task_123.pt"


class TestTrainingServiceGetJobMetrics:
    @pytest.fixture
    def service(self):
        from model_services.training_service import TrainingService

        # Skip __init__ side effects — only _job_manager is needed for get_job_metrics
        svc = object.__new__(TrainingService)
        svc._job_manager = MagicMock()
        return svc

    def test_after_slices_correctly(self, service):
        metrics = [
            {"global_step": 1, "metric_name": "loss", "value": 0.5},
            {"global_step": 2, "metric_name": "loss", "value": 0.4},
            {"global_step": 3, "metric_name": "loss", "value": 0.3},
        ]
        service._job_manager.get_job_metrics.return_value = metrics

        result_all = service.get_job_metrics("job-1", after=0)
        result_after_2 = service.get_job_metrics("job-1", after=2)
        result_after_10 = service.get_job_metrics("job-1", after=10)
        assert result_all == metrics
        assert result_after_2 == [
            {"global_step": 3, "metric_name": "loss", "value": 0.3}
        ]
        assert result_after_10 == []
