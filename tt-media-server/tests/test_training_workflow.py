# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config.constants import (
    DeviceTypes,
    JobTypes,
    ModelRunners,
    TRAINING_RUNNER_SUPPORTED_DEVICES,
    TrainingMeshShapes,
)


def _training_service_patches(mock_settings):
    """Stack the common patches needed to instantiate TrainingService."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch(
            "model_services.training_service.get_settings",
            return_value=mock_settings,
        )
    )
    stack.enter_context(patch("model_services.base_service.get_scheduler"))
    stack.enter_context(patch("model_services.base_service.settings", mock_settings))
    stack.enter_context(
        patch("model_services.base_job_service.settings", mock_settings)
    )
    mock_jm = stack.enter_context(
        patch("model_services.base_job_service.get_job_manager")
    )
    stack.enter_context(patch("model_services.base_service.TTLogger"))
    stack.enter_context(patch("model_services.base_service.HuggingFaceUtils"))
    stack.enter_context(patch("model_services.training_service.os.makedirs"))
    mock_jm.return_value.create_job = AsyncMock(return_value={"job_id": "test"})
    return stack, mock_jm


class TestGemmaTrainingServiceCreateJob:
    """Tests for TrainingService.create_job with Gemma LoRA config."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.model_runner = "training-gemma-lora"
        settings.device = "p150"
        settings.download_weights_from_service = False
        return settings

    @pytest.fixture
    def mock_request(self):
        request = MagicMock()
        request._task_id = "unique_task_123"
        request._output_model_path = None
        return request

    @pytest.mark.asyncio
    async def test_create_job_sets_output_model_path(self, mock_settings, mock_request):
        """Test TrainingService.create_job sets correct model path for Gemma"""
        stack, _ = _training_service_patches(mock_settings)
        with stack:
            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            assert mock_request._output_model_path == "model_store/unique_task_123"


class TestLlamaTrainingServiceCreateJob:
    """Tests for TrainingService.create_job with Llama LoRA multichip config."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.model_runner = "training-llama-lora"
        settings.device = "p300"
        settings.download_weights_from_service = False
        return settings

    @pytest.fixture
    def mock_request(self):
        request = MagicMock()
        request._task_id = "llama_task_456"
        request._output_model_path = None
        return request

    @pytest.mark.asyncio
    async def test_create_job_sets_output_model_path(self, mock_settings, mock_request):
        stack, _ = _training_service_patches(mock_settings)
        with stack:
            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            assert mock_request._output_model_path == "model_store/llama_task_456"

    @pytest.mark.asyncio
    async def test_create_job_sets_device_type_from_settings(
        self, mock_settings, mock_request
    ):
        stack, _ = _training_service_patches(mock_settings)
        with stack:
            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            assert mock_request.device_type == "p300"

    @pytest.mark.asyncio
    async def test_create_job_resolves_llama_model_name(
        self, mock_settings, mock_request
    ):
        stack, mock_jm = _training_service_patches(mock_settings)
        with stack:
            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            _, kwargs = mock_jm.return_value.create_job.call_args
            assert kwargs["model"] == "Llama-3.1-8B"

    @pytest.mark.asyncio
    async def test_create_job_passes_events_and_metrics(
        self, mock_settings, mock_request
    ):
        stack, mock_jm = _training_service_patches(mock_settings)
        with stack:
            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            _, kwargs = mock_jm.return_value.create_job.call_args
            assert kwargs["start_event"] is not None
            assert kwargs["cancel_event"] is not None
            assert kwargs["job_metrics"] is not None
            assert kwargs["job_logs"] is not None
            assert kwargs["result_path"] == "model_store/llama_task_456"


class TestLlamaRunnerSupportedDevices:
    """Tests for Llama LoRA runner device type configuration and validation."""

    def test_llama_runner_has_supported_devices(self):
        supported = TRAINING_RUNNER_SUPPORTED_DEVICES[ModelRunners.TRAINING_LLAMA_LORA]
        assert len(supported) > 0

    def test_llama_runner_supported_device_has_mesh_shape(self):
        for dt in TRAINING_RUNNER_SUPPORTED_DEVICES[ModelRunners.TRAINING_LLAMA_LORA]:
            assert dt.name in TrainingMeshShapes.__members__, (
                f"Device {dt.name} is supported for Llama training but has no "
                f"TrainingMeshShapes entry"
            )

    def test_llama_mesh_shapes_are_multichip(self):
        for dt in TRAINING_RUNNER_SUPPORTED_DEVICES[ModelRunners.TRAINING_LLAMA_LORA]:
            mesh = TrainingMeshShapes[dt.name].value
            num_devices = mesh[0] * mesh[1]
            assert num_devices >= 2, (
                f"Llama requires multichip, but {dt.name} mesh {mesh} "
                f"only has {num_devices} device(s)"
            )

    def test_p150_not_supported_for_llama(self):
        supported = TRAINING_RUNNER_SUPPORTED_DEVICES[ModelRunners.TRAINING_LLAMA_LORA]
        assert DeviceTypes.P150 not in supported


class TestGemmaTrainingServiceGetJobMetrics:
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

    def test_raises_value_error_when_job_not_found(self, service):
        service._job_manager.get_job_metrics.return_value = None
        with pytest.raises(ValueError, match="Job nonexistent not found"):
            service.get_job_metrics("nonexistent")
