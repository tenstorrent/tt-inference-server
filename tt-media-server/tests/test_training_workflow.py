# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import re
import sys

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
        stack, mock_jm = _training_service_patches(mock_settings)
        with stack:
            from model_services.training_service import TrainingService

            service = TrainingService()
            await service.create_job(JobTypes.TRAINING, mock_request)

            assert mock_request._output_model_path == "models_save/unique_task_123.pt"


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

            assert mock_request._output_model_path == "models_save/llama_task_456.pt"

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
            assert kwargs["result_path"] == "models_save/llama_task_456.pt"


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


def _import_llama_runner():
    """Import the real TrainingLlamaLoraRunner, repairing sys.modules if needed.

    test_worker_utils.py replaces tt_model_runners.base_device_runner with a
    bare Mock() at module scope, which corrupts any later import that inherits
    from BaseDeviceRunner. Detect and repair this before importing.
    """
    import types

    bdr_key = "tt_model_runners.base_device_runner"
    runner_key = "tt_model_runners.forge_training_runners.training_llama_lora_runner"

    bdr_mod = sys.modules.get(bdr_key)
    if bdr_mod is not None and not isinstance(bdr_mod, types.ModuleType):
        repaired = types.ModuleType(bdr_key)
        repaired.BaseDeviceRunner = type("BaseDeviceRunner", (), {})
        sys.modules[bdr_key] = repaired
        sys.modules.pop(runner_key, None)

    from tt_model_runners.forge_training_runners.training_llama_lora_runner import (
        TrainingLlamaLoraRunner,
    )

    return TrainingLlamaLoraRunner


class TestLlamaRunnerConstants:
    """Tests for TrainingLlamaLoraRunner class-level constants."""

    @pytest.fixture(autouse=True)
    def runner_cls(self):
        self.runner_cls = _import_llama_runner()

    def test_device_mesh_shapes_populated(self):
        assert len(self.runner_cls.DEVICE_MESH_SHAPES) > 0

    def test_device_mesh_shapes_match_constants(self):
        for dt in TRAINING_RUNNER_SUPPORTED_DEVICES[ModelRunners.TRAINING_LLAMA_LORA]:
            assert dt.value in self.runner_cls.DEVICE_MESH_SHAPES
            assert (
                self.runner_cls.DEVICE_MESH_SHAPES[dt.value]
                == TrainingMeshShapes[dt.name].value
            )

    def test_mesh_axis_names(self):
        assert self.runner_cls.MESH_AXIS_NAMES == ("batch", "model")

    def test_sharding_patterns_cover_key_layers(self):
        """Verify sharding patterns match the expected Llama layer names."""
        layer_names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.self_attn.q_proj.base_layer",
            "model.layers.0.self_attn.v_proj.lora_B.default",
        ]
        patterns = self.runner_cls.MODEL_SHARDING_PATTERNS

        for layer_name in layer_names:
            matched = any(re.search(p[0], layer_name) for p in patterns)
            assert matched, (
                f"Layer '{layer_name}' is not matched by any sharding pattern"
            )

    def test_sharding_patterns_do_not_match_unrelated_layers(self):
        """Verify sharding patterns don't match unrelated layer names."""
        unrelated = [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ]
        patterns = self.runner_cls.MODEL_SHARDING_PATTERNS

        for layer_name in unrelated:
            matched = any(re.search(p[0], layer_name) for p in patterns)
            assert not matched, (
                f"Layer '{layer_name}' should NOT be matched by any sharding pattern"
            )


class TestLlamaRunnerDeviceValidation:
    """Tests for device type validation inside the runner's run() method."""

    @pytest.fixture
    def runner(self):
        cls = _import_llama_runner()
        runner = object.__new__(cls)
        runner.device_id = "0,1"
        runner.logger = MagicMock()
        return runner

    def test_unsupported_device_raises(self, runner):
        request = MagicMock()
        request.device_type = "p150"
        request._training_logs = None

        with pytest.raises(ValueError, match="Llama Lora training requires"):
            runner.run([request])

    def test_supported_device_does_not_raise_on_validation(self, runner):
        """Verify validation passes for each supported device type."""
        supported = TRAINING_RUNNER_SUPPORTED_DEVICES[ModelRunners.TRAINING_LLAMA_LORA]
        for dt in supported:
            request = MagicMock()
            request.device_type = dt.value
            request._training_logs = None
            request._start_event = None

            with patch.object(runner, "_create_mesh") as mock_mesh:
                mock_mesh.side_effect = StopIteration("stop after validation")
                try:
                    runner.run([request])
                except StopIteration:
                    pass
                mock_mesh.assert_called_once_with(dt.value)


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
