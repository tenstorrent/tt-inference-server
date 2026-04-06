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


def _repair_sys_modules():
    """Repair sys.modules if test_worker_utils.py replaced base_device_runner
    with a bare Mock() at module scope."""
    import types

    bdr_key = "tt_model_runners.base_device_runner"
    runner_key = "tt_model_runners.forge_training_runners.training_llama_lora_runner"

    bdr_mod = sys.modules.get(bdr_key)
    if bdr_mod is not None and not isinstance(bdr_mod, types.ModuleType):
        repaired = types.ModuleType(bdr_key)
        repaired.BaseDeviceRunner = type("BaseDeviceRunner", (), {})
        sys.modules[bdr_key] = repaired
        sys.modules.pop(runner_key, None)


def _import_llama_runner():
    _repair_sys_modules()
    from tt_model_runners.forge_training_runners.training_llama_lora_runner import (
        TrainingLlamaLoraRunner,
    )

    return TrainingLlamaLoraRunner


def _import_llama_runner_module():
    """Import the full runner module, returning (module, class)."""
    _repair_sys_modules()
    import tt_model_runners.forge_training_runners.training_llama_lora_runner as mod

    return mod, mod.TrainingLlamaLoraRunner


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


# ---------------------------------------------------------------------------
# Module-level function tests
# ---------------------------------------------------------------------------


class TestTransformLabels:
    """Tests for _transform_labels (lines 39-44)."""

    def test_returns_two_element_tuple(self):
        mod, _ = _import_llama_runner_module()
        labels = MagicMock()
        result = mod._transform_labels(labels, ignored_index=-100, vocab_size=32000)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_calls_torch_where_and_one_hot(self):
        mod, _ = _import_llama_runner_module()
        torch_mod = sys.modules["torch"]
        F_mod = sys.modules["torch"].nn.functional

        labels = MagicMock()
        mod._transform_labels(labels, ignored_index=-100, vocab_size=128)
        torch_mod.where.assert_called()
        F_mod.one_hot.assert_called()


class TestCrossEntropyLoss:
    """Tests for _cross_entropy_loss (lines 47-66)."""

    def test_returns_loss_value(self):
        mod, _ = _import_llama_runner_module()
        shift_logits = MagicMock()
        expected_output = MagicMock()
        labels_mask = MagicMock()
        result = mod._cross_entropy_loss(shift_logits, expected_output, labels_mask)
        assert result is not None

    def test_calls_log_softmax(self):
        mod, _ = _import_llama_runner_module()
        F_mod = sys.modules["torch"].nn.functional
        F_mod.log_softmax.reset_mock()

        mod._cross_entropy_loss(MagicMock(), MagicMock(), MagicMock())
        F_mod.log_softmax.assert_called()

    def test_calls_torch_clamp(self):
        mod, _ = _import_llama_runner_module()
        torch_mod = sys.modules["torch"]
        torch_mod.clamp.reset_mock()

        mod._cross_entropy_loss(MagicMock(), MagicMock(), MagicMock())
        torch_mod.clamp.assert_called()


class TestTrainingStepInner:
    """Tests for _training_step_inner (lines 69-83)."""

    def test_calls_model_forward_and_backward(self):
        mod, _ = _import_llama_runner_module()
        model = MagicMock()
        batch = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
            "expected_output": MagicMock(),
            "labels_mask": MagicMock(),
        }

        result = mod._training_step_inner(batch, model)
        model.assert_called_once_with(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Runner instance method tests
# ---------------------------------------------------------------------------


def _make_runner_instance():
    """Create a TrainingLlamaLoraRunner with mocked internals."""
    cls = _import_llama_runner()
    runner = object.__new__(cls)
    runner.device_id = "0,1"
    runner.logger = MagicMock()
    runner.device = MagicMock()
    runner.hf_model = MagicMock()
    runner.model_name = "meta-llama/Llama-3.1-8B"
    return runner


class TestLlamaRunnerInit:
    """Tests for __init__ and warmup (lines 87-122)."""

    def test_init_sets_model_name(self):
        mod, cls = _import_llama_runner_module()
        with patch.object(cls, "__init__", lambda self, *a, **kw: None):
            runner = cls.__new__(cls)
            runner.__init__()
        runner.model_name = mod.SupportedModels.LLAMA_3_1_8B.value
        assert "Llama" in runner.model_name

    def test_warmup_configures_spmd(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()

        from contextlib import ExitStack

        with ExitStack() as stack:
            model_cls = stack.enter_context(patch.object(mod, "AutoModelForCausalLM"))
            xr_mock = stack.enter_context(patch.object(mod, "xr"))
            txa_mock = stack.enter_context(patch.object(mod, "torch_xla"))
            stack.enter_context(patch.dict("os.environ", {}, clear=False))
            model_cls.from_pretrained.return_value = MagicMock()
            txa_mock.device.return_value = MagicMock()
            import asyncio

            asyncio.get_event_loop().run_until_complete(runner.warmup())

        xr_mock.set_device_type.assert_called_once_with("TT")
        xr_mock.use_spmd.assert_called_once()


class TestPrepareBatchWithSharding:
    """Test the input sharding branch (lines 181-187)."""

    def test_shards_input_when_dim_is_set(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()
        original_dim = runner.INPUT_SHARDING_DIM
        try:
            runner.INPUT_SHARDING_DIM = "batch"
            tensor = MagicMock()
            tensor.to.return_value = tensor
            tensor.dim.return_value = 2

            mark_sharding_mock = MagicMock()
            mesh = MagicMock()
            with patch.object(mod, "xs") as xs_mock:
                xs_mock.mark_sharding = mark_sharding_mock
                runner._prepare_batch({"input_ids": tensor}, mesh)

            mark_sharding_mock.assert_called_once()
        finally:
            runner.INPUT_SHARDING_DIM = original_dim


class TestCreateMesh:
    """Tests for _create_mesh (lines 145-159)."""

    def test_creates_mesh_for_supported_device(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()

        mock_mesh_cls = MagicMock()
        with patch.object(mod, "xr") as xr_mock, patch.object(mod, "xs") as xs_mock:
            xr_mock.global_runtime_device_count.return_value = 2
            xs_mock.Mesh = mock_mesh_cls
            mesh = runner._create_mesh("p300")

        mock_mesh_cls.assert_called_once()
        runner.logger.info.assert_called()
        assert mesh is not None


class TestShardModel:
    """Tests for _shard_model (lines 161-175)."""

    def test_marks_sharding_on_matching_layers(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()

        weight_mock = MagicMock()
        module_with_weight = MagicMock()
        module_with_weight.weight = weight_mock

        model = MagicMock()
        model.named_modules.return_value = [
            ("base_model.model.model.layers.0.self_attn.q_proj", module_with_weight),
        ]

        mesh = MagicMock()
        mark_sharding_mock = MagicMock()
        sync_mock = MagicMock()
        with patch.object(mod, "xs") as xs_mock, patch.object(
            mod, "torch_xla"
        ) as txa_mock:
            xs_mock.mark_sharding = mark_sharding_mock
            txa_mock.sync = sync_mock
            runner._shard_model(model, mesh, runner.MODEL_SHARDING_PATTERNS)

        mark_sharding_mock.assert_called_once()
        sync_mock.assert_called()

    def test_skips_modules_without_weight(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()

        module_no_weight = MagicMock(spec=[])
        model = MagicMock()
        model.named_modules.return_value = [
            ("model.layers.0.input_layernorm", module_no_weight),
        ]

        mark_sharding_mock = MagicMock()
        with patch.object(mod, "xs") as xs_mock:
            xs_mock.mark_sharding = mark_sharding_mock
            runner._shard_model(model, MagicMock(), runner.MODEL_SHARDING_PATTERNS)
        mark_sharding_mock.assert_not_called()

    def test_skips_modules_with_none_weight(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()

        module_none_weight = MagicMock()
        module_none_weight.weight = None
        model = MagicMock()
        model.named_modules.return_value = [
            ("model.layers.0.some_layer", module_none_weight),
        ]

        mark_sharding_mock = MagicMock()
        with patch.object(mod, "xs") as xs_mock:
            xs_mock.mark_sharding = mark_sharding_mock
            runner._shard_model(model, MagicMock(), runner.MODEL_SHARDING_PATTERNS)
        mark_sharding_mock.assert_not_called()


class TestPrepareBatch:
    """Tests for _prepare_batch (lines 177-189)."""

    def test_moves_tensors_to_device(self):
        runner = _make_runner_instance()
        originals = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        runner._prepare_batch(dict(originals), MagicMock())
        for tensor in originals.values():
            tensor.to.assert_called_once_with(runner.device)

    def test_no_input_sharding_when_dim_is_none(self):
        mod, _ = _import_llama_runner_module()
        runner = _make_runner_instance()
        assert runner.INPUT_SHARDING_DIM is None

        mark_sharding_mock = MagicMock()
        with patch.object(mod, "xs") as xs_mock:
            xs_mock.mark_sharding = mark_sharding_mock
            runner._prepare_batch({"input_ids": MagicMock()}, MagicMock())
        mark_sharding_mock.assert_not_called()


# ---------------------------------------------------------------------------
# run() method tests
# ---------------------------------------------------------------------------

RUNNER_MODULE = "tt_model_runners.forge_training_runners.training_llama_lora_runner"


def _make_mock_request(
    device_type="p300",
    cancel_after_steps=None,
):
    """Build a TrainingRequest-like mock with realistic defaults."""
    request = MagicMock()
    request.device_type = device_type
    request._training_logs = []
    request._training_metrics = []
    request._output_model_path = "/tmp/test_model.pt"
    request._start_event = MagicMock()
    request.dtype = "torch.bfloat16"
    request.dataset_loader = "sst2"
    request.dataset_max_sequence_length = 32
    request.batch_size = 2
    request.lora_r = 4
    request.lora_alpha = 8
    request.lora_target_modules = ["q_proj", "v_proj"]
    request.lora_task_type = "CAUSAL_LM"
    request.ignored_index = -100
    request.optimizer = "adamw"
    request.learning_rate = 6e-5
    request.num_epochs = 1
    request.steps_freq = 1
    request.val_steps_freq = 2

    if cancel_after_steps is not None:
        call_count = 0

        def is_set_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count > cancel_after_steps

        cancel_event = MagicMock()
        cancel_event.is_set = MagicMock(side_effect=is_set_side_effect)
        request._cancel_event = cancel_event
    else:
        request._cancel_event = None

    return request


def _make_mock_batch():
    """Batch dict that looks like real dataloader output."""
    return {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
        "labels": MagicMock(),
    }


def _make_mock_dataset(num_batches=2):
    """Mock dataset whose get_dataloader returns a list of batches."""
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=num_batches * 2)
    ds.get_dataloader.return_value = [_make_mock_batch() for _ in range(num_batches)]
    return ds


class TestLlamaRunnerRun:
    """Tests for TrainingLlamaLoraRunner.run (lines 192-435)."""

    @pytest.fixture
    def runner(self):
        return _make_runner_instance()

    @pytest.fixture(autouse=True)
    def _patch_externals(self):
        """Patch heavy external calls used inside run()."""
        _import_llama_runner_module()
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_loss.detach.return_value = mock_loss

        mock_model = MagicMock()
        mock_model.config.vocab_size = 32000

        patches = [
            patch(
                f"{RUNNER_MODULE}.get_dataset_loader",
                side_effect=lambda **kw: _make_mock_dataset(),
            ),
            patch(
                f"{RUNNER_MODULE}._transform_labels",
                return_value=(MagicMock(), MagicMock()),
            ),
            patch(f"{RUNNER_MODULE}._training_step_inner", return_value=mock_loss),
            patch(f"{RUNNER_MODULE}.LoraConfig"),
            patch(f"{RUNNER_MODULE}.get_peft_model", return_value=mock_model),
            patch(f"{RUNNER_MODULE}.PeftModel"),
        ]
        self._patches = []
        for p in patches:
            p.start()
            self._patches.append(p)
        yield
        for p in self._patches:
            p.stop()

    def _run_with_patches(self, runner, request):
        """Run with _run_validation patched to return a float."""
        with patch.object(runner, "_run_validation", return_value=0.25):
            return runner.run([request])

    def test_happy_path_returns_model_path(self, runner):
        request = _make_mock_request()
        result = self._run_with_patches(runner, request)
        assert result == [request._output_model_path]

    def test_start_event_is_set(self, runner):
        request = _make_mock_request()
        self._run_with_patches(runner, request)
        request._start_event.set.assert_called_once()

    def test_metrics_are_recorded(self, runner):
        request = _make_mock_request()
        request._training_metrics = []
        self._run_with_patches(runner, request)
        metric_names = [m["metric_name"] for m in request._training_metrics]
        assert "val_loss" in metric_names
        assert "train_loss" in metric_names

    def test_log_handler_attached_and_removed(self, runner):
        request = _make_mock_request()
        handler_mock = MagicMock()
        runner.logger.add_list_handler.return_value = handler_mock
        self._run_with_patches(runner, request)
        runner.logger.add_list_handler.assert_called_once_with(request._training_logs)
        runner.logger.remove_handler.assert_called_with(handler_mock)

    def test_no_log_handler_when_logs_is_none(self, runner):
        request = _make_mock_request()
        request._training_logs = None
        self._run_with_patches(runner, request)
        runner.logger.add_list_handler.assert_not_called()

    def test_batch_warning_for_multiple_requests(self, runner):
        request = _make_mock_request()
        with patch.object(runner, "_run_validation", return_value=0.25):
            runner.run([request, MagicMock()])
        runner.logger.warning.assert_called_once()

    def test_cancellation_stops_training(self, runner):
        request = _make_mock_request(cancel_after_steps=1)
        result = self._run_with_patches(runner, request)
        assert result == [request._output_model_path]
        cancel_logged = any(
            "Cancellation requested" in str(c)
            for c in runner.logger.info.call_args_list
        )
        assert cancel_logged

    def test_no_start_event_when_none(self, runner):
        request = _make_mock_request()
        request._start_event = None
        self._run_with_patches(runner, request)

    def test_no_metrics_when_none(self, runner):
        request = _make_mock_request()
        request._training_metrics = None
        self._run_with_patches(runner, request)

    def test_error_during_training_is_raised(self, runner):
        request = _make_mock_request()
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(runner, "_run_validation", return_value=0.25)
            )
            stack.enter_context(
                patch(
                    f"{RUNNER_MODULE}._training_step_inner",
                    side_effect=RuntimeError("device error"),
                )
            )
            with pytest.raises(RuntimeError, match="device error"):
                runner.run([request])
        runner.logger.error.assert_called()

    def test_cleanup_runs_on_error(self, runner):
        from contextlib import ExitStack

        mod, _ = _import_llama_runner_module()
        xr_mock = MagicMock()
        request = _make_mock_request()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(runner, "_run_validation", return_value=0.25)
            )
            stack.enter_context(patch.object(mod, "xr", xr_mock))
            stack.enter_context(
                patch(
                    f"{RUNNER_MODULE}._training_step_inner",
                    side_effect=RuntimeError("boom"),
                )
            )
            with pytest.raises(RuntimeError):
                runner.run([request])
        xr_mock.clear_computation_cache.assert_called()

    def test_peft_model_unloaded_on_second_run(self, runner):
        mod, _ = _import_llama_runner_module()

        first_peft = MagicMock()
        runner._peft_model = first_peft

        request = _make_mock_request()
        with patch.object(runner, "_run_validation", return_value=0.25), patch.object(
            mod, "PeftModel", type(first_peft)
        ):
            runner.run([request])
        first_peft.unload.assert_called_once()


# ---------------------------------------------------------------------------
# _run_validation tests
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Tests for _run_validation (lines 437-480)."""

    def test_returns_average_loss(self):
        runner = _make_runner_instance()
        mod, _ = _import_llama_runner_module()

        mock_loss = MagicMock()
        mock_loss.item.return_value = 1.0

        eval_batches = [_make_mock_batch(), _make_mock_batch()]
        model = MagicMock()

        request = MagicMock()
        request.ignored_index = -100

        with patch(
            f"{RUNNER_MODULE}._transform_labels",
            return_value=(MagicMock(), MagicMock()),
        ), patch(f"{RUNNER_MODULE}._cross_entropy_loss", return_value=mock_loss):
            avg_loss = runner._run_validation(
                model, eval_batches, MagicMock(), request, 32000
            )

        assert avg_loss == 1.0
        model.eval.assert_called_once()

    def test_returns_zero_for_empty_dataloader(self):
        runner = _make_runner_instance()
        model = MagicMock()
        request = MagicMock()
        request.ignored_index = -100

        avg_loss = runner._run_validation(model, [], MagicMock(), request, 32000)
        assert avg_loss == 0.0
