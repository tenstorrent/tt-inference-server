# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from unittest.mock import MagicMock, patch

import pytest
from config.constants import DatasetLoaders, ModelNames, SupportedModels
from domain.training_request import TrainingRequest
from pydantic import ValidationError

RUNNER_MODULE = "tt_model_runners.forge_training_runners.trainer_training_lora_runner"

MODEL_ID = ModelNames.GEMMA_1_1_2B_IT.value
HF_REPO_ID = SupportedModels.GEMMA_1_1_2B_IT.value


def _settings(mesh_shape=(1, 1)):
    settings = MagicMock()
    settings.device = "p150"
    settings.device_mesh_shape = mesh_shape
    settings.training_model = MODEL_ID
    settings.use_dynamic_batcher = False
    return settings


def _build_runner(settings):
    with patch(
        "tt_model_runners.base_device_runner.get_settings", return_value=settings
    ), patch("tt_model_runners.base_device_runner.setup_runner_environment"):
        from tt_model_runners.forge_training_runners.trainer_training_lora_runner import (
            TrainerTrainingLoraRunner,
        )

        runner = TrainerTrainingLoraRunner("0")

    # Swapped in before warmup, since the callbacks capture it there. The
    # runner logs with `extra=`, which conftest's shared MockLogger rejects.
    runner.logger = MagicMock()
    return runner


def _request(**overrides):
    fields = {"device_type": "p150"}
    fields.update(overrides)
    request = TrainingRequest(**fields)
    request._output_model_path = "/tmp/job"
    request._training_metrics = []
    request._training_logs = []
    request._training_checkpoints = []
    return request


@pytest.fixture
def runner():
    return _build_runner(_settings())


class _Loss:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _fake_trainer(
    grad_accum=1, steps_freq=1, num_epochs=1, val_batches=None, train_batches=None
):
    trainer = MagicMock()
    trainer.config.gradient_accumulation_steps = grad_accum
    trainer.config.metrics.steps_freq = steps_freq
    trainer.config.num_epochs = num_epochs
    trainer.global_step = 0
    trainer.epoch = 0
    trainer.optimizer.param_groups = [{"lr": 0.001}]
    if train_batches is None:
        trainer.train_dataloader.__len__.side_effect = TypeError()
    else:
        trainer.train_dataloader.__len__.return_value = train_batches
    if val_batches is None:
        trainer.val_dataloader.__len__.side_effect = TypeError()
    else:
        trainer.val_dataloader.__len__.return_value = val_batches
    return trainer


class TestWarmup:
    @pytest.mark.asyncio
    async def test_builds_trainer_and_callbacks(self, runner):
        assert await runner.warmup() is True
        assert runner._trainer is not None
        # The control callback must run last so the metrics and checkpoint
        # callbacks still see the step it stops on.
        assert runner._trainer.callback_handler.callbacks == [
            runner._metrics_callback,
            runner._checkpoint_callback,
            runner._control_callback,
        ]

    @pytest.mark.asyncio
    async def test_rejects_multichip(self):
        runner = _build_runner(_settings(mesh_shape=(1, 2)))
        with pytest.raises(NotImplementedError, match="single-chip"):
            await runner.warmup()


class TestBaseModelDtype:
    @pytest.mark.asyncio
    async def test_loads_base_model_once_while_dtype_is_unchanged(self, runner):
        with patch(f"{RUNNER_MODULE}.AutoModelForCausalLM") as auto_model:
            await runner.warmup()
            runner._trainer.setup(runner._job_config(_request()))
            runner._trainer.setup(runner._job_config(_request()))

        assert auto_model.from_pretrained.call_count == 1

    @pytest.mark.asyncio
    async def test_reloads_base_model_when_dtype_changes(self, runner):
        with patch(f"{RUNNER_MODULE}.AutoModelForCausalLM") as auto_model:
            await runner.warmup()
            runner._trainer.setup(runner._job_config(_request(dtype="torch.float32")))

        assert auto_model.from_pretrained.call_count == 2
        assert runner._trainer._base_model_dtype == "torch.float32"

    @pytest.mark.asyncio
    async def test_reuses_the_reloaded_base_model(self, runner):
        with patch(f"{RUNNER_MODULE}.AutoModelForCausalLM") as auto_model:
            await runner.warmup()
            for _ in range(2):
                runner._trainer.setup(
                    runner._job_config(_request(dtype="torch.float32"))
                )

        # Warmup plus one reload; the second float32 job reuses what is resident.
        assert auto_model.from_pretrained.call_count == 2

    @pytest.mark.asyncio
    async def test_dtype_survives_a_job_teardown(self, runner):
        await runner.warmup()
        runner.run([_request(dtype="torch.float32")])

        assert runner._trainer.config is None
        assert runner._trainer._base_model_dtype == "torch.float32"


class TestJobConfig:
    def test_maps_request_fields(self, runner):
        request = _request(
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=3,
            gradient_accumulation_steps=4,
            dataset_max_sequence_length=256,
            lora_r=16,
            lora_alpha=32,
            steps_freq=5,
            save_interval=25,
        )
        config = runner._job_config(request)

        assert config.dataset_id == "sst2"
        assert config.model_name == HF_REPO_ID
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 3
        assert config.gradient_accumulation_steps == 4
        assert config.max_length == 256
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.training_model_type == "lora"
        assert config.metrics.steps_freq == 5
        assert config.checkpoint.project_dir == "/tmp/job"
        # Checkpoints are written by AdapterCheckpointCallback, not by blacksmith.
        assert config.checkpoint.save_strategy == "none"

    def test_weight_decay_defaults_to_adamws_own(self, runner):
        assert runner._job_config(_request()).weight_decay == 0.01

    def test_weight_decay_is_client_settable(self, runner):
        assert runner._job_config(_request(weight_decay=0.0)).weight_decay == 0.0

    def test_built_in_datasets_carry_no_custom_config(self, runner):
        assert runner._job_config(_request()).custom_dataset is None

    def test_maps_a_custom_dataset(self, runner):
        config = runner._job_config(
            _request(
                dataset_loader=DatasetLoaders.CUSTOM.value,
                train_dataset_path="/datasets/train.jsonl",
                val_dataset_path="/datasets/val.jsonl",
                file_type="jsonl",
                template="alpaca",
                column_mapping={"instruction": "prompt", "output": "response"},
            )
        )

        assert config.dataset_id == "custom"
        assert config.custom_dataset.file_type == "jsonl"
        assert config.custom_dataset.train_dataset_path == "/datasets/train.jsonl"
        assert config.custom_dataset.val_dataset_path == "/datasets/val.jsonl"
        assert config.custom_dataset.template == "alpaca"
        assert config.custom_dataset.column_mapping == {
            "instruction": "prompt",
            "output": "response",
        }

    def test_a_custom_dataset_may_omit_validation_and_columns(self, runner):
        config = runner._job_config(
            _request(
                dataset_loader=DatasetLoaders.CUSTOM.value,
                train_dataset_path="/datasets/train.json",
                file_type="json",
                template="alpaca",
            )
        )

        # blacksmith skips validation without a path and infers the columns by
        # name without a mapping.
        assert config.custom_dataset.val_dataset_path is None
        assert config.custom_dataset.column_mapping is None

    def test_never_configures_a_mesh(self, runner):
        config = runner._job_config(_request())
        assert config.mesh_shape is None
        assert config.model_sharding_patterns is None

    def test_wandb_is_disabled(self, runner):
        assert runner._job_config(_request()).logging.use_wandb is False


class TestValidate:
    def test_accepts_a_matching_request(self, runner):
        runner._validate(_request())

    def test_rejects_other_device(self, runner):
        with pytest.raises(ValueError, match="does not match"):
            runner._validate(_request(device_type="p300"))

    def test_accepts_a_supported_dtype(self, runner):
        runner._validate(_request(dtype="torch.float32"))

    def test_rejects_unsupported_dtype(self, runner):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            runner._validate(_request(dtype="torch.float8"))

    def test_rejects_unknown_dataset(self, runner):
        with pytest.raises(ValueError, match="Unsupported dataset_loader"):
            runner._validate(_request(dataset_loader="nonexistent"))


class TestTrainingRequest:
    def test_a_custom_dataset_requires_path_file_type_and_template(self):
        with pytest.raises(ValueError, match="train_dataset_path"):
            _request(
                dataset_loader=DatasetLoaders.CUSTOM.value,
                file_type="json",
                template="alpaca",
            )
        with pytest.raises(ValueError, match="file_type"):
            _request(
                dataset_loader=DatasetLoaders.CUSTOM.value,
                train_dataset_path="/datasets/train.json",
                template="alpaca",
            )
        with pytest.raises(ValueError, match="template"):
            _request(
                dataset_loader=DatasetLoaders.CUSTOM.value,
                train_dataset_path="/datasets/train.json",
                file_type="json",
            )

    def test_built_in_datasets_need_no_paths(self):
        assert _request().train_dataset_path is None
        assert _request().file_type is None
        assert _request().template is None

    def test_val_steps_freq_zero_is_allowed(self):
        assert _request(val_steps_freq=0).val_steps_freq == 0

    def test_val_steps_freq_must_not_be_negative(self):
        with pytest.raises(ValidationError, match="val_steps_freq"):
            _request(val_steps_freq=-1)


class TestRun:
    @pytest.mark.asyncio
    async def test_returns_output_path_and_binds_callbacks(self, runner):
        await runner.warmup()
        request = _request()

        assert runner.run([request]) == ["/tmp/job"]
        assert runner._control_callback._request is request
        # The trainer is left ready for the next job.
        assert runner._trainer.config is None

    # These patch the class rather than the instance: run() starts with
    # trainer.setup(), whose cleanup() clears the instance __dict__.
    @pytest.mark.asyncio
    async def test_reraises_failure_reported_through_callbacks(self, runner):
        await runner.warmup()
        boom = RuntimeError("device fell over")

        def fake_train(trainer):
            # What Trainer._train_lifecycle does: notify, then return normally.
            trainer.callback_handler("on_error", boom)

        with patch.object(type(runner._trainer), "train", fake_train):
            with pytest.raises(RuntimeError, match="device fell over"):
                runner.run([_request()])

    @pytest.mark.asyncio
    async def test_reports_pre_training_failure_to_callbacks(self, runner):
        await runner.warmup()
        boom = RuntimeError("dataset missing")

        with patch.object(type(runner._trainer), "_load_dataloaders", side_effect=boom):
            with pytest.raises(RuntimeError, match="dataset missing"):
                runner.run([_request()])

        assert runner._control_callback.error is boom

    @pytest.mark.asyncio
    async def test_cleans_up_after_failure(self, runner):
        await runner.warmup()

        with patch.object(
            type(runner._trainer), "train", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                runner.run([_request()])

        assert runner._trainer.config is None


class TestJobLoraTrainer:
    @pytest.mark.asyncio
    async def test_cleanup_unloads_adapter_and_keeps_base_model(self, runner):
        await runner.warmup()
        trainer = runner._trainer
        trainer.setup(runner._job_config(_request()))
        # Patched locally: conftest's `peft` mock is shared for the whole
        # session, so its call counts leak between tests.
        with patch(f"{RUNNER_MODULE}.get_peft_model") as get_peft_model:
            trainer.model = trainer._load_model()
        peft_model = get_peft_model.return_value
        assert trainer.peft_model is peft_model

        trainer.cleanup()

        peft_model.unload.assert_called_once()
        assert trainer._base_model is peft_model.unload.return_value
        assert not hasattr(trainer, "peft_model")
        assert not hasattr(trainer, "model")
        assert trainer.config is None
        assert trainer.global_step == 0
        # Callbacks are owned by the runner and outlive every job.
        assert trainer.callback_handler.callbacks == [
            runner._metrics_callback,
            runner._checkpoint_callback,
            runner._control_callback,
        ]


class TestJobControlCallback:
    def _callback(self, request):
        from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
            JobControlCallback,
        )

        callback = JobControlCallback(MagicMock())
        callback.bind(request)
        return callback

    def test_stops_at_max_steps(self):
        from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
            StopTraining,
        )

        request = _request(max_steps=2)
        request._cancel_event = None
        callback = self._callback(request)
        trainer = _fake_trainer()

        trainer.global_step = 1
        callback.on_train_batch_end(trainer)

        trainer.global_step = 2
        with pytest.raises(StopTraining, match="max_steps=2"):
            callback.on_train_batch_end(trainer)

    def test_does_not_stop_when_max_steps_is_zero(self):
        request = _request(max_steps=0)
        request._cancel_event = None
        callback = self._callback(request)
        trainer = _fake_trainer()
        trainer.global_step = 1000

        callback.on_train_batch_end(trainer)

    def test_stops_on_cancel(self):
        from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
            StopTraining,
        )

        request = _request(max_steps=0)
        request._cancel_event = MagicMock()
        request._cancel_event.is_set.return_value = True
        callback = self._callback(request)

        with pytest.raises(StopTraining, match="Job cancelled"):
            callback.on_train_batch_end(_fake_trainer())

    def test_stops_on_cancel_during_validation(self):
        from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
            StopTraining,
        )

        request = _request(max_steps=0)
        request._cancel_event = MagicMock()
        request._cancel_event.is_set.return_value = True
        callback = self._callback(request)
        trainer = _fake_trainer()

        with pytest.raises(StopTraining, match="Job cancelled"):
            callback.on_validation_batch_end(trainer, batch=None, loss=None)
        callback._logger.info.assert_called()
        assert "Job cancelled at step 0" in callback._logger.info.call_args.args[0]

    def test_stop_is_not_recorded_as_an_error(self):
        from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
            StopTraining,
        )

        request = _request()
        request._cancel_event = None
        callback = self._callback(request)

        callback.on_error(_fake_trainer(), StopTraining("max steps"))
        assert callback.error is None

    def test_real_failure_is_recorded(self):
        request = _request()
        request._cancel_event = None
        callback = self._callback(request)
        boom = RuntimeError("boom")

        callback.on_error(_fake_trainer(), boom)
        assert callback.error is boom
        error_messages = [
            call.args[0] for call in callback._logger.error.call_args_list
        ]
        assert any(
            msg.startswith("Job failed at step 0: boom") for msg in error_messages
        )
        assert any("Full traceback:" in msg for msg in error_messages)


class TestJobMetricsCallback:
    def _callback(self, request):
        from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
            JobMetricsCallback,
        )

        callback = JobMetricsCallback(MagicMock())
        callback.bind(request)
        return callback

    def test_averages_micro_batches_within_a_step(self):
        request = _request()
        callback = self._callback(request)
        trainer = _fake_trainer(grad_accum=2, steps_freq=1)

        callback.on_backward_end(trainer, _Loss(2.0))
        # First micro-batch: no optimizer step, so nothing is reported yet.
        callback.on_train_batch_end(trainer)
        assert request._training_metrics == []

        callback.on_backward_end(trainer, _Loss(4.0))
        trainer.global_step = 1
        callback.on_train_batch_end(trainer)

        assert callback.last_train_loss == 3.0
        assert [m["value"] for m in request._training_metrics] == [3.0]

    def test_averages_steps_within_the_logging_window(self):
        request = _request()
        callback = self._callback(request)
        trainer = _fake_trainer(grad_accum=1, steps_freq=2)

        for step, loss in enumerate((1.0, 3.0), start=1):
            callback.on_backward_end(trainer, _Loss(loss))
            trainer.global_step = step
            callback.on_train_batch_end(trainer)

        assert [m["value"] for m in request._training_metrics] == [2.0]

    def test_logs_epoch_and_step_every_step_without_loss(self):
        request = _request()
        callback = self._callback(request)
        trainer = _fake_trainer(grad_accum=1, steps_freq=2, num_epochs=3)

        for step, loss in enumerate((1.0, 3.0), start=1):
            callback.on_backward_end(trainer, _Loss(loss))
            trainer.global_step = step
            callback.on_train_batch_end(trainer)

        progress_messages = [
            call.args[0]
            for call in callback._logger.info.call_args_list
            if "train_loss" not in call.args[0]
        ]
        assert progress_messages == [
            "Epoch 1/3 | Step 1",
            "Epoch 1/3 | Step 2",
        ]
        loss_messages = [
            call.args[0]
            for call in callback._logger.info.call_args_list
            if "train_loss" in call.args[0]
        ]
        assert loss_messages == [
            "Epoch 1/3 | Step 2 | train_loss: 2.000000",
        ]

    def test_logs_train_batch_count_at_train_start(self):
        request = _request()
        callback = self._callback(request)
        trainer = _fake_trainer(num_epochs=2, train_batches=42)

        callback.on_train_start(trainer)

        messages = [call.args[0] for call in callback._logger.info.call_args_list]
        assert messages == [
            "Epoch 1/2 | Step 0 | Starting training (42 batches per epoch)",
        ]

    def test_logs_validation_batch_progress_without_loss(self):
        request = _request()
        callback = self._callback(request)
        trainer = _fake_trainer(num_epochs=2, val_batches=2)
        trainer.global_step = 10
        trainer.epoch = 0

        callback.on_validation_start(trainer)
        callback.on_validation_batch_end(trainer, batch=None, loss=_Loss(1.0))
        callback.on_validation_batch_end(trainer, batch=None, loss=_Loss(2.0))
        callback.on_validation_end(trainer, 1.5)

        messages = [call.args[0] for call in callback._logger.info.call_args_list]
        assert messages == [
            "Epoch 1/2 | Step 10 | Starting validation (2 batches)",
            "Epoch 1/2 | Step 10 | Validation batch 1/2",
            "Epoch 1/2 | Step 10 | Validation batch 2/2",
            "Epoch 1/2 | Step 10 | val_loss: 1.500000",
        ]
        assert [m["metric_name"] for m in request._training_metrics] == ["val_loss"]

    def test_records_validation_loss(self):
        request = _request()
        callback = self._callback(request)

        callback.on_validation_end(_fake_trainer(), 1.25)

        assert callback.last_val_loss == 1.25
        assert request._training_metrics[0]["metric_name"] == "val_loss"
        assert request._training_metrics[0]["value"] == 1.25
