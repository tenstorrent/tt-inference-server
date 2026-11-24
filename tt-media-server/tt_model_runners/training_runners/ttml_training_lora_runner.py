# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""LoRA fine-tuning runner backed by ttml's ``SFTTrainer`` (tt-metal's tt-train)."""

import contextlib
import math
from typing import Optional

from config.constants import (
    ModelNames,
    SupportedModels,
    TrainingOptimizers,
)
from domain.training_request import TrainingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.training_runners.callbacks import (
    AdapterCheckpointCallback,
    JobControlCallback,
    JobMetricsCallback,
    StopTraining,
)
from ttml.experimental.trainer import (
    AdamWConfig,
    CheckpointConfig,
    CustomDatasetConfig,
    DatasetConfig,
    LoggingConfig,
    LoraConfig,
    MetricsConfig,
    ModelConfig,
    NoSave,
    SFTTrainer,
    SFTTrainerConfig,
    StepSave,
    StoppingCriteria,
    TrainMetricsConfig,
    ValidationMetricsConfig,
)
from utils.decorators import log_execution_time

# ttml's ids, which it matches case-insensitively.
DATASET_IDS = ("sst2", "alpaca", "custom")

OPTIMIZER_CONFIGS = {
    TrainingOptimizers.ADAMW.value: AdamWConfig,
}


class TtmlTrainingLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads=num_torch_threads)
        if not self.settings.training_model:
            raise ValueError("TRAINING_MODEL and MODEL_RUNNER both must be set")
        self.model_name = SupportedModels[
            ModelNames(self.settings.training_model).name
        ].value

        self._trainer = None
        self._metrics_callback = None
        self._checkpoint_callback = None
        self._control_callback = None

    @property
    def _is_multichip(self) -> bool:
        return math.prod(self.settings.device_mesh_shape) > 1

    @log_execution_time("Setting up ttml LoRA training")
    async def warmup(self) -> bool:
        if self._is_multichip:
            raise NotImplementedError(
                f"{type(self).__name__} supports single-chip training only, but "
                f"device_mesh_shape is {self.settings.device_mesh_shape}."
            )

        self._metrics_callback = JobMetricsCallback(self.logger)
        self._checkpoint_callback = AdapterCheckpointCallback(
            self.logger, self._metrics_callback
        )
        self._control_callback = JobControlCallback(self.logger)
        # Order matters: the control callback stops the loop by raising, so the
        # metrics and checkpoint callbacks have to see the final step first.
        self._trainer = SFTTrainer(
            callbacks=[
                self._metrics_callback,
                self._checkpoint_callback,
                self._control_callback,
            ]
        )
        return True

    @log_execution_time("ttml LoRA training")
    def run(self, training_requests: list) -> list:
        if len(training_requests) > 1:
            self.logger.warning(
                f"Batch processing not supported. Processing only first of "
                f"{len(training_requests)} requests"
            )

        request: TrainingRequest = training_requests[0]

        log_handler = None
        if request._training_logs is not None:
            log_handler = self.logger.add_list_handler(request._training_logs)

        self.logger.info(
            f"Job started at Device {self.device_id}",
            extra={"log_type": "info", "step": 0},
        )

        callbacks = (
            self._metrics_callback,
            self._checkpoint_callback,
            self._control_callback,
        )
        for callback in callbacks:
            callback.bind(request)

        trainer = self._trainer
        try:
            self._validate(request)
            self._setup_trainer(trainer, request)
            # Cancellation ends the loop early; the job still succeeded.
            with contextlib.suppress(StopTraining):
                trainer.train()
        finally:
            # Closes the device the job opened, so the next job opens it afresh.
            try:
                trainer.cleanup()
            except Exception as cleanup_error:
                self.logger.error(
                    f"Device {self.device_id}: trainer cleanup failed: {cleanup_error}"
                )
            self.logger.info(
                f"Device {self.device_id}: Training completed - memory cleaned up"
            )
            if log_handler:
                self.logger.remove_handler(log_handler)

        return [request._output_model_path]

    def _setup_trainer(self, trainer, request: TrainingRequest) -> None:
        self.logger.info(
            "Downloading and preparing model weights for training.",
            extra={"log_type": "info", "step": 0},
        )
        # `setup` reports only its own failures to the callbacks, and the config
        # is validated as it is built, before `setup` is entered.
        try:
            config = self._job_config(request)
        except Exception as exception:
            trainer.callback_handler("on_setup_error", exception)
            raise
        trainer.setup(config)

    def _validate(self, request: TrainingRequest) -> None:
        error = self._validation_error(request)
        if error is not None:
            self.logger.error(error, extra={"log_type": "error", "step": 0})
            raise ValueError(error)

    def _validation_error(self, request: TrainingRequest) -> Optional[str]:
        # The dtype and LoRA targets are the model's to check: `setup` rejects
        # them before it opens the device or downloads weights.
        if request.device_type != self.settings.device:
            return (
                f"Request device '{request.device_type}' does not match "
                f"server device '{self.settings.device}'"
            )
        if request.model.lora is None:
            return "Full fine-tuning is not supported yet; model.lora must be set"
        if request.dataset.id.lower() not in DATASET_IDS:
            return (
                f"Unsupported dataset.id '{request.dataset.id}'; "
                f"expected one of {list(DATASET_IDS)}"
            )
        if request.optimizer.type not in OPTIMIZER_CONFIGS:
            return (
                f"Unsupported optimizer.type '{request.optimizer.type}'; "
                f"expected one of {sorted(OPTIMIZER_CONFIGS)}"
            )
        return None

    def _dataset_config(self, request: TrainingRequest) -> DatasetConfig:
        dataset = request.dataset
        custom_options = None
        if dataset.custom_options is not None:
            custom_options = CustomDatasetConfig(**dataset.custom_options.model_dump())
        return DatasetConfig(id=dataset.id, custom_options=custom_options)

    def _job_config(self, request: TrainingRequest) -> SFTTrainerConfig:
        optimizer = request.optimizer
        lora = request.model.lora
        return SFTTrainerConfig(
            model=ModelConfig(
                name=self.model_name,
                lora=LoraConfig(**lora.model_dump()) if lora is not None else None,
            ),
            dataset=self._dataset_config(request),
            # One optimizer over every trainable weight (the adapters).
            optimizer=[
                OPTIMIZER_CONFIGS[optimizer.type](
                    lr=optimizer.lr, weight_decay=optimizer.weight_decay
                )
            ],
            dtype=request.dtype,
            max_length=request.max_length,
            batch_size=request.batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            seed=request.seed,
            stopping_criteria=StoppingCriteria(
                **request.stopping_criteria.model_dump()
            ),
            metrics=MetricsConfig(
                train=TrainMetricsConfig(
                    frequency=request.metrics.train.frequency,
                    metric_names=["loss"],
                ),
                validation=ValidationMetricsConfig(
                    frequency=request.metrics.validation.frequency,
                    metric_names=["loss"],
                ),
            ),
            logging=LoggingConfig(
                log_level="INFO",
                use_wandb=False,
                wandb_run_name=f"tt-media-server-{self.device_id}",
            ),
            # Only AdapterCheckpointCallback saves, and the save strategy is all it reads.
            checkpoint=CheckpointConfig(save_strategy=self._save_strategy(request)),
        )

    def _save_strategy(self, request: TrainingRequest):
        save_strategy = request.checkpoint.save_strategy
        if save_strategy.type == "none":
            return NoSave()
        return StepSave(
            frequency=save_strategy.frequency,
            project_dir=request._output_model_path or "",
        )
