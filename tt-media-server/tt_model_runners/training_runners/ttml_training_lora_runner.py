# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""LoRA fine-tuning runner backed by ttml's ``SFTTrainer`` (tt-metal's tt-train)."""

import contextlib
import math
from typing import Optional

from config.constants import (
    DatasetLoaders,
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
from ttml.unified_trainer import (
    AdamWConfig,
    CheckpointConfig,
    CustomDatasetConfig,
    LoggingConfig,
    LoraConfig,
    MetricsConfig,
    SFTTrainer,
    SFTTrainerConfig,
)
from ttml.unified_trainer.data import IGNORED_INDEX
from utils.decorators import log_execution_time

DATASET_IDS = {
    DatasetLoaders.SST2.value: "sst2",
    DatasetLoaders.ALPACA.value: "alpaca",
    DatasetLoaders.CUSTOM.value: "custom",
}

# Each is built from the request's learning_rate and weight_decay.
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
            # max_steps or cancellation ends the loop early; the job still succeeded.
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
        trainer.setup(self._job_config(request))

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
        if request.dataset_loader not in DATASET_IDS:
            return (
                f"Unsupported dataset_loader '{request.dataset_loader}'; "
                f"expected one of {sorted(DATASET_IDS)}"
            )
        if request.optimizer not in OPTIMIZER_CONFIGS:
            return (
                f"Unsupported optimizer '{request.optimizer}'; "
                f"expected one of {sorted(OPTIMIZER_CONFIGS)}"
            )
        # SFTTrainer masks labels with a fixed index.
        if request.ignored_index != IGNORED_INDEX:
            return (
                f"Unsupported ignored_index {request.ignored_index}; "
                f"expected {IGNORED_INDEX}"
            )
        return None

    def _custom_dataset_config(
        self, request: TrainingRequest
    ) -> Optional[CustomDatasetConfig]:
        if request.dataset_loader != DatasetLoaders.CUSTOM.value:
            return None
        return CustomDatasetConfig(
            file_type=request.file_type,
            train_dataset_path=request.train_dataset_path,
            val_dataset_path=request.val_dataset_path,
            template=request.template,
            column_mapping=request.column_mapping,
        )

    def _job_config(self, request: TrainingRequest) -> SFTTrainerConfig:
        # No max_steps: JobControlCallback enforces the request's, so the run
        # still ends through StopTraining like a cancellation.
        return SFTTrainerConfig(
            dataset_id=DATASET_IDS[request.dataset_loader],
            custom_dataset=self._custom_dataset_config(request),
            model_name=self.model_name,
            dtype=request.dtype,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            val_steps_freq=request.val_steps_freq,
            logging=LoggingConfig(
                log_level="INFO",
                use_wandb=False,
                wandb_run_name=f"tt-media-server-{self.device_id}",
            ),
            metrics=MetricsConfig(
                steps_freq=request.steps_freq,
                train_metrics=["loss"],
                val_metrics=["loss"],
            ),
            # Only AdapterCheckpointCallback saves, and these are all it reads.
            checkpoint=CheckpointConfig(
                steps_freq=max(request.save_interval, 1),
                save_strategy="step" if request.save_interval > 0 else "none",
                project_dir=request._output_model_path or "",
            ),
            # One optimizer over every trainable weight (the adapters).
            optimizer=[
                OPTIMIZER_CONFIGS[request.optimizer](
                    lr=request.learning_rate,
                    weight_decay=request.weight_decay,
                )
            ],
            lora=LoraConfig(
                r=request.lora_r,
                alpha=request.lora_alpha,
                target_modules=request.lora_target_modules,
                task_type=request.lora_task_type,
            ),
            seed=request.seed,
            max_length=request.dataset_max_sequence_length,
        )
