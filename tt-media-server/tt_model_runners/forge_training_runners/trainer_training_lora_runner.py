# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""LoRA fine-tuning runner backed by tt-blacksmith's ``CrankLLMTrainer``."""

import contextlib
import math
from typing import Optional

import torch
from blacksmith.configs import (
    CheckpointConfig,
    CustomDatasetConfig,
    LLMTrainerConfig,
    LoggingConfig,
    LoraConfig,
    MetricsConfig,
)
from blacksmith.models.hf_models import TORCH_DTYPES
from blacksmith.tools.trainer.strategies.crank_llm_trainer import (
    IGNORED_INDEX,
    CrankLLMTrainer,
)
from config.constants import (
    DatasetLoaders,
    ModelNames,
    SupportedModels,
    TrainingOptimizers,
)
from domain.training_request import TrainingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
    AdapterCheckpointCallback,
    JobControlCallback,
    JobMetricsCallback,
    StopTraining,
)
from utils.decorators import log_execution_time

DATASET_IDS = {
    DatasetLoaders.SST2.value: "sst2",
    DatasetLoaders.ALPACA.value: "alpaca",
    DatasetLoaders.CUSTOM.value: "custom",
}


class TrainerTrainingLoraRunner(BaseDeviceRunner):
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

    @log_execution_time("Setting up trainer-backed LoRA training")
    async def warmup(self) -> bool:
        if self._is_multichip:
            raise NotImplementedError(
                f"{type(self).__name__} supports single-chip training only, but "
                f"device_mesh_shape is {self.settings.device_mesh_shape}. See the "
                "MULTICHIP TODO at the top of this module."
            )

        self._metrics_callback = JobMetricsCallback(self.logger)
        self._checkpoint_callback = AdapterCheckpointCallback(
            self.logger, self._metrics_callback
        )
        self._control_callback = JobControlCallback(self.logger)
        # Order matters: the control callback stops the loop by raising, so the
        # metrics and checkpoint callbacks have to see the final step first.
        self._trainer = CrankLLMTrainer(
            callbacks=[
                self._metrics_callback,
                self._checkpoint_callback,
                self._control_callback,
            ]
        )
        return True

    @log_execution_time("Trainer-backed LoRA training")
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
            try:
                trainer.cleanup()
            except Exception as cleanup_error:
                self.logger.error(
                    f"Device {self.device_id}: trainer cleanup failed: {cleanup_error}"
                )
            # Each job compiles the loss afresh; drop the previous job's graphs so
            # they neither hold memory nor count toward dynamo's recompile limit.
            try:
                torch._dynamo.reset()
            except Exception as reset_error:
                self.logger.error(
                    f"Device {self.device_id}: dynamo reset failed: {reset_error}"
                )
            self.logger.info(
                f"Device {self.device_id}: Training completed - memory cleaned up"
            )
            if log_handler:
                self.logger.remove_handler(log_handler)

        return [request._output_model_path]

    def _setup_trainer(self, trainer, request: TrainingRequest) -> None:
        # TODO(mmilosevicTT): Log from a trainer callback once setup() has one before the model loads.
        self.logger.info(
            "Downloading and preparing model weights for training.",
            extra={"log_type": "info", "step": 0},
        )
        try:
            trainer.setup(self._job_config(request))
        except Exception as exception:
            # `train()` reports its own failures to the callbacks; setup's never reach them.
            trainer.callback_handler("on_error", exception)
            raise

    def _validate(self, request: TrainingRequest) -> None:
        error = self._validation_error(request)
        if error is not None:
            self.logger.error(error, extra={"log_type": "error", "step": 0})
            raise ValueError(error)

    def _validation_error(self, request: TrainingRequest) -> Optional[str]:
        if request.device_type != self.settings.device:
            return (
                f"Request device '{request.device_type}' does not match "
                f"server device '{self.settings.device}'"
            )
        if request.dtype not in TORCH_DTYPES:
            return (
                f"Unsupported dtype '{request.dtype}'; "
                f"expected one of {sorted(TORCH_DTYPES)}"
            )
        if request.dataset_loader not in DATASET_IDS:
            return (
                f"Unsupported dataset_loader '{request.dataset_loader}'; "
                f"expected one of {sorted(DATASET_IDS)}"
            )
        # CrankLLMTrainer always builds AdamW and masks labels with a fixed index.
        if request.optimizer != TrainingOptimizers.ADAMW.value:
            return (
                f"Unsupported optimizer '{request.optimizer}'; "
                f"expected '{TrainingOptimizers.ADAMW.value}'"
            )
        if request.ignored_index != IGNORED_INDEX:
            return (
                f"Unsupported ignored_index {request.ignored_index}; "
                f"expected {IGNORED_INDEX}"
            )
        return None

    def _logging_config(self) -> LoggingConfig:
        return LoggingConfig(
            log_level="INFO",
            use_wandb=False,
            wandb_project="",
            wandb_run_name=f"tt-media-server-{self.device_id}",
            wandb_tags=[],
            wandb_watch_mode="gradients",
            wandb_log_freq=0,
            model_to_wandb=False,
        )

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

    def _job_config(self, request: TrainingRequest) -> LLMTrainerConfig:
        return LLMTrainerConfig(
            dataset_id=DATASET_IDS[request.dataset_loader],
            custom_dataset=self._custom_dataset_config(request),
            model_name=self.model_name,
            dtype=request.dtype,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            weight_decay=request.weight_decay,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            val_steps_freq=request.val_steps_freq,
            logging=self._logging_config(),
            metrics=MetricsConfig(
                steps_freq=request.steps_freq,
                train_metrics=["loss"],
                val_metrics=["loss"],
            ),
            # Only AdapterCheckpointCallback saves, so the rotation and
            # init-from-checkpoint settings below are never acted on.
            checkpoint=CheckpointConfig(
                steps_freq=max(request.save_interval, 1),
                epoch_freq=1,
                save_strategy="step" if request.save_interval > 0 else "none",
                project_dir=request._output_model_path or "",
                final_checkpoint_name="adapter",
                save_optim=False,
                keep_last_n=1,
                keep_best_n=0,
                keep_all_checkpoints=True,
                checkpoint_metric="val/loss",
                checkpoint_metric_mode="min",
                init_from_checkpoint=False,
                init_option="last",
                checkpoint_path="",
            ),
            lora=LoraConfig(
                r=request.lora_r,
                alpha=request.lora_alpha,
                target_modules=request.lora_target_modules,
                task_type=request.lora_task_type,
            ),
            seed=request.seed,
            deterministic=False,
            use_tt=True,
            mesh_shape=None,
            mesh_axis_names=None,
            input_sharding_dim=None,
            model_sharding_patterns=None,
            max_length=request.dataset_max_sequence_length,
        )
