# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""LoRA fine-tuning runner backed by an external trainer class.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch_xla
import torch_xla.runtime as xr
from blacksmith.tools.configs import (
    CheckpointConfig,
    CustomDatasetConfig,
    LoggingConfig,
    MetricsConfig,
)
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.trainer.configs.base import TORCH_DTYPES
from blacksmith.tools.trainer.configs.lora_llm import LoraLLMConfig
from blacksmith.tools.trainer.strategies.lora_llm_trainer import LoraLLMTrainer
from config.constants import DatasetLoaders, ModelNames, SupportedModels
from domain.training_request import TrainingRequest
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.forge_training_runners.blacksmith_callbacks import (
    AdapterCheckpointCallback,
    JobControlCallback,
    JobMetricsCallback,
)
from utils.decorators import log_execution_time

# dtype the base model is loaded with at warmup. Matches TrainingRequest's
# default so the common case does not pay for a reload on the first job; a
# request asking for another dtype reloads the base model (see _load_base_model).
WARMUP_DTYPE = "torch.bfloat16"

# Device-level compile options, matching training_lora_runner.
DEVICE_COMPILE_OPTIONS = {"fp32_dest_acc_en": True, "math_fidelity": "hifi4"}

MODEL_COMPILE_OPTIONS = {
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_legacy_compile": True,
    "tt_use_aot_autograd": False,
}

DATASET_IDS = {
    DatasetLoaders.SST2.value: "sst2",
    DatasetLoaders.ALPACA.value: "alpaca",
    DatasetLoaders.CUSTOM.value: "custom",
}


@dataclass
class DeploymentConfig:
    model_name: str
    logging: LoggingConfig
    dtype: str = WARMUP_DTYPE
    use_tt: bool = True
    mesh_shape: Optional[List[int]] = None
    mesh_axis_names: Optional[List[str]] = None
    input_sharding_dim: Optional[str] = None
    model_sharding_patterns: Optional[list] = None

    def torch_dtype(self) -> "torch.dtype":
        try:
            return TORCH_DTYPES[self.dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}; expected one of {sorted(TORCH_DTYPES)}"
            ) from error


class JobLoraTrainer(LoraLLMTrainer):
    def setup(self, config=None, **kwargs) -> None:
        if self.config is not None:
            self.cleanup()

        self.config = config
        self.logger = TrainingLogger(config.logging)
        self.device_manager = DeviceManager(config)
        torch_xla.set_custom_compile_options(DEVICE_COMPILE_OPTIONS)

        self._load_base_model(config)

    def _load_base_model(self, config) -> None:
        """Load the shared base model, reusing the resident one where possible.
        """
        if (
            getattr(self, "_base_model", None) is not None
            and getattr(self, "_base_model_dtype", None) == config.dtype
        ):
            return

        resident_dtype = getattr(self, "_base_model_dtype", None)
        if resident_dtype is not None:
            self.logger.info(
                f"Reloading base model: dtype changed from {resident_dtype} to "
                f"{config.dtype}"
            )

        # Release the resident copy first so the two never occupy memory at once.
        self._base_model = None
        self._base_model_dtype = None
        self._base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, use_cache=False
        ).to(config.torch_dtype())
        self._base_model_dtype = config.dtype

    def _load_model(self) -> "torch.nn.Module":
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            task_type=self.config.lora_task_type,
        )
        self.peft_model = get_peft_model(self._base_model, lora_config)
        self.peft_model.to(self.device_manager.device)
        torch_xla.sync(wait=True)
        return torch.compile(
            self.peft_model, backend="tt", options=MODEL_COMPILE_OPTIONS
        )

    def cleanup(self) -> None:
        """``Trainer.cleanup()`` plus the adapter unload, keeping the base model."""
        peft_model = getattr(self, "peft_model", None)
        if peft_model is not None:
            self._base_model = peft_model.unload()
            if hasattr(self._base_model, "peft_config"):
                delattr(self._base_model, "peft_config")

        preserved = {"callback_handler", "_base_model", "_base_model_dtype"}
        for attr in list(self.__dict__):
            if attr not in preserved:
                delattr(self, attr)
        self.config = None
        self.reproducibility_manager = None
        self.logger = None
        self.global_step = 0
        self.epoch = 0


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

        self.logger.info(
            f"Device {self.device_id}: Setting up tt-blacksmith LoRA trainer for "
            f"{self.model_name}..."
        )

        self._metrics_callback = JobMetricsCallback(self.logger)
        self._checkpoint_callback = AdapterCheckpointCallback(
            self.logger, self._metrics_callback
        )
        self._control_callback = JobControlCallback(self.logger)
        # Order matters: the control callback stops the loop by raising, so the
        # metrics and checkpoint callbacks have to see the final step first.
        self._trainer = JobLoraTrainer(
            callbacks=[
                self._metrics_callback,
                self._checkpoint_callback,
                self._control_callback,
            ]
        )
        # Opens the device and loads the weights before this worker reports
        # ready, so a busy device fails startup instead of the first user's job.
        self._trainer.setup(self._deployment_config())

        self.logger.info(
            f"Device {self.device_id}: tt-blacksmith LoRA trainer ready "
            f"(dtype={WARMUP_DTYPE})"
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

        if request._start_event:
            request._start_event.set()
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
            # Swaps in the job's config. The logger and device manager are
            # rebuilt from it; only the base model carries over from warmup.
            trainer.setup(self._job_config(request))
            trainer.model = trainer._load_model()
            trainer.train_dataloader, trainer.val_dataloader = (
                trainer._load_dataloaders()
            )
            trainer.optimizer = trainer._load_optimizer()

            trainer.train()

            # `_train_lifecycle` notifies callbacks and returns normally, so a
            # failed job only surfaces through the control callback.
            if self._control_callback.error is not None:
                raise self._control_callback.error
        except Exception as exception:
            # Anything raised outside `train()` never reached the callbacks.
            if self._control_callback.error is not exception:
                trainer.callback_handler("on_error", exception)
            raise
        finally:
            try:
                trainer.cleanup()
            except Exception as cleanup_error:
                self.logger.error(
                    f"Device {self.device_id}: trainer cleanup failed: {cleanup_error}"
                )
            xr.clear_computation_cache()
            self.logger.info(
                f"Device {self.device_id}: Training completed - memory cleaned up"
            )
            if log_handler:
                self.logger.remove_handler(log_handler)

        return [request._output_model_path]

    def _validate(self, request: TrainingRequest) -> None:
        if request.device_type != self.settings.device:
            raise ValueError(
                f"Request device '{request.device_type}' does not match "
                f"server device '{self.settings.device}'"
            )
        if request.dtype not in TORCH_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{request.dtype}'; "
                f"expected one of {sorted(TORCH_DTYPES)}"
            )
        if request.dataset_loader not in DATASET_IDS:
            raise ValueError(
                f"Unsupported dataset_loader '{request.dataset_loader}'; "
                f"expected one of {sorted(DATASET_IDS)}"
            )

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

    def _deployment_config(self) -> DeploymentConfig:
        return DeploymentConfig(
            model_name=self.model_name, logging=self._logging_config()
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

    def _job_config(self, request: TrainingRequest) -> LoraLLMConfig:
        return LoraLLMConfig(
            dataset_id=DATASET_IDS[request.dataset_loader],
            custom_dataset=self._custom_dataset_config(request),
            model_name=self.model_name,
            dtype=request.dtype,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            optim=request.optimizer,
            weight_decay=request.weight_decay,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            training_model_type="lora",
            val_steps_freq=request.val_steps_freq,
            logging=self._logging_config(),
            metrics=MetricsConfig(
                steps_freq=request.steps_freq,
                epoch_freq=1,
                train_metrics=["loss"],
                val_metrics=["loss"],
            ),
            checkpoint=CheckpointConfig(
                steps_freq=max(request.save_interval, 1),
                epoch_freq=1,
                save_strategy="none",
                project_dir=request._output_model_path or "",
                final_checkpoint_name="adapter",
                save_optim=False,
                keep_last_n=0,
                keep_best_n=0,
                checkpoint_metric="val_loss",
                checkpoint_metric_mode="min",
                storage_backend="local",
                sync_to_storage=False,
                load_from_storage=False,
                remote_path="",
                resume_from_checkpoint=False,
                resume_option="last",
                checkpoint_path="",
            ),
            framework="pytorch",
            seed=0,
            deterministic=False,
            use_tt=True,
            mesh_shape=None,
            mesh_axis_names=None,
            input_sharding_dim=None,
            model_sharding_patterns=None,
            max_length=request.dataset_max_sequence_length,
            ignored_index=request.ignored_index,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_target_modules=request.lora_target_modules,
            lora_task_type=request.lora_task_type,
        )
