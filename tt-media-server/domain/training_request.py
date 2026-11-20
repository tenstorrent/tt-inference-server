# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


import time
from multiprocessing import Event
from typing import Annotated, Literal, Optional, Union

from config.constants import DeviceTypes, TrainingOptimizers
from config.settings import settings
from domain.base_request import BaseRequest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

# The request mirrors ttml's SFTTrainerConfig (tt-metal's
# ttml.experimental.trainer.configs), minus what the server sets: model.name, the
# device and the checkpoint directory. Unknown fields are rejected, as ttml
# rejects them, so a misspelled or renamed field fails instead of training
# with its default.


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraOptions(_Options):
    r: int = Field(default=4, gt=0)
    alpha: int = Field(default=8, gt=0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        min_length=1,
    )
    task_type: str = "CAUSAL_LM"


class ModelOptions(_Options):
    # null trains every weight (full fine-tuning), as in ttml.
    lora: Optional[LoraOptions] = Field(default_factory=LoraOptions)


class CustomDatasetOptions(_Options):
    # Anything `datasets.load_dataset` reads ("json", "csv", ...); "jsonl" is read as "json".
    file_type: str = "json"
    train_dataset_path: str = Field(min_length=1)
    # No validation when unset.
    val_dataset_path: Optional[str] = None
    template: str = "alpaca"
    # Template field -> dataset column, for columns not named like the template's fields.
    column_mapping: Optional[dict[str, str]] = None


class DatasetOptions(_Options):
    # "sst2", "alpaca" or "custom", matched case-insensitively as ttml matches them.
    id: str = Field(default="sst2", min_length=1)
    custom_options: Optional[CustomDatasetOptions] = None

    @model_validator(mode="after")
    def _check_custom_options(self) -> "DatasetOptions":
        if self.id.lower() == "custom" and self.custom_options is None:
            raise ValueError("custom_options is required when id is 'custom'")
        return self


class OptimizerOptions(_Options):
    type: str = TrainingOptimizers.ADAMW.value
    lr: float = Field(default=2e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)


class StoppingCriteria(_Options):
    # "epoch": passes over the train data. "step": optimizer steps.
    type: Literal["epoch", "step"] = "step"
    limit: int = Field(default=500, gt=0)


class TrainMetricsOptions(_Options):
    frequency: int = Field(default=10, ge=1)


class ValidationMetricsOptions(_Options):
    # 0 disables validation.
    frequency: int = Field(default=50, ge=0)


class MetricsOptions(_Options):
    train: TrainMetricsOptions = Field(default_factory=TrainMetricsOptions)
    validation: ValidationMetricsOptions = Field(
        default_factory=ValidationMetricsOptions
    )


class StepSave(_Options):
    type: Literal["step"] = "step"
    frequency: int = Field(default=100, ge=1)


class NoSave(_Options):
    type: Literal["none"] = "none"


SaveStrategy = Annotated[Union[StepSave, NoSave], Field(discriminator="type")]


class CheckpointOptions(_Options):
    save_strategy: SaveStrategy = Field(default_factory=StepSave)


class TrainingRequest(BaseRequest):
    model_config = ConfigDict(extra="forbid")

    model: ModelOptions = Field(default_factory=ModelOptions)
    dataset: DatasetOptions = Field(default_factory=DatasetOptions)
    optimizer: OptimizerOptions = Field(default_factory=OptimizerOptions)
    dtype: str = "bfloat16"

    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    seed: int = Field(default=0, ge=0, lt=2**32)
    # Every batch is padded to this many tokens; longer examples are dropped.
    max_length: int = Field(default=64, gt=0)
    stopping_criteria: StoppingCriteria = Field(default_factory=StoppingCriteria)

    metrics: MetricsOptions = Field(default_factory=MetricsOptions)
    checkpoint: CheckpointOptions = Field(default_factory=CheckpointOptions)

    device_type: str = DeviceTypes.P150.value

    _output_model_path: str = PrivateAttr(default=None)
    _start_event: Event = PrivateAttr(default=None)
    _cancel_event: Event = PrivateAttr(default=None)
    _training_metrics: list = PrivateAttr(default=None)
    _training_logs: list = PrivateAttr(default=None)
    _training_checkpoints: list = PrivateAttr(default=None)
    _progress_tracker: object = PrivateAttr(default=None)
    _worker_assignment: object = PrivateAttr(default=None)
    # Worker-local timestamp used to throttle writes to the shared tracker.
    _last_heartbeat_time: float = PrivateAttr(default=0.0)

    def touch_progress(self) -> None:
        """Publish a throttled runner heartbeat to the API process."""
        if self._progress_tracker is None:
            return
        now = time.monotonic()
        if (
            now - self._last_heartbeat_time
            < settings.training_progress_heartbeat_interval_seconds
        ):
            return
        self._progress_tracker.value = now
        self._last_heartbeat_time = now
