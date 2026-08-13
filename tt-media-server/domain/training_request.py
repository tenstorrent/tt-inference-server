# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


from multiprocessing import Event
from typing import Optional

from config.constants import DatasetLoaders, DeviceTypes, TrainingOptimizers
from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr, model_validator


class TrainingRequest(BaseRequest):
    dataset_loader: str = DatasetLoaders.SST2.value
    dataset_max_sequence_length: int = 64

    train_dataset_path: Optional[str] = None
    val_dataset_path: Optional[str] = None
    file_type: Optional[str] = None
    template: Optional[str] = None
    column_mapping: Optional[dict[str, str]] = None

    @model_validator(mode="after")
    def check_custom_dataset(self) -> "TrainingRequest":
        if self.dataset_loader != DatasetLoaders.CUSTOM.value:
            return self
        missing = [
            name
            for name, value in (
                ("train_dataset_path", self.train_dataset_path),
                ("file_type", self.file_type),
                ("template", self.template),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{', '.join(repr(n) for n in missing)} required when "
                f"dataset_loader is '{DatasetLoaders.CUSTOM.value}'"
            )
        return self

    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = Field(default=0.01, ge=0.0)
    num_epochs: int = 1
    val_steps_freq: int = 50
    steps_freq: int = 10
    gradient_accumulation_steps: int = Field(default=1, ge=1)

    dtype: str = "torch.bfloat16"

    lora_r: int = 4
    lora_alpha: int = 8
    lora_target_modules: list[str] = ["q_proj", "v_proj"]
    lora_task_type: str = "CAUSAL_LM"

    ignored_index: int = -100

    device_type: str = DeviceTypes.P150.value
    optimizer: str = TrainingOptimizers.ADAMW.value

    save_interval: int = Field(default=100, ge=0)
    max_steps: int = Field(default=500, ge=0)

    _output_model_path: str = PrivateAttr(default=None)
    _start_event: Event = PrivateAttr(default=None)
    _cancel_event: Event = PrivateAttr(default=None)
    _training_metrics: list = PrivateAttr(default=None)
    _training_logs: list = PrivateAttr(default=None)
    _training_checkpoints: list = PrivateAttr(default=None)
