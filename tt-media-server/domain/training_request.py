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

    # Only read when dataset_loader is Custom. Paths are resolved against, and
    # confined to, the server's training_datasets_dir. Validation is optional;
    # without it the run reports training loss alone. The column mapping names
    # which dataset columns fill the prompt template's instruction, input and
    # output slots, and can be omitted when they already carry those names.
    train_dataset_path: Optional[str] = None
    val_dataset_path: Optional[str] = None
    column_mapping: Optional[dict[str, str]] = None

    @model_validator(mode="after")
    def check_custom_dataset(self) -> "TrainingRequest":
        if (
            self.dataset_loader == DatasetLoaders.CUSTOM.value
            and not self.train_dataset_path
        ):
            raise ValueError(
                f"'train_dataset_path' is required when dataset_loader is "
                f"'{DatasetLoaders.CUSTOM.value}'"
            )
        return self

    batch_size: int = 4
    learning_rate: float = 2e-5
    # torch.optim.AdamW's own default, which is what the training-lora runner
    # ends up with since it only passes `lr`.
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
