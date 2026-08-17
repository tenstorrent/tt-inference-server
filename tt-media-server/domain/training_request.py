# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


from multiprocessing import Event

from config.constants import DatasetLoaders, DeviceTypes, TrainingOptimizers
from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr


class TrainingRequest(BaseRequest):
    dataset_loader: str = DatasetLoaders.SST2.value
    dataset_max_sequence_length: int = 64

    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 1
    val_steps_freq: int = 50
    steps_freq: int = 10

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
