# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from domain.base_request import BaseRequest
from typing import Optional
from pydantic import model_validator
from config.settings import settings
from pydantic import PrivateAttr

class TrainingRequest(BaseRequest):
    batch_size: int = 4
    learning_rate: float = 6e-5
    num_epochs: int = 1
    val_steps_freq: int = 50
    steps_freq: int = 10

    dtype: str = "torch.bfloat16"

    lora_r: int = 4
    lora_alpha: int = 8
    lora_target_modules: list[str] = ["q_proj", "v_proj"]
    lora_task_type: str = "CAUSAL_LM"

    ignored_index: int = -100

    _output_model_path: str = PrivateAttr(default=None)
    _dataset_loader: str = PrivateAttr(default=None)
    _dataset_max_length: int = PrivateAttr(default=None)