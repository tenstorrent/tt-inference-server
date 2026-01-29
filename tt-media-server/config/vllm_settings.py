# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import SupportedModels
from pydantic import BaseModel


class VLLMSettings(BaseModel):
    model: str = SupportedModels.BGE_LARGE_EN_V1_5.value
    min_context_length: int = 32
    max_model_length: int = 2048
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 1  # tt-xla only supports max_num_seqs=1 currently
    gpu_memory_utilization: float = 0.1
