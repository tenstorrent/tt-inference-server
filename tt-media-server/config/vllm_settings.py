# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import SupportedModels
from pydantic import BaseModel


class VLLMSettings(BaseModel):
    model: str = SupportedModels.LLAMA_3_2_1B.value
    min_context_length: int = 32
    max_model_length: int = 2048
    max_num_seqs: int = 1
    max_num_batched_tokens: int = max_model_length * max_num_seqs
    gpu_memory_utilization: float = 0.1
