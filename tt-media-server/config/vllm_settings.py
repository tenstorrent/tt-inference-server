# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pydantic import BaseModel


class VLLMSettings(BaseModel):
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    min_context_length: int = 32
    max_model_length: int = 65536
    max_num_batched_tokens: int = 2048
    seed: int = 9472
    max_num_seqs: int = 32  # tt-xla only supports max_num_seqs=1 currently
    gpu_memory_utilization: float = 1
