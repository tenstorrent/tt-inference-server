# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

from config.constants import SupportedModels
from pydantic import BaseModel


class VLLMSettings(BaseModel):
    model: str = SupportedModels.QWEN_3_4B.value
    min_context_length: int = 32
    max_model_length: int = int(os.environ.get("MAX_MODEL_LENGTH", 4096))
    max_num_seqs: int = int(os.environ.get("MAX_NUM_SEQS", 1))
    max_num_batched_tokens: int = max_model_length * max_num_seqs
    gpu_memory_utilization: float = 0.1
    # When False (default), sampling runs on the TT device (vllm-tt's on-device
    # sampler path); when True, sampling runs on CPU. Env-driven so it can be
    # flipped per-run without rebuilding the image. See
    # debug-docs/sampling_params_on_device_analysis.md.
    cpu_sampling: bool = os.environ.get("CPU_SAMPLING", "false").lower() != "false"
