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
    # Fraction of TT-device memory allocated for model weights + KV cache.
    # Env-driven so it can be flipped per-run without rebuilding the image.
    # README documents the GPU_MEMORY_UTILIZATION knob (bare, not VLLM__-
    # prefixed — VLLMSettings is a BaseModel, not BaseSettings, and reads
    # env via os.environ.get only). Empty / whitespace env values fall back
    # to the 0.1 default to avoid `float("")` crashing module import.
    _gmu_env = os.environ.get("GPU_MEMORY_UTILIZATION", "").strip()
    gpu_memory_utilization: float = float(_gmu_env) if _gmu_env else 0.1
