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
    # README already documents the GPU_MEMORY_UTILIZATION knob; this re-wires
    # the field that became hardcoded at some point. Raise above 0.1 when
    # boosting max_num_seqs — KV cache scales linearly with batch.
    gpu_memory_utilization: float = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.1))
