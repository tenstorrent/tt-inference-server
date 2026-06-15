# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

from config.constants import SupportedModels
from pydantic import BaseModel


class VLLMSettings(BaseModel):
    model: str = SupportedModels.QWEN_3_4B.value
    min_context_length: int = 128
    max_model_length: int = int(os.environ.get("MAX_MODEL_LENGTH", 4096))
    max_num_seqs: int = int(os.environ.get("MAX_NUM_SEQS", 1))
    # When chunked prefill is engaged (PREFILL_CHUNK_SIZE set, e.g. forge LLMs),
    # the prefill is processed in chunks, so the batched-token budget only needs
    # to cover batch_size * prefill_chunk_size rather than the full context per
    # sequence. The smaller budget frees device memory for KV cache, allowing a
    # higher gpu_memory_utilization. Matches the tt-xla benchmark convention
    # (max_num_batched_tokens = batch_size * prefill_chunk_size). Falls back to
    # batch_size * full-context when PREFILL_CHUNK_SIZE is unset (legacy path).
    _prefill_chunk_size_env = os.environ.get("PREFILL_CHUNK_SIZE", "").strip()
    prefill_chunk_size: int = (
        int(_prefill_chunk_size_env) if _prefill_chunk_size_env else 0
    )
    max_num_batched_tokens: int = (
        max_num_seqs * prefill_chunk_size
        if prefill_chunk_size
        else max_model_length * max_num_seqs
    )
    # Fraction of TT-device memory allocated for model weights + KV cache.
    # Env-driven so it can be flipped per-run without rebuilding the image.
    # README documents the GPU_MEMORY_UTILIZATION knob (bare, not VLLM__-
    # prefixed — VLLMSettings is a BaseModel, not BaseSettings, and reads
    # env via os.environ.get only). Empty / whitespace env values fall back
    # to the 0.1 default to avoid `float("")` crashing module import.
    _gmu_env = os.environ.get("GPU_MEMORY_UTILIZATION", "").strip()
    gpu_memory_utilization: float = float(_gmu_env) if _gmu_env else 0.1
