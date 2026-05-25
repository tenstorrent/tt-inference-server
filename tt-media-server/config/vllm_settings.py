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
    # When False (default), sampling runs on the TT device (vllm-tt's on-device
    # sampler path); when True, sampling runs on CPU. Env-driven so it can be
    # flipped per-run without rebuilding the image. See
    # debug-docs/sampling_params_on_device_analysis.md.
    cpu_sampling: bool = os.environ.get("CPU_SAMPLING", "false").lower() != "false"
    # Enables vllm-tt's trace-capture compile mode (precompiled forward graph).
    # tt-xla's own b32 benchmarks universally set enable_trace=True; our
    # additional_config was missing this knob, which appears to be the wall
    # for batch=32 Llama-3.2-3B engine init (`_xla_warm_up_cache` Error
    # code: 13). Default True to match upstream; env-flippable for triage.
    enable_trace: bool = os.environ.get("ENABLE_TRACE", "true").lower() != "false"
    # vllm-tt constant-evaluation pass — bakes immutable weights into the
    # compiled graph as constants. Hardcoded to True historically; tt-xla's
    # working b32 configs do NOT set this. Env-driven so we can flip it for
    # b32 triage. Default True preserves current behavior.
    enable_const_eval: bool = (
        os.environ.get("ENABLE_CONST_EVAL", "true").lower() != "false"
    )
    # Experimental weight dtype for vllm-tt (e.g. "bfp_bf8" for block fp8).
    # Hardcoded to "bfp_bf8" historically; tt-xla's working b32 configs do
    # NOT set this. Set EXPERIMENTAL_WEIGHT_DTYPE="" (empty) to omit the
    # key entirely from additional_config (matching tt-xla). Default
    # "bfp_bf8" preserves current behavior.
    experimental_weight_dtype: str = os.environ.get(
        "EXPERIMENTAL_WEIGHT_DTYPE", "bfp_bf8"
    )
