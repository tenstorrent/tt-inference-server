# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import torch

from config.constants import TrainingOptimizers

OPTIMIZER_MAP = {
    TrainingOptimizers.ADAMW.value: torch.optim.AdamW,
}

DTYPE_MAP = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}', must be one of {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[dtype_str]
