# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import random

import numpy as np
import torch
from config.constants import TrainingOptimizers

OPTIMIZER_MAP = {
    TrainingOptimizers.ADAMW.value: torch.optim.AdamW,
}


def seed_rngs(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs.

    Covers host-side randomness (dataset shuffling, LoRA init, dropout);
    device-side TT RNG is not seeded here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
