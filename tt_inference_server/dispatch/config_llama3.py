# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape constants for the Llama 3 model family (Meta)."""

try:
    from .config_qwen2_5 import ModelConfig
except ImportError:
    from .config_qwen2_5 import ModelConfig  # script mode

CONFIGS = {
    "8B":  ModelConfig(hidden_size=4096, num_heads=32, num_kv_heads=8,  head_dim=128, intermediate_size=14336),
    "70B": ModelConfig(hidden_size=8192, num_heads=64, num_kv_heads=8,  head_dim=128, intermediate_size=28672),
}
