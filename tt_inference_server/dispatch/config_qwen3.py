# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape constants for the Qwen 3 model family."""

try:
    from .config_qwen2_5 import ModelConfig
except ImportError:
    from .config_qwen2_5 import ModelConfig  # script mode

CONFIGS = {
    "0.6B": ModelConfig(hidden_size=1024, num_heads=16, num_kv_heads=8,  head_dim=64,  intermediate_size=3072),
    "1.7B": ModelConfig(hidden_size=2048, num_heads=16, num_kv_heads=8,  head_dim=128, intermediate_size=6144),
    "4B":   ModelConfig(hidden_size=2560, num_heads=32, num_kv_heads=8,  head_dim=128, intermediate_size=9728),  # noqa: E501
    "8B":   ModelConfig(hidden_size=4096, num_heads=32, num_kv_heads=8,  head_dim=128, intermediate_size=12288),
    "14B":  ModelConfig(hidden_size=5120, num_heads=40, num_kv_heads=8,  head_dim=128, intermediate_size=17408),
    "32B":  ModelConfig(hidden_size=5120, num_heads=64, num_kv_heads=8,  head_dim=80,  intermediate_size=25600),
}
