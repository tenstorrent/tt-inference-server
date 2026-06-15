# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape constants for Mistral and Mixtral model families."""

from dataclasses import dataclass

try:
    from .config_qwen2_5 import ModelConfig
except ImportError:
    from .config_qwen2_5 import ModelConfig  # script mode


@dataclass
class MoEModelConfig(ModelConfig):
    num_experts: int = 8
    num_experts_per_tok: int = 2


# Per-size dim tables (CONFIGS) were retired in #3 Phase D — dims for listed models now
# live in model_matrix.toml; novel models derive via ModelConfig.from_hf_config.
# MoEModelConfig is retained: detect_model_family builds it for MoE archs from hf_config.
