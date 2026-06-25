# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape constants for the Qwen 3 model family."""

try:
    from .config_qwen2_5 import ModelConfig
except ImportError:
    from .config_qwen2_5 import ModelConfig  # script mode

# Per-size dim tables (CONFIGS) were retired in #3 Phase D — dims for listed models now
# live in model_matrix.toml; novel models derive via ModelConfig.from_hf_config. This
# module is kept so registry._FAMILY_MAP can resolve Qwen3ForCausalLM to ModelConfig.
__all__ = ["ModelConfig"]
