# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for decode-node-poc.

DeepSeek V3 KV cache parameters for P/D disaggregation.
"""

from dataclasses import dataclass, field


@dataclass
class DeepSeekKVConfig:
    """DeepSeek V3 KV cache configuration."""

    num_layers: int = 61
    kvpe_dim: int = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
    block_size: int = 32
    dtype_bytes: int = 1  # bfloat8_b = 1 byte per element

    def get_layer_shape(self, seq_len: int) -> tuple[int, int, int, int]:
        """Get KV cache shape for a single layer."""
        num_blocks = seq_len // self.block_size
        return (num_blocks, 1, self.block_size, self.kvpe_dim)

    def get_layer_bytes(self, seq_len: int) -> int:
        """Get bytes for a single layer's KV cache."""
        shape = self.get_layer_shape(seq_len)
        elements = shape[0] * shape[1] * shape[2] * shape[3]
        return elements * self.dtype_bytes

    def get_total_bytes(self, seq_len: int) -> int:
        """Get total bytes for all layers' KV cache."""
        return self.get_layer_bytes(seq_len) * self.num_layers


@dataclass
class DecodeNodeConfig:
    """Configuration for the decode node POC."""

    # KV cache config
    kv_config: DeepSeekKVConfig = field(default_factory=DeepSeekKVConfig)

    # Test sequence lengths
    test_seq_lengths: list[int] = field(
        default_factory=lambda: [1024, 4096, 8192, 32768]
    )

    # MPI ranks
    prefill_rank: int = 0
    decode_rank: int = 1

    # Communication tags
    REQUEST_TAG: int = 100
    RESPONSE_TAG: int = 200
    KV_LAYER_TAG_BASE: int = 1000  # layer i uses tag KV_LAYER_TAG_BASE + i

    # Number of warmup iterations
    warmup_iterations: int = 1
