# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape validation, padding, and L1 budget checking.

Primary job: given a tensor shape, determine whether a kernel config is
valid and if not, find the nearest valid config. Prioritizes correctness
over throughput — any working config is better than a compiler error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

TILE = 32
BF16_TILE_BYTES = 32 * 32 * 2  # 2048 bytes per bf16 tile
# Fallback when no HardwareConfig is provided — matches the p150 91.5% budget.
_DEFAULT_L1_BUDGET_BYTES = 1_439_170


class ShapeNotSupportedError(ValueError):
    """Raised when no valid kernel config can be found for a given shape."""


@dataclass
class PaddedShape:
    original: Tuple[int, ...]
    padded: Tuple[int, ...]
    tiles: Tuple[int, ...]

    @property
    def needed_padding(self) -> bool:
        return self.original != self.padded


def tile_align(dim: int) -> int:
    """Round dim up to the nearest multiple of TILE."""
    return math.ceil(dim / TILE) * TILE


def to_tiles(dim: int) -> int:
    return tile_align(dim) // TILE


def pad_shape(shape: Tuple[int, ...]) -> PaddedShape:
    """Return a PaddedShape that is tile-aligned in every dimension."""
    padded = tuple(tile_align(d) for d in shape)
    tiles = tuple(d // TILE for d in padded)
    return PaddedShape(original=shape, padded=padded, tiles=tiles)


def l1_bytes_for_attn(
    head_dim_tiles: int,
    kv_chunk_tiles: int,
    n_intermediate_dfbs: int = 17,
    block_count: int = 2,
) -> int:
    """Estimate L1 CB footprint for a flash attention config."""
    tiles_per_head = head_dim_tiles * block_count
    tiles_per_chunk = kv_chunk_tiles * block_count
    # Rough budget: head-sized buffers + chunk-sized buffers + scalar buffers
    head_bufs = 8  # q, k, v, kt, out, o_corr, pv, l_bcast
    chunk_bufs = 5  # qk, scaled, exp, m_bcast, mask
    scalar_bufs = 7  # m, l, alpha, m_new, chunk_max, chunk_sum, scale/ninf/zero
    total_tiles = (
        head_bufs * tiles_per_head
        + chunk_bufs * tiles_per_chunk
        + scalar_bufs * block_count
    )
    return total_tiles * BF16_TILE_BYTES


def l1_bytes_for_swiglu(
    M_tiles: int,
    K_tiles: int,
    N_tiles: int,
    block_count: int = 2,
) -> int:
    """Estimate L1 CB footprint for a SwiGLU config."""
    # 8 DFBs: gate, w_gate, bias_gate, up, w_up, bias_up, act, out
    mk = M_tiles * K_tiles
    kn = K_tiles * N_tiles
    mn = M_tiles * N_tiles
    total_tiles = (2 * mk + 2 * kn + 4 * mn) * block_count
    return total_tiles * BF16_TILE_BYTES


def resolve_attn_config(
    N_heads: int,
    N_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    hw_config=None,
) -> Dict:
    """Find a valid flash attention config for the given model dimensions.

    Returns a dict of kwargs for make_flash_attn_kernel (includes hw_config).
    Raises ShapeNotSupportedError if no valid config exists.
    """
    l1_budget = hw_config.l1_budget() if hw_config is not None else _DEFAULT_L1_BUDGET_BYTES
    head_dim_padded = tile_align(head_dim)
    head_dim_tiles = head_dim_padded // TILE

    # Try kv_chunk_tiles from 1 up to 4 — larger chunks use more L1
    for kv_chunk_tiles in (1, 2, 4):
        l1 = l1_bytes_for_attn(head_dim_tiles, kv_chunk_tiles)
        if l1 <= l1_budget:
            return {
                "N_heads": N_heads,
                "N_kv_heads": N_kv_heads,
                "head_dim_tiles": head_dim_tiles,
                "kv_chunk_tiles": kv_chunk_tiles,
                "hw_config": hw_config,
            }

    raise ShapeNotSupportedError(
        f"Flash attention config not feasible: head_dim={head_dim} requires "
        f"{l1_bytes_for_attn(head_dim_tiles, 1) // 1024} KiB L1 but budget is "
        f"{l1_budget // 1024} KiB ({hw_config.name if hw_config else 'default'}). "
        f"Consider reducing head_dim or L1 budget."
    )


def resolve_swiglu_config(
    M: int,
    K: int,
    N: int,
    activation: str = "silu",
    hw_config=None,
) -> Dict:
    """Find a valid SwiGLU config for the given dimensions.

    Returns a dict of kwargs for make_swiglu_kernel.
    Raises ShapeNotSupportedError if no valid config exists.
    """
    l1_budget = hw_config.l1_budget() if hw_config is not None else _DEFAULT_L1_BUDGET_BYTES
    M_tiles = to_tiles(M)
    K_tiles = to_tiles(K)
    N_tiles = to_tiles(N)

    l1 = l1_bytes_for_swiglu(M_tiles, K_tiles, N_tiles)
    if l1 > l1_budget:
        raise ShapeNotSupportedError(
            f"SwiGLU config not feasible: M={M} K={K} N={N} requires "
            f"{l1 // 1024} KiB L1 but budget is {l1_budget // 1024} KiB. "
            f"Consider reducing batch/sequence dimension."
        )

    return {
        "M_tiles": M_tiles,
        "K_tiles": K_tiles,
        "N_tiles": N_tiles,
        "activation": activation,
    }


def resolve_norm_config(seq_len: int, hidden_dim: int) -> Dict:
    """Return tile dims for a norm kernel (RMSNorm or LayerNorm)."""
    return {
        "seq_tiles": to_tiles(seq_len),
        "hidden_tiles": to_tiles(hidden_dim),
    }
