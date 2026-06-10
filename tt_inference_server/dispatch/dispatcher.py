# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KernelDispatcher — shape-keyed kernel instance cache."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .compat import (
    resolve_attn_config,
    resolve_norm_config,
    resolve_swiglu_config,
    ShapeNotSupportedError,
    TILE,
    to_tiles,
)


class KernelDispatcher:
    """Dispatches tensor shapes to cached, compiled kernel instances."""

    def __init__(self, device, model_config=None):
        self._device = device
        self._model_config = model_config
        self._cache: Dict[Tuple, Any] = {}

    def swiglu(self, gate, w_gate, bias_gate, up, w_up, bias_up, out, activation="silu"):
        from tt_inference_server.kernels.swiglu import make_swiglu_kernel
        M, K, N = gate.shape[0], gate.shape[1], w_gate.shape[1]
        key = ("swiglu", M, K, N, activation)
        if key not in self._cache:
            cfg = resolve_swiglu_config(M, K, N, activation)
            self._cache[key] = make_swiglu_kernel(**cfg)
        return self._cache[key](gate, w_gate, bias_gate, up, w_up, bias_up, out)

    def flash_attn(self, Q, K, V, scale, neg_inf, zero, zero_head, ones, mask, out,
                   N_heads: int = None, N_kv_heads: int = None):
        head_dim = Q.shape[-1]
        n_heads = N_heads or Q.shape[0]
        n_kv = N_kv_heads or K.shape[0]
        kv_seq = K.shape[0] // n_kv if n_kv else K.shape[0]
        key = ("flash_attn", n_heads, n_kv, head_dim)
        if key not in self._cache:
            from tt_inference_server.kernels.flash_attn import make_flash_attn_kernel
            cfg = resolve_attn_config(n_heads, n_kv, head_dim, kv_seq)
            self._cache[key] = make_flash_attn_kernel(**cfg)
        return self._cache[key](Q, K, V, scale, neg_inf, zero, zero_head, ones, mask, out)

    def kv_decode(self, Q, K_cache, V_cache, scale, neg_inf, zero, zero_head, ones, out):
        from tt_inference_server.kernels.kv_decode import make_kv_decode_kernel
        N_kv_heads, head_dim = Q.shape[0], Q.shape[1]
        max_seq = K_cache.shape[0] // N_kv_heads
        key = ("kv_decode", N_kv_heads, head_dim, max_seq)
        if key not in self._cache:
            self._cache[key] = make_kv_decode_kernel(
                N_kv_heads=N_kv_heads,
                head_dim_tiles=to_tiles(head_dim),
                max_seq_tiles=to_tiles(max_seq),
            )
        return self._cache[key](Q, K_cache, V_cache, scale, neg_inf, zero, zero_head, ones, out)

    def rmsnorm(self, x, weight, scaler, out):
        from tt_inference_server.kernels.rmsnorm import make_rmsnorm_kernel
        seq, hidden = x.shape[0], x.shape[1]
        key = ("rmsnorm", seq, hidden)
        if key not in self._cache:
            cfg = resolve_norm_config(seq, hidden)
            self._cache[key] = make_rmsnorm_kernel(**cfg)
        return self._cache[key](x, weight, scaler, out)

    def layernorm(self, x, weight, bias, scaler, out):
        from tt_inference_server.kernels.layernorm import make_layernorm_kernel
        seq, hidden = x.shape[0], x.shape[1]
        key = ("layernorm", seq, hidden)
        if key not in self._cache:
            cfg = resolve_norm_config(seq, hidden)
            self._cache[key] = make_layernorm_kernel(**cfg)
        return self._cache[key](x, weight, bias, scaler, out)

    def moe_route(self, hidden, w_router, ones, probs_out, N_experts: int, top_k: int):
        """Run router projection + softmax on device, then top-k on host."""
        import torch
        import ttnn
        hidden_dim = hidden.shape[1]
        expert_tiles = max(1, (N_experts + 31) // 32)
        key = ("moe_router", hidden_dim, N_experts)
        if key not in self._cache:
            from tt_inference_server.kernels.moe_router import make_moe_router_kernel
            self._cache[key] = make_moe_router_kernel(
                hidden_tiles=to_tiles(hidden_dim),
                expert_tiles=expert_tiles,
            )
        self._cache[key](hidden, w_router, ones, probs_out)
        probs = ttnn.to_torch(probs_out).float()[:, :N_experts]
        weights, indices = torch.topk(probs, k=top_k, dim=-1)
        return indices, weights

    def cache_size(self) -> int:
        return len(self._cache)
