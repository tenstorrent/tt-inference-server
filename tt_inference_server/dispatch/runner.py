# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTModelRunner — full on-device inference for HuggingFace transformer models.

Keeps model weights and activations on the TT device throughout the forward
pass. Only crosses PCIe at model boundaries: one embedding lookup per token in,
one logit row per token out.

Usage::

    import ttnn
    from tt_inference_server.dispatch.runner import TTModelRunner

    device = ttnn.open_device(device_id=0)
    runner = TTModelRunner("models/llama3-8b", device)
    print(runner.generate("The capital of France is", max_new_tokens=40))
    ttnn.close_device(device)

CLI::

    python -m tt_inference_server.dispatch.runner --model models/gemma3-4b --prompt "Hello world"
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


TILE = 32


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

# Activation strings mapped to ttnn.gelu — both the matrix's canonical "gelu_tanh"/"gelu"
# and the raw HuggingFace act names that the auto-derive (unlisted) path passes through.
_GELU_ACTS = ("gelu_tanh", "gelu", "gelu_pytorch_tanh", "gelu_new", "gelu_fast")
# Tanh-approximation GELU family -> ttnn.gelu(fast_and_approximate_mode=True).
# Plain "gelu" is exact-erf (GPTNeoX) and uses fast_and_approximate_mode=False.
_GELU_TANH_ACTS = ("gelu_tanh", "gelu_pytorch_tanh", "gelu_new", "gelu_fast")


def _tile_align(n: int) -> int:
    return math.ceil(n / TILE) * TILE


def _pad_to_tiles(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        pc = _tile_align(t.shape[0]) - t.shape[0]
        return torch.nn.functional.pad(t, (0, pc)) if pc else t
    pr = _tile_align(t.shape[-2]) - t.shape[-2]
    pc = _tile_align(t.shape[-1]) - t.shape[-1]
    if pr or pc:
        return torch.nn.functional.pad(t, (0, pc, 0, pr))
    return t


def _to_tt(t: torch.Tensor, device):
    import ttnn
    t = _pad_to_tiles(t.bfloat16().contiguous())
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _pad_qkv_per_head(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Pad each head's output slice to a tile boundary independently.

    w shape: (in_dim, n_heads * head_dim) — weight already transposed.
    Returns: (in_dim, n_heads * hd_p) where hd_p = tile_align(head_dim).

    Required when head_dim is not a multiple of 32 (e.g. BLOOM head_dim=80).
    Without this the view in _attention that reshapes to (n_heads, hd_p) fails
    because n_heads * head_dim ≠ n_heads * hd_p.
    """
    hd_p = _tile_align(head_dim)
    if hd_p == head_dim:
        return w                               # already aligned — no-op
    in_dim = w.shape[0]
    # (in_dim, n_heads*head_dim) → (n_heads, head_dim, in_dim) → pad → back
    w_heads = w.T.reshape(n_heads, head_dim, in_dim)          # (n_heads, hd, in)
    pad_amt = hd_p - head_dim
    w_padded = torch.nn.functional.pad(w_heads, (0, 0, 0, pad_amt))   # (n_heads, hd_p, in)
    return w_padded.reshape(n_heads * hd_p, in_dim).T.contiguous()    # (in, n_heads*hd_p)


def _pad_o_proj_per_head(w_raw: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Pad the per-head input dimension of the O projection weight for tile alignment.

    The flash-attn output is (TILE, n_heads * hd_p) where hd_p = tile_align(head_dim).
    When head_dim is not tile-aligned (e.g. 80 → 96), the O-proj weight must accept
    n_heads * hd_p inputs instead of n_heads * head_dim.

    w_raw: (out_dim, n_heads * head_dim) — raw HF weight, pre-transpose
    Returns: (out_dim, n_heads * hd_p)
    """
    hd_p = _tile_align(head_dim)
    if hd_p == head_dim:
        return w_raw
    out_dim = w_raw.shape[0]
    w = w_raw.reshape(out_dim, n_heads, head_dim)
    w_padded = torch.nn.functional.pad(w, (0, hd_p - head_dim))
    return w_padded.reshape(out_dim, n_heads * hd_p).contiguous()


# CPU-fallback tracking. _matmul_safe falls back to torch.matmul on L1 overflow —
# correct, but it defeats the "hardware-native" goal, so the test harness treats any
# fallback as a FAILURE (a model that only "works" on CPU isn't running on the card).
# Counter is module-level because _matmul_safe is a free function.
_CPU_FALLBACK_COUNT = 0


def reset_cpu_fallback_count():
    global _CPU_FALLBACK_COUNT
    _CPU_FALLBACK_COUNT = 0


def get_cpu_fallback_count() -> int:
    return _CPU_FALLBACK_COUNT


def _matmul_safe(a, b, device, output_tensor=None):
    """ttnn.matmul with CPU fallback on L1 overflow.

    Some hidden sizes (e.g. 2560) exceed the 1.5 MB per-core Tensix SRAM limit
    for static circular buffers. When that happens we fall back to torch.matmul
    on CPU and re-upload the result. This is slower but correct.

    output_tensor: pre-allocated device tensor to write result into (avoids mmap).
    Ignored in the CPU fallback path and when ttnn doesn't support the parameter.
    """
    import ttnn
    try:
        if output_tensor is not None:
            # ttnn >= 0.71 uses optional_output_tensor; older builds use output_tensor
            try:
                return ttnn.matmul(a, b, optional_output_tensor=output_tensor)
            except TypeError:
                return ttnn.matmul(a, b, output_tensor=output_tensor)
        return ttnn.matmul(a, b)
    except Exception as e:
        msg = str(e)
        if "1572864" not in msg and "Statically allocated circular buffers" not in msg:
            raise
        # L1 overflow — fall back to CPU matmul (output_tensor not used on CPU path).
        # Count it: the harness treats any CPU fallback as a failure (not hardware-native).
        global _CPU_FALLBACK_COUNT
        _CPU_FALLBACK_COUNT += 1
        a_cpu = ttnn.to_torch(a).float()
        b_cpu = ttnn.to_torch(b).float()
        result = torch.matmul(a_cpu, b_cpu).bfloat16()
        return _to_tt(result, device)


def _getattr_nested(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _find_text_backbone(hf_model):
    """Return (backbone, layers_attr) for the text decoder layers.

    layers_attr is either 'layers' (most models) or 'h' (BLOOM, Falcon).
    """
    for path in ["model", "language_model.model", "language_model",
                 "transformer", "gpt_neox", "decoder"]:
        obj = _getattr_nested(hf_model, path)
        if obj is None:
            continue
        if hasattr(obj, "layers"):
            return obj, "layers"
        if hasattr(obj, "h"):
            return obj, "h"
    raise RuntimeError(f"Cannot find text backbone in {type(hf_model).__name__}")


def _find_embed_tokens(backbone):
    """Find embedding table on the backbone regardless of attribute name."""
    for attr in ("embed_tokens", "embed_in", "word_embeddings", "wte"):
        m = getattr(backbone, attr, None)
        if m is not None:
            return m
    raise RuntimeError(f"Cannot find embedding table in {type(backbone).__name__}")


def _find_final_norm(backbone):
    """Find final layer norm regardless of attribute name."""
    for attr in ("norm", "final_layer_norm", "ln_f", "model_norm"):
        m = getattr(backbone, attr, None)
        if m is not None:
            return m
    raise RuntimeError(f"Cannot find final norm in {type(backbone).__name__}")


def _find_layer_attn(layer):
    """Find the attention sub-module for a decoder layer."""
    for attr in ("self_attn", "attention", "self_attention"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError(f"Cannot find attention in {type(layer).__name__}")


def _find_layer_mlp(layer):
    """Find the MLP sub-module for a decoder layer."""
    for attr in ("mlp", "feed_forward", "ff"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError(f"Cannot find MLP in {type(layer).__name__}")


def _find_layer_norms(layer):
    """Return (norm1, norm2, norm3, norm4) weight modules for a decoder layer.

    norm1 = pre-attention, norm2 = post-attention (pre-MLP),
    norm3/norm4 = Gemma3 extra norms (None otherwise).
    """
    n1 = (getattr(layer, "input_layernorm", None) or
          getattr(layer, "attn_norm", None) or
          getattr(layer, "ln_1", None) or
          getattr(layer, "norm_1", None))
    n2 = (getattr(layer, "post_attention_layernorm", None) or
          getattr(layer, "ffn_norm", None) or
          getattr(layer, "ln_2", None) or
          getattr(layer, "norm_2", None))
    n3 = getattr(layer, "pre_feedforward_layernorm", None)
    n4 = getattr(layer, "post_feedforward_layernorm", None)
    if n1 is None or n2 is None:
        raise RuntimeError(
            f"Cannot find layer norms in {type(layer).__name__}: {dir(layer)}")
    return n1, n2, n3, n4


def _split_qkv_weights(attn, cfg):
    """Return (q_cpu, k_cpu, v_cpu) bfloat16 weight tensors shaped (in, out).

    Handles separate q/k/v projections and all common fused-QKV variants:
    - query_key_value (GPTNeoX, BLOOM): interleaved per head [Q_0,K_0,V_0, Q_1,K_1,V_1, ...]
    - qkv_proj (Phi-3):                block layout [Q_all; K_all; V_all]
    - wqkv (InternLM2):                block layout [Q_all; K_all; V_all]
    - c_attn (GPT-2 style):            block layout [Q_all; K_all; V_all]
    """
    qkv_interleaved = getattr(attn, "query_key_value", None)
    if qkv_interleaved is not None:
        # GPTNeoX / BLOOM: rows interleaved per head [Q_0,K_0,V_0, Q_1,K_1,V_1, ...]
        w = qkv_interleaved.weight.detach().bfloat16()  # (n_heads*3*head_dim, in)
        n_h  = cfg.num_heads
        n_kv = cfg.num_kv_heads
        hd   = cfg.head_dim
        # Reshape to (n_heads, 3, head_dim, in), then extract Q/K/V per head
        w_r = w.view(n_h, 3, hd, w.shape[1])
        q = w_r[:, 0, :, :].reshape(n_h  * hd, w.shape[1]).T.contiguous()
        k = w_r[:n_kv, 1, :, :].reshape(n_kv * hd, w.shape[1]).T.contiguous()
        v = w_r[:n_kv, 2, :, :].reshape(n_kv * hd, w.shape[1]).T.contiguous()
        return q, k, v

    # Block-layout fused QKV — [Q_all; K_all; V_all]
    fused = (getattr(attn, "qkv_proj", None) or
             getattr(attn, "wqkv", None) or
             getattr(attn, "c_attn", None))
    if fused is not None:
        w = fused.weight.detach().bfloat16()   # (q_out+k_out+v_out, in)
        q_out = cfg.num_heads    * cfg.head_dim
        k_out = cfg.num_kv_heads * cfg.head_dim
        v_out = cfg.num_kv_heads * cfg.head_dim
        total = q_out + k_out + v_out
        if w.shape[0] != total:
            # Some models pad — take the first total rows
            w = w[:total]
        q = w[:q_out].T.contiguous()
        k = w[q_out:q_out + k_out].T.contiguous()
        v = w[q_out + k_out:].T.contiguous()
        return q, k, v

    # Standard separate projections
    q = attn.q_proj.weight.detach().bfloat16().T.contiguous()
    k = attn.k_proj.weight.detach().bfloat16().T.contiguous()
    v = attn.v_proj.weight.detach().bfloat16().T.contiguous()
    return q, k, v


def _get_qkv_bias(attn, cfg) -> "torch.Tensor | None":
    """Return concatenated (q_b, k_b, v_b) CPU float bias, or None.

    Handles interleaved (query_key_value) and block (qkv_proj etc.) layouts.
    The returned tensor has the same head-dim padding as the weight matrices
    produced by _split_qkv_weights / _pad_qkv_per_head: each head's slice is
    zero-padded from head_dim to tile_align(head_dim).
    """
    hd   = cfg.head_dim
    hd_p = _tile_align(hd)
    n_h  = cfg.num_heads
    n_kv = cfg.num_kv_heads

    def _pad_bias_heads(b_raw, n_heads):
        """Pad (n_heads * head_dim,) → (n_heads * hd_p,)."""
        if hd_p == hd:
            return b_raw.float()
        bh = b_raw.float().view(n_heads, hd)
        return torch.nn.functional.pad(bh, (0, hd_p - hd)).view(n_heads * hd_p)

    qkv_interleaved = getattr(attn, "query_key_value", None)
    if qkv_interleaved is not None:
        b = getattr(qkv_interleaved, "bias", None)
        if b is None:
            return None
        b = b.detach()
        # Interleaved: [Q_0,K_0,V_0, Q_1,K_1,V_1, ...]
        b_r = b.view(n_h, 3, hd)
        q_b = _pad_bias_heads(b_r[:, 0, :].reshape(n_h  * hd), n_h)
        k_b = _pad_bias_heads(b_r[:n_kv, 1, :].reshape(n_kv * hd), n_kv)
        v_b = _pad_bias_heads(b_r[:n_kv, 2, :].reshape(n_kv * hd), n_kv)
        return torch.cat([q_b, k_b, v_b])

    fused = (getattr(attn, "qkv_proj", None) or getattr(attn, "wqkv", None) or
             getattr(attn, "c_attn", None))
    if fused is not None:
        b = getattr(fused, "bias", None)
        if b is None:
            return None
        b = b.detach()
        q_out = n_h  * hd
        k_out = n_kv * hd
        q_b = _pad_bias_heads(b[:q_out], n_h)
        k_b = _pad_bias_heads(b[q_out:q_out + k_out], n_kv)
        v_b = _pad_bias_heads(b[q_out + k_out:q_out + k_out + k_out], n_kv)
        return torch.cat([q_b, k_b, v_b])

    # Separate q/k/v — each may have its own bias
    def _get_sep_bias(mod_name):
        m = getattr(attn, mod_name, None)
        b = getattr(m, "bias", None) if m else None
        return b.detach().float() if b is not None else None
    q_b = _get_sep_bias("q_proj")
    k_b = _get_sep_bias("k_proj")
    v_b = _get_sep_bias("v_proj")
    if q_b is None and k_b is None and v_b is None:
        return None
    q_b2 = _pad_bias_heads(q_b, n_h)  if q_b is not None else torch.zeros(n_h  * hd_p)
    k_b2 = _pad_bias_heads(k_b, n_kv) if k_b is not None else torch.zeros(n_kv * hd_p)
    v_b2 = _pad_bias_heads(v_b, n_kv) if v_b is not None else torch.zeros(n_kv * hd_p)
    return torch.cat([q_b2, k_b2, v_b2])


def _find_o_proj(attn):
    """Find the output-projection module regardless of attribute name."""
    for attr in ("o_proj", "out_proj", "dense", "wo", "c_proj"):
        m = getattr(attn, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m
    raise RuntimeError(f"Cannot find output projection in {type(attn).__name__}")


def _find_o_proj_weight(attn):
    """Find output projection weight regardless of attribute name."""
    return _find_o_proj(attn).weight


def _find_mlp_modules(mlp):
    """Return (gate_mod, up_mod, down_mod) MLP modules so callers can read .bias.

    Mirrors _find_mlp_weights' resolution order. gate_mod is None for 2-proj and
    fused gate+up MLPs. For fused gate+up (Phi-3) up_mod is the fused module —
    its bias (if any) spans [gate;up]; Phi-3 has no MLP bias so this stays None
    in practice.
    """
    gate = getattr(mlp, "gate_proj", None) or getattr(mlp, "w1", None)
    up   = getattr(mlp, "up_proj", None) or getattr(mlp, "w3", None)
    down = getattr(mlp, "down_proj", None) or getattr(mlp, "w2", None)
    if gate is not None and up is not None and down is not None:
        return gate, up, down

    gate_up = getattr(mlp, "gate_up_proj", None)
    down2   = getattr(mlp, "down_proj", None)
    if gate_up is not None and down2 is not None:
        return None, gate_up, down2

    up_2 = (getattr(mlp, "dense_h_to_4h", None) or   # GPTNeoX, BLOOM
            getattr(mlp, "c_fc", None) or              # Starcoder2
            getattr(mlp, "ff_proj", None) or           # OLMo
            getattr(mlp, "fc_in", None))
    dn_2 = (getattr(mlp, "dense_4h_to_h", None) or   # GPTNeoX, BLOOM
            getattr(mlp, "c_proj", None) or            # Starcoder2
            getattr(mlp, "ff_out", None) or            # OLMo
            getattr(mlp, "fc_out", None))
    return None, up_2, dn_2


def _find_mlp_weights(mlp, cfg):
    """Return (gate_w, up_w, down_w) raw weight tensors.

    gate_w is None for 2-projection MLPs (no gating).
    Handles SwiGLU (gate+up+down), fused gate_up (Phi-3), and
    2-proj variants (GPTNeoX dense_h_to_4h, Starcoder2 c_fc, OLMo ff_proj).
    """
    # SwiGLU with standard names
    gate = (getattr(mlp, "gate_proj", None) or
            getattr(mlp, "w1", None))     # InternLM2
    up   = (getattr(mlp, "up_proj", None) or
            getattr(mlp, "w3", None))     # InternLM2
    down = (getattr(mlp, "down_proj", None) or
            getattr(mlp, "w2", None))     # InternLM2

    if gate is not None and up is not None and down is not None:
        return gate.weight, up.weight, down.weight

    # Fused gate+up (Phi-3: gate_up_proj contains [gate; up] concatenated)
    gate_up = getattr(mlp, "gate_up_proj", None)
    down2   = getattr(mlp, "down_proj", None)
    if gate_up is not None and down2 is not None:
        w   = gate_up.weight.detach().bfloat16()  # (gate_out+up_out, in)
        mid = w.shape[0] // 2
        return w[:mid], w[mid:], down2.weight

    # 2-proj MLP (no gate) — return gate=None
    up_2 = (getattr(mlp, "dense_h_to_4h", None) or   # GPTNeoX, BLOOM
            getattr(mlp, "c_fc", None) or              # Starcoder2
            getattr(mlp, "ff_proj", None) or           # OLMo
            getattr(mlp, "fc_in", None))
    dn_2 = (getattr(mlp, "dense_4h_to_h", None) or   # GPTNeoX, BLOOM
            getattr(mlp, "c_proj", None) or            # Starcoder2
            getattr(mlp, "ff_out", None) or            # OLMo
            getattr(mlp, "fc_out", None))
    if up_2 is not None and dn_2 is not None:
        return None, up_2.weight, dn_2.weight

    raise RuntimeError(
        f"Cannot find MLP weights in {type(mlp).__name__}. "
        f"Attributes: {[a for a in dir(mlp) if not a.startswith('_')]}"
    )


# ------------------------------------------------------------------
# Per-layer weight container
# ------------------------------------------------------------------

@dataclass
class _LayerWeights:
    norm1_w:  object   # ttnn.Tensor (TILE, hidden_p) -- input_layernorm
    norm1_sc: object   # ttnn.Tensor (TILE, TILE) -- 1/hidden scaler
    norm2_w:  object   # post_attention_layernorm
    norm2_sc: object
    qkv_w:    object   # ttnn.Tensor (hidden_p, q_out_p+k_out_p+v_out_p) -- fused
    q_end:    int      # slice boundary: qkv[:q_end] = Q
    k_end:    int      # slice boundary: qkv[q_end:k_end] = K, qkv[k_end:] = V
    o_w:      object   # ttnn.Tensor (n_heads*head_dim_p, hidden_p)
    gate_w:   object   # ttnn.Tensor (hidden_p, intermediate_p)
    up_w:     object
    down_w:   object   # ttnn.Tensor (intermediate_p, hidden_p)
    qkv_b:    object   # torch.Tensor (total_qkv_p,) CPU float -- None if no bias
    gate_b:   object   # ttnn.Tensor (TILE, intermediate_p) -- None if no bias
    up_b:     object   # ttnn.Tensor (TILE, intermediate_p) -- None if no bias
    o_b:      object = None  # ttnn.Tensor (TILE, hidden_p) o-proj bias -- None if absent
    down_b:   object = None  # ttnn.Tensor (TILE, hidden_p) MLP down bias -- None if absent
    # Optional extra norms (Gemma 3 style: norm applied to sub-layer OUTPUT)
    norm3_w:  object = None  # pre_feedforward_layernorm
    norm3_sc: object = None
    norm4_w:  object = None  # post_feedforward_layernorm
    norm4_sc: object = None
    # Residual pattern: "llama" (norm before sublayer) or "gemma" (norm wraps sublayer output)
    norm_style: str = "llama"
    # Optional per-head Q/K norms (Gemma 3, Qwen 3) -- stored as (head_dim,) CPU tensors
    q_norm_w: object = None   # torch.Tensor or None
    k_norm_w: object = None
    # LayerNorm weight+bias (CPU float tensors; None for RMSNorm models)
    norm1_w_cpu: object = None
    norm1_b: object = None
    norm2_w_cpu: object = None
    norm2_b: object = None


def _layernorm_cpu(x: torch.Tensor, weight, bias, eps: float) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var  = ((x - mean).pow(2)).mean(-1, keepdim=True)
    x_norm = (x - mean) / (var + eps).sqrt()
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


# ------------------------------------------------------------------
# Main runner class
# ------------------------------------------------------------------

class TTModelRunner:
    """Runs a HuggingFace transformer model fully on a TT device.

    Weights are uploaded to device once at init. Each token generation step
    runs all layers on device and only reads back the final hidden state for
    lm_head projection.
    """

    def __init__(
        self,
        model_path: str,
        device,
        max_seq: int = 2048,
        lm_head_on_device: bool = False,
        unsafe: bool = False,
    ):
        import ttnn
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from tt_inference_server.dispatch.registry import detect_model_family
        from tt_inference_server.dispatch.dispatcher import (
            KernelDispatcher, derive_capabilities)

        self._device = device
        self._max_seq = max_seq
        self._lm_head_on_device = lm_head_on_device
        self._unsafe = unsafe

        print(f"Loading {model_path} ...")
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        archs = getattr(hf_config, "architectures", None) or []
        arch = archs[0] if archs else "<unknown>"
        self._cfg = detect_model_family(hf_config)
        self._hf_cfg = hf_config

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("  Loading weights ...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        hf_model.eval()

        self._dispatcher = KernelDispatcher(device, unsafe=unsafe)

        # Two-tier resolution (#3): prefer the matrix entry for listed models; fall back
        # to HF-config auto-derivation for novel/unlisted ones (the universal floor). The
        # resolved entry + capabilities are wired here; the inline gates that consume them
        # are migrated one at a time in Phase C, so behavior is unchanged at this step.
        self._entry = self._dispatcher.lookup(model_path)
        self._listed = self._entry is not None
        if self._entry is not None:
            # Matrix is authoritative for dims for listed models (#3 Phase D); novel
            # models keep detect_model_family()'s HF-config introspection above.
            self._cfg = self._entry.model_config()
            self._caps = self._entry.capabilities(self._dispatcher._hw_config)
            self._community = (self._entry.status == "community")
            print(f"  Matrix: '{self._entry.name}' (status={self._entry.status}, "
                  f"backend={self._caps.attn_backend}, fast_path={self._caps.fast_path}, "
                  f"lm_head_ondevice={self._caps.lm_head_ondevice})")
            if self._community and not unsafe:
                print(f"  WARNING: '{self._entry.name}' is status=community (unverified); "
                      "correctness is not guaranteed. Pass --unsafe to silence (#25).")
        else:
            self._caps = derive_capabilities(hf_config)
            self._community = True
            print(f"  Matrix: no entry for arch '{arch}' -> auto-derived (community, "
                  f"unverified). fast_path={self._caps.fast_path}, "
                  f"lm_head_ondevice={self._caps.lm_head_ondevice}. "
                  "Output validity unverified on this hardware.")

        print("  Uploading to device ...")
        try:
            self._load_weights(hf_model)
        except (AttributeError, KeyError, RuntimeError) as e:
            raise RuntimeError(
                f"Weight loading failed for architecture '{arch}'.\n"
                f"  Model: {model_path}\n"
                f"  Error: {e}\n"
                f"  Fix: add '{arch}' support to configs/registry.py and a "
                f"matching config module, or check that the HF model structure "
                f"matches the expected layer attribute names."
            ) from None
        self._build_rope_table()
        self._build_attn_scalars()
        self._init_kv_cache()
        self._init_scratch_bufs()
        self._setup_paged_decode()
        self._setup_ondevice_lmhead()

        del hf_model  # free CPU memory
        print(f"  Ready. {len(self._layers)} layers on device.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        chat: bool = True,
    ) -> str:
        """Generate a response to the given prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = greedy).
            chat: If True and the tokenizer has a chat template, apply it.
                  Set to False for raw completion prompts.
        """
        if chat and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                input_ids = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )[0].tolist()
            except Exception:
                input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        else:
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        self.reset_cache()

        t0 = time.perf_counter()

        # Prefill: run each prompt token to populate KV cache
        next_id = None
        for pos, tok_id in enumerate(input_ids):
            next_id = self._decode_step(tok_id, pos)

        output_ids = []
        # Decode: generate new tokens
        for _ in range(max_new_tokens):
            if next_id == self._tokenizer.eos_token_id:
                break
            output_ids.append(next_id)
            next_id = self._decode_step(next_id, len(input_ids) + len(output_ids) - 1)

        elapsed = time.perf_counter() - t0
        n_new = len(output_ids)
        if n_new:
            print(f"  {n_new} tokens in {elapsed:.1f}s = {n_new/elapsed:.1f} tok/s")

        return self._tokenizer.decode(output_ids, skip_special_tokens=True)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        chat: bool = True,
    ):
        """Generator yielding decoded text deltas one token at a time.

        Same decode loop as generate(), but re-decodes the running id list each step and
        yields the newly-appended substring — correct across BPE merges/multi-byte chars.
        The final yielded item is a dict {"finish_reason", "prompt_tokens",
        "completion_tokens"} so callers can build an OpenAI-style usage block.
        """
        if chat and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                input_ids = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )[0].tolist()
            except Exception:
                input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        else:
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        self.reset_cache()

        next_id = None
        for pos, tok_id in enumerate(input_ids):
            next_id = self._decode_step(tok_id, pos)

        output_ids = []
        emitted = ""
        finish_reason = "length"
        for _ in range(max_new_tokens):
            if next_id == self._tokenizer.eos_token_id:
                finish_reason = "stop"
                break
            output_ids.append(next_id)
            text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            if len(text) > len(emitted):
                yield text[len(emitted):]
                emitted = text
            next_id = self._decode_step(next_id, len(input_ids) + len(output_ids) - 1)

        yield {
            "finish_reason": finish_reason,
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
        }

    def benchmark(self, prompt: str, n_tokens: int = 50, warmup: int = 3) -> tuple:
        """Measure steady-state decode throughput. Returns (tok_s, output_text).

        Prefill and warmup steps are untimed so kernel compilation and pipeline
        startup don't inflate elapsed. Only the n_tokens timed decode steps count.
        """
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        self.reset_cache()

        # Prefill — untimed (kernel compilation happens here on first call)
        next_id = None
        for pos, tok_id in enumerate(input_ids):
            next_id = self._decode_step(tok_id, pos)

        # Warmup decode steps — untimed, ensures device pipeline is fully active
        for i in range(warmup):
            if next_id == self._tokenizer.eos_token_id:
                break
            next_id = self._decode_step(next_id, len(input_ids) + i)

        # Timed decode
        output_ids = []
        t0 = time.perf_counter()
        for i in range(n_tokens):
            if next_id == self._tokenizer.eos_token_id:
                break
            output_ids.append(next_id)
            next_id = self._decode_step(next_id, len(input_ids) + warmup + i)
        elapsed = time.perf_counter() - t0

        n = len(output_ids)
        tok_s = n / elapsed if elapsed > 0 and n > 0 else 0.0
        print(f"  {n} tokens in {elapsed:.1f}s = {tok_s:.1f} tok/s  (warmup={warmup})")
        return tok_s, self._tokenizer.decode(output_ids, skip_special_tokens=True)

    def reset_cache(self):
        """Clear the KV cache for a new conversation."""
        for buf in self._k_hist_cpu + self._v_hist_cpu:
            buf.zero_()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _decode_step(self, token_id: int, kv_pos: int) -> int:
        """Single-token forward pass. Returns next token id (int)."""
        import ttnn
        if self._paged_attn:
            # Write the position into the pinned device tensor once; all layers' paged
            # KV-write + SDPA read it (trace-safe — device-tensor index, not a Python int).
            self._cur_pos_cpu[0] = kv_pos
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(self._cur_pos_cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                self._cur_pos_dev)
        if self._traced:
            return self._decode_step_traced(token_id)
        hidden_tt = self._embed(token_id)
        if self._embed_ln_w_cpu is not None:        # BLOOM word_embeddings_layernorm
            hidden_tt = self._apply_embed_ln(hidden_tt)
        for i in range(len(self._layers)):
            hidden_tt = self._layer_forward(hidden_tt, i, kv_pos)
        return self._lm_head(hidden_tt)

    def _apply_embed_ln(self, hidden_tt):
        """Apply the embedding LayerNorm (BLOOM) on CPU, returning (TILE, hidden_p)."""
        import ttnn
        eps = getattr(self._hf_cfg, "layer_norm_eps",
                      getattr(self._hf_cfg, "rms_norm_eps", 1e-5))
        hsz = self._cfg.hidden_size
        hp  = _tile_align(hsz)
        hcpu = ttnn.to_torch(hidden_tt)[0, :hsz].float()
        ncpu = _layernorm_cpu(hcpu, self._embed_ln_w_cpu, self._embed_ln_b_cpu, eps)
        v_row = torch.nn.functional.pad(ncpu.bfloat16(), (0, hp - hsz)).unsqueeze(0)
        v_2d  = v_row.expand(TILE, -1).contiguous()
        return ttnn.from_torch(v_2d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _embed_traced(self):
        """Host-free embed: reads the pinned token-id device tensor (no from_torch)."""
        import ttnn
        emb = ttnn.embedding(self._tok_dev, self._embed_tt)
        emb = ttnn.reshape(emb, [1, _tile_align(self._cfg.hidden_size)])
        h = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
        if self._embed_scale != 1.0:
            h = ttnn.multiply(h, self._embed_scale)
        return h

    def _run_embed_layers(self):
        hidden_tt = self._embed_traced()
        for i in range(len(self._layers)):
            hidden_tt = self._layer_forward(hidden_tt, i, 0)   # kv_pos unused on paged path
        return hidden_tt

    def _run_decode_graph(self):
        """The captured decode graph: embed + layers, plus on-device lm_head when enabled.

        Returns the argmax-index tensor when lm_head is folded in (#31), otherwise the
        final hidden state (lm_head then runs eagerly off the trace).
        """
        hidden_tt = self._run_embed_layers()
        if self._ondevice_lmhead:
            return self._lm_head_graph(hidden_tt)
        return hidden_tt

    def _decode_step_traced(self, token_id: int) -> int:
        """Trace-replayed decode (#30): host-free embed+layers run from a captured trace.

        Per-token PCIe: write token id + position into pinned buffers, execute the trace,
        read back the result. With DISPATCH_ONDEVICE_LMHEAD=1 (#31) the final norm +
        lm_head + argmax are folded into the trace, so the read-back is just the 4-byte
        sampled id; otherwise the final hidden state is read back for an eager CPU lm_head.
        """
        import ttnn
        dev = self._device
        self._tok_cpu[0, 0] = token_id
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self._tok_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self._tok_dev)
        if self._trace_id is None:
            # warmup (compiles kernels) then capture the decode graph as a replayable trace
            self._run_decode_graph()
            self._trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
            self._traced_out = self._run_decode_graph()
            ttnn.end_trace_capture(dev, self._trace_id, cq_id=0)
            print(f"  Decode trace captured (id={self._trace_id}"
                  f"{', lm_head folded in' if self._ondevice_lmhead else ''})")
        ttnn.execute_trace(dev, self._trace_id, cq_id=0, blocking=True)
        if self._ondevice_lmhead:
            return int(ttnn.to_torch(self._traced_out).flatten()[0])
        return self._lm_head(self._traced_out)

    def _embed(self, token_id: int):
        """Lookup embedding on-device, apply optional scale. Returns (TILE, hidden_p)."""
        import ttnn
        self._token_id_cpu[0, 0] = token_id
        token_tt = ttnn.from_torch(self._token_id_cpu, device=self._device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # ttnn.embedding returns ROW_MAJOR [1, 1, hidden_size]; reshape to [1, hidden_p]
        # then convert to TILE_LAYOUT which pads the first dim to TILE automatically.
        emb_tt = ttnn.embedding(token_tt, self._embed_tt)
        ttnn.deallocate(token_tt)
        hidden_p = _tile_align(self._cfg.hidden_size)
        emb_tt = ttnn.reshape(emb_tt, [1, hidden_p])
        hidden_tt = ttnn.to_layout(emb_tt, ttnn.TILE_LAYOUT)
        if self._embed_scale != 1.0:
            hidden_tt = ttnn.multiply(hidden_tt, self._embed_scale)
        return hidden_tt

    def _layer_forward(self, hidden_tt, layer_idx: int, kv_pos: int):
        import ttnn
        lw = self._layers[layer_idx]

        def rmsnorm_buf():
            return self._rmsnorm_buf_tt

        def seq_norm(h_tt, w_tt, sc_tt, w_cpu, b_cpu):
            """Pre-sublayer norm for the sequential (llama-style) residual path.
            CPU LayerNorm (mean-sub + bias) for LayerNorm models (BLOOM, StableLM,
            Falcon); on-device RMSNorm otherwise. Mirrors the gpt_neox branch."""
            if not self._uses_layernorm:
                return self._dispatcher.rmsnorm(h_tt, w_tt, sc_tt, rmsnorm_buf())
            eps = getattr(self._hf_cfg, "layer_norm_eps",
                          getattr(self._hf_cfg, "rms_norm_eps", 1e-5))
            hsz = self._cfg.hidden_size
            hp  = _tile_align(hsz)
            hcpu = ttnn.to_torch(h_tt)[0, :hsz].float()
            ncpu = _layernorm_cpu(hcpu, w_cpu, b_cpu, eps)
            v_row = torch.nn.functional.pad(ncpu.bfloat16(), (0, hp - hsz)).unsqueeze(0)
            v_2d  = v_row.expand(TILE, -1).contiguous()
            return ttnn.from_torch(v_2d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # output_tensor= for residual adds writes into pre-allocated hidden scratch,
        # eliminating 2 device tensor allocations per layer per token.
        aot = {"output_tensor": self._hidden_scratch_tt} if self._add_ot else {}

        if lw.norm_style == "gemma":
            # Gemma pattern:
            #   normed  = norm1(hidden)
            #   attn_raw = attn(normed)
            #   hidden  += norm2(attn_raw)      ← norm wraps attn OUTPUT
            #   normed2 = norm3(hidden)
            #   mlp_raw = mlp(normed2)
            #   hidden  += norm4(mlp_raw)        ← norm wraps mlp OUTPUT
            normed1_tt  = self._dispatcher.rmsnorm(hidden_tt, lw.norm1_w, lw.norm1_sc, rmsnorm_buf())
            attn_raw_tt = self._attention(normed1_tt, layer_idx, kv_pos)
            attn_normed = self._dispatcher.rmsnorm(attn_raw_tt, lw.norm2_w, lw.norm2_sc, rmsnorm_buf())
            hidden_tt   = ttnn.add(hidden_tt, attn_normed, **aot)

            normed2_tt  = self._dispatcher.rmsnorm(hidden_tt, lw.norm3_w, lw.norm3_sc, rmsnorm_buf())
            mlp_raw_tt  = self._mlp(normed2_tt, layer_idx)
            mlp_normed  = self._dispatcher.rmsnorm(mlp_raw_tt, lw.norm4_w, lw.norm4_sc, rmsnorm_buf())
            hidden_tt   = ttnn.add(hidden_tt, mlp_normed, **aot)
        elif lw.norm_style == "gpt_neox":
            # GPTNeoX parallel residual pattern:
            #   attn and MLP both branch from the same pre-norm hidden state
            #   hidden += attn(norm1(hidden)) + mlp(norm2(hidden))
            if self._uses_layernorm:
                eps = getattr(self._hf_cfg, "layer_norm_eps",
                              getattr(self._hf_cfg, "rms_norm_eps", 1e-5))
                hidden_size = self._cfg.hidden_size
                hidden_p    = _tile_align(hidden_size)
                h_torch = ttnn.to_torch(hidden_tt)
                # hidden_tt is (TILE, hidden_p); the actual data is in row 0
                hidden_cpu = h_torch[0, :hidden_size].float()
                normed1_cpu = _layernorm_cpu(hidden_cpu, lw.norm1_w_cpu, lw.norm1_b, eps)
                normed2_cpu = _layernorm_cpu(hidden_cpu, lw.norm2_w_cpu, lw.norm2_b, eps)
                # Upload as (TILE, hidden_p) — same shape as rmsnorm output buffer
                def _ln_to_tt(v_cpu):
                    v_row = torch.nn.functional.pad(
                        v_cpu.bfloat16(), (0, hidden_p - hidden_size)).unsqueeze(0)
                    v_2d = v_row.expand(TILE, -1).contiguous()
                    return ttnn.from_torch(v_2d, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=self._device,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
                normed1_tt = _ln_to_tt(normed1_cpu)
                normed2_tt = _ln_to_tt(normed2_cpu)
            else:
                normed1_tt = self._dispatcher.rmsnorm(hidden_tt, lw.norm1_w, lw.norm1_sc, rmsnorm_buf())
                normed2_tt = self._dispatcher.rmsnorm(hidden_tt, lw.norm2_w, lw.norm2_sc, rmsnorm_buf())
            attn_out_tt = self._attention(normed1_tt, layer_idx, kv_pos)
            mlp_out_tt = self._mlp(normed2_tt, layer_idx)
            hidden_tt = ttnn.add(hidden_tt, attn_out_tt, **aot)
            hidden_tt = ttnn.add(hidden_tt, mlp_out_tt, **aot)
        else:
            # Llama / Qwen / Mistral / (LayerNorm: BLOOM, StableLM, Falcon) pattern:
            #   hidden += attn(norm1(hidden))
            #   hidden += mlp(norm2(hidden))
            normed1_tt = seq_norm(hidden_tt, lw.norm1_w, lw.norm1_sc, lw.norm1_w_cpu, lw.norm1_b)
            attn_out_tt = self._attention(normed1_tt, layer_idx, kv_pos)
            hidden_tt = ttnn.add(hidden_tt, attn_out_tt, **aot)

            normed2_tt = seq_norm(hidden_tt, lw.norm2_w, lw.norm2_sc, lw.norm2_w_cpu, lw.norm2_b)
            mlp_out_tt = self._mlp(normed2_tt, layer_idx)
            hidden_tt = ttnn.add(hidden_tt, mlp_out_tt, **aot)

        return hidden_tt

    def _write_kv_to_device(self, layer_idx: int, kv_pos: int, k_cpu, v_cpu):
        """Write one token's post-RoPE K/V into the on-device KV cache.

        k_cpu / v_cpu: float32 tensors shaped (N_kv_heads, head_dim).
        Uses pre-allocated CPU staging buffers to avoid torch.zeros() each call.
        """
        import ttnn
        cfg  = self._cfg
        hd   = cfg.head_dim
        hd_p = _tile_align(hd)
        nkv  = cfg.num_kv_heads

        self._k_in_cpu.zero_()
        self._v_in_cpu.zero_()
        self._k_in_cpu[0, :, 0, :hd] = k_cpu.bfloat16()
        self._v_in_cpu[0, :, 0, :hd] = v_cpu.bfloat16()

        k_tt = ttnn.from_torch(self._k_in_cpu, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self._device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_tt = ttnn.from_torch(self._v_in_cpu, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self._device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.kv_cache.update_cache_for_token_(self._k_dev[layer_idx], k_tt, kv_pos, 0)
        ttnn.kv_cache.update_cache_for_token_(self._v_dev[layer_idx], v_tt, kv_pos, 0)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)

    def _attention(self, normed_tt, layer_idx: int, kv_pos: int):
        """Attention: QKV on device, flash-attn on Tensix, O projection on device.

        The flash_attn kernel uses grid=(N_heads // heads_per_core, 1) so it
        fits within the hardware grid limit (13 cores x-axis on p150).  For
        models with N_heads > 13 (e.g. Llama 3 8B: 32 heads) heads_per_core is
        set by resolve_attn_config so each core handles multiple Q-heads
        sequentially, sharing the KV read for heads in the same GQA group.
        """
        import ttnn
        cfg  = self._cfg
        lw   = self._layers[layer_idx]
        hd   = cfg.head_dim
        hd_p = _tile_align(hd)

        # Fused QKV projection — write into pre-allocated scratch (no mmap)
        qkv_ot  = self._qkv_scratch_tt if self._matmul_ot else None
        qkv_tt  = _matmul_safe(normed_tt, lw.qkv_w, self._device, output_tensor=qkv_ot)

        # Paged path (issue #30): trace-safe device-indexed KV write + paged SDPA decode.
        if self._paged_attn:
            return self._paged_attention(qkv_tt, layer_idx, lw)

        # On-device path (issue #8): split + RoPE + KV-write + flash_attn entirely on
        # device — no QKV readback. Gated at load time by self._ondevice_attn.
        if self._ondevice_attn:
            return self._attention_ondevice(qkv_tt, layer_idx, kv_pos, lw)

        qkv_cpu = ttnn.to_torch(qkv_tt)[0].float()   # (total_qkv_p,)
        if qkv_ot is None:
            ttnn.deallocate(qkv_tt)
        if lw.qkv_b is not None:
            qkv_cpu += lw.qkv_b

        q_cpu = qkv_cpu[:lw.q_end].view(cfg.num_heads,    hd_p)[:, :hd].contiguous()
        k_cpu = qkv_cpu[lw.q_end:lw.k_end].view(cfg.num_kv_heads, hd_p)[:, :hd].contiguous()
        v_cpu = qkv_cpu[lw.k_end:].view(cfg.num_kv_heads, hd_p)[:, :hd].contiguous()

        # Optional per-head Q/K norm (Gemma 3, Qwen 3)
        if lw.q_norm_w is not None:
            q_cpu = self._apply_head_norm(q_cpu, lw.q_norm_w)
        if lw.k_norm_w is not None:
            k_cpu = self._apply_head_norm(k_cpu, lw.k_norm_w)

        # RoPE — skipped for ALiBi models (BLOOM), which carry no rotary embedding
        if not getattr(self, "_no_rope", False):
            q_cpu = self._apply_rope(q_cpu, kv_pos)
            k_cpu = self._apply_rope(k_cpu, kv_pos)

        # Update CPU history mirrors (kept for debugging / fallback)
        self._k_hist_cpu[layer_idx][:, kv_pos, :] = k_cpu.float()
        self._v_hist_cpu[layer_idx][:, kv_pos, :] = v_cpu.float()

        # ALiBi (BLOOM): per-head positional bias can't fold into the shared-mask
        # flash_attn kernel — run CPU softmax+ALiBi attention from the history mirrors.
        if getattr(self, "_uses_alibi", False):
            return self._attention_cpu_alibi(q_cpu, layer_idx, kv_pos, lw)

        # Write post-RoPE K/V into the on-device KV cache
        self._write_kv_to_device(layer_idx, kv_pos, k_cpu, v_cpu)

        # Upload Q to device: [N_heads*TILE, head_dim_p]
        # Kernel reads tile-row h = element rows [h*TILE:(h+1)*TILE], so each head needs
        # its full TILE block filled. repeat_interleave(TILE) replicates each head row.
        self._q_cpu.zero_()
        self._q_cpu[:, :hd] = q_cpu.bfloat16().repeat_interleave(TILE, dim=0)
        q_tt = _to_tt(self._q_cpu, self._device)

        # Reshape device KV cache [1, N_kv, max_seq_p, hd_p] →
        # [N_kv * max_seq_p, hd_p] as expected by flash_attn_kernel
        nkv      = cfg.num_kv_heads
        ms_p     = _tile_align(self._max_seq)
        K_2d = ttnn.reshape(self._k_dev[layer_idx], [nkv * ms_p, hd_p])
        V_2d = ttnn.reshape(self._v_dev[layer_idx], [nkv * ms_p, hd_p])

        # Build causal mask: [TILE, max_seq_p] — 0 for valid positions, -inf for future
        # Reuse pre-allocated CPU buffer; only the boundary tile changes each step
        self._mask_cpu.fill_(float("-inf"))
        self._mask_cpu[0, :kv_pos + 1] = 0.0
        mask_tt = _to_tt(self._mask_cpu, self._device)

        # Run flash attention on Tensix cores
        self._dispatcher.flash_attn(
            q_tt, K_2d, V_2d,
            self._fa_scale_tt, self._fa_ninf_tt, self._fa_zero_tt,
            self._fa_zero_head_tt, self._fa_ones_tt,
            mask_tt, self._fa_out_tt,
            N_heads=cfg.num_heads, N_kv_heads=cfg.num_kv_heads,
        )

        # O projection — _fa_out_tt is [TILE, N_heads*hd_p], already O-proj compatible
        return self._o_proj(self._fa_out_tt, lw)

    def _dev_rope(self, x4, c4, s4):
        """Elementwise RoPE on device. x4: [1,1,H,head_dim]; c4/s4: [1,1,1,head_dim].

        Validated 1:1 vs CPU _apply_rope for full rope. Only valid when
        rotary_ndims == head_dim (gated by self._ondevice_attn).
        """
        import ttnn
        hd   = self._cfg.head_dim
        half = hd // 2
        H    = x4.shape[2]
        a  = ttnn.slice(x4, [0, 0, 0, 0],    [1, 1, H, half])
        b  = ttnn.slice(x4, [0, 0, 0, half], [1, 1, H, hd])
        rh = ttnn.concat([ttnn.neg(b), a], dim=-1)
        return ttnn.add(ttnn.multiply(x4, c4), ttnn.multiply(rh, s4))

    def _attention_ondevice(self, qkv_tt, layer_idx: int, kv_pos: int, lw):
        """On-device attention: split QKV + RoPE + KV-cache write + flash_attn, no readback.

        Feeds the same proven flash_attn + O-proj path as the CPU route; only the
        split/RoPE/KV-staging move on-device. Eliminates the per-token QKV->CPU
        readback (issue #8). All reshapes validated on-card in
        tests/diagnostic/probe_split_rope_layout.py.
        """
        import ttnn
        cfg  = self._cfg
        hd   = cfg.head_dim
        hd_p = _tile_align(hd)                 # == hd here (gated)
        nh   = cfg.num_heads
        nkv  = cfg.num_kv_heads
        qp   = nh * hd_p
        kp   = nkv * hd_p
        qend, kend, qkvp = qp, qp + kp, qp + 2 * kp
        ms_p = _tile_align(self._max_seq)

        # cos/sin for this position, gathered on-device from the resident tables
        self._pos_idx_cpu[0, 0] = kv_pos
        pidx = ttnn.from_torch(self._pos_idx_cpu, dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        c4 = ttnn.unsqueeze_to_4D(ttnn.embedding(pidx, self._rope_cos_dev, layout=ttnn.TILE_LAYOUT))
        s4 = ttnn.unsqueeze_to_4D(ttnn.embedding(pidx, self._rope_sin_dev, layout=ttnn.TILE_LAYOUT))

        # split row 0 of qkv into per-head Q/K/V, then RoPE Q and K
        row = ttnn.slice(qkv_tt, [0, 0], [1, qkvp])
        q4  = ttnn.reshape(ttnn.slice(row, [0, 0],    [1, qend]), [1, 1, nh,  hd_p])
        k4  = ttnn.reshape(ttnn.slice(row, [0, qend], [1, kend]), [1, 1, nkv, hd_p])
        v4  = ttnn.reshape(ttnn.slice(row, [0, kend], [1, qkvp]), [1, 1, nkv, hd_p])
        q_rot = self._dev_rope(q4, c4, s4)
        k_rot = self._dev_rope(k4, c4, s4)

        # flash_attn Q layout: [N_heads*TILE, hd_p] (each head row replicated across a tile)
        q_tt = ttnn.repeat_interleave(ttnn.reshape(q_rot, [nh, hd_p]), TILE, dim=0)
        q_tt = ttnn.to_layout(q_tt, ttnn.TILE_LAYOUT)

        # write post-RoPE K and V into the on-device KV cache at kv_pos
        k_in = ttnn.pad(ttnn.transpose(k_rot, 1, 2), [(0, 0), (0, 0), (0, TILE - 1), (0, 0)], 0.0)
        v_in = ttnn.pad(ttnn.transpose(v4,    1, 2), [(0, 0), (0, 0), (0, TILE - 1), (0, 0)], 0.0)
        ttnn.kv_cache.update_cache_for_token_(self._k_dev[layer_idx], k_in, kv_pos, 0)
        ttnn.kv_cache.update_cache_for_token_(self._v_dev[layer_idx], v_in, kv_pos, 0)

        # reshape device KV cache for flash_attn, build causal mask, run kernel
        K_2d = ttnn.reshape(self._k_dev[layer_idx], [nkv * ms_p, hd_p])
        V_2d = ttnn.reshape(self._v_dev[layer_idx], [nkv * ms_p, hd_p])
        self._mask_cpu.fill_(float("-inf"))
        self._mask_cpu[0, :kv_pos + 1] = 0.0
        mask_tt = _to_tt(self._mask_cpu, self._device)
        self._dispatcher.flash_attn(
            q_tt, K_2d, V_2d,
            self._fa_scale_tt, self._fa_ninf_tt, self._fa_zero_tt,
            self._fa_zero_head_tt, self._fa_ones_tt,
            mask_tt, self._fa_out_tt,
            N_heads=nh, N_kv_heads=nkv,
        )
        return self._o_proj(self._fa_out_tt, lw)

    def _setup_paged_decode(self):
        """Set up the paged trace-safe decode path (issue #30), gated by env.

        DISPATCH_PAGED_ATTN=1 enables eager paged attention; DISPATCH_TRACE=1 also
        captures the decode step as a replayable trace. Only the clean GQA case is
        supported (nh==32 so the decode SDPA's pad-to-32 is a no-op and grouping stays
        nh/nkv); other shapes fall back to the existing path.
        """
        import os, ttnn
        self._paged_attn = False
        self._traced = False
        want = os.environ.get("DISPATCH_PAGED_ATTN", "0") == "1" or os.environ.get("DISPATCH_TRACE", "0") == "1"
        if not want:
            return
        cfg = self._cfg
        hd, nh, nkv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
        hd_p = _tile_align(hd)
        group = (nh // nkv) if nkv else 0
        # Proportional kv-head padding (#35): the decode SDPA pads Q to 32 heads and groups
        # GQA as padded_nh/nkv. Pad q->32 AND kv->32/group so the padded grouping equals the
        # real group; real q-heads then map to real kv-heads (probe_nh_lt32_padkv.py, 1.0
        # cos-sim). For nh==32 this is a no-op (nkv_pad==nkv) — preserves the validated path.
        # Requires group | 32 (else padded grouping can't match the real group). NOTE: padding
        # nh<32 up to 32 wastes attention compute (32/nh x query heads) — benchmark the tax;
        # group-does-not-divide-32 models (starcoder g=12, qwen2.5 g=7) take the custom route.
        ok_shape = (nkv > 0 and nh % nkv == 0 and group > 0 and 32 % group == 0 and nh <= 32)
        if not (self._ondevice_attn_eligible and hd_p == hd and ok_shape):
            print(f"  Paged decode: NOT eligible (need eligible on-device attn + tile head_dim "
                  f"+ nh<=32 with group|32; nh={nh}, nkv={nkv}, group={group}, hd_p==hd={hd_p == hd})")
            return
        self._nh, self._nkv = nh, nkv          # real head counts
        self._nh_pad = 32                       # decode SDPA pads Q to 32
        self._nkv_pad = 32 // group             # == nkv when nh==32 (no-op)
        self._block = 32
        self._nblocks = math.ceil(_tile_align(self._max_seq) / self._block)
        n = len(self._layers)

        def zc():
            return ttnn.zeros([self._nblocks, self._nkv_pad, self._block, hd], dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._kp = [zc() for _ in range(n)]
        self._vp = [zc() for _ in range(n)]
        pt = torch.arange(self._nblocks, dtype=torch.int32).reshape(1, self._nblocks)
        self._page_table = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                           device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        grid = ttnn.num_cores_to_corerangeset(1, self._device.compute_with_storage_grid_size(), True)
        self._kv_shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, hd], ttnn.ShardOrientation.ROW_MAJOR))
        self._sdpa_pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 1), q_chunk_size=32, k_chunk_size=self._block)
        self._cur_pos_cpu = torch.zeros(1, dtype=torch.int32)
        self._cur_pos_dev = ttnn.from_torch(self._cur_pos_cpu, dtype=ttnn.int32,
                                            layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert hasattr(self, "_rope_cos_dev"), "rope device tables missing (should be uploaded in _build_rope_table)"
        # Pinned token-id buffer for host-free embed inside the trace (#30)
        self._tok_cpu = torch.zeros(1, 1, dtype=torch.int32)
        self._tok_dev = ttnn.from_torch(self._tok_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                        device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._traced = os.environ.get("DISPATCH_TRACE", "0") == "1"
        self._trace_id = None
        self._traced_hidden = None
        self._paged_attn = True
        pad_note = "" if self._nh_pad == nh else f", q {nh}->32 / kv {nkv}->{self._nkv_pad} (group {group})"
        print(f"  Paged decode: ENABLED ({self._nblocks} blocks × {self._block}, {n} layers, "
              f"nkv={nkv}{pad_note})"
              f"{' + TRACE' if self._traced else ''}")

    def _paged_attention(self, qkv_tt, layer_idx: int, lw):
        """Trace-safe paged attention (issue #30): device-indexed KV write + paged SDPA.

        Position is read from the pinned self._cur_pos_dev (written once per token in
        _decode_step), so this contains no Python-int args and no host round-trips —
        it is fully trace-capturable. Validated core: probe_traced_decode.py.
        """
        import ttnn
        cfg = self._cfg
        hd = cfg.head_dim
        hd_p = _tile_align(hd)
        nh, nkv = cfg.num_heads, cfg.num_kv_heads
        qp, kp = nh * hd_p, nkv * hd_p
        qend, kend, qkvp = qp, qp + kp, qp + 2 * kp

        # cos/sin for the current position, gathered on-device from the pinned pos tensor
        pidx = ttnn.reshape(ttnn.typecast(self._cur_pos_dev, ttnn.uint32), [1, 1])
        c4 = ttnn.unsqueeze_to_4D(ttnn.embedding(pidx, self._rope_cos_dev, layout=ttnn.TILE_LAYOUT))
        s4 = ttnn.unsqueeze_to_4D(ttnn.embedding(pidx, self._rope_sin_dev, layout=ttnn.TILE_LAYOUT))

        # split row 0 -> per-head Q/K/V, RoPE Q and K
        row = ttnn.slice(qkv_tt, [0, 0], [1, qkvp])
        q4 = ttnn.reshape(ttnn.slice(row, [0, 0],    [1, qend]), [1, 1, nh,  hd_p])
        k4 = ttnn.reshape(ttnn.slice(row, [0, qend], [1, kend]), [1, 1, nkv, hd_p])
        v4 = ttnn.reshape(ttnn.slice(row, [0, kend], [1, qkvp]), [1, 1, nkv, hd_p])
        q_rot = self._dev_rope(q4, c4, s4)
        k_rot = self._dev_rope(k4, c4, s4)

        # pad kv heads to TILE, shard, write to paged cache at the device position.
        # The cache has self._nkv_pad heads; paged_update_cache writes its first nkv_pad heads
        # from the 32-padded input (real kv in 0..nkv-1, pad heads nkv..nkv_pad-1 stay zero).
        pad = [(0, 0), (0, 0), (0, TILE - nkv), (0, 0)]
        k_sh = ttnn.interleaved_to_sharded(ttnn.pad(k_rot, pad, 0.0), self._kv_shard)
        v_sh = ttnn.interleaved_to_sharded(ttnn.pad(v4,    pad, 0.0), self._kv_shard)
        ttnn.experimental.paged_update_cache(self._kp[layer_idx], k_sh,
                                             update_idxs_tensor=self._cur_pos_dev, page_table=self._page_table)
        ttnn.experimental.paged_update_cache(self._vp[layer_idx], v_sh,
                                             update_idxs_tensor=self._cur_pos_dev, page_table=self._page_table)

        # Proportional kv-pad (#35): pad Q to 32 heads so padded grouping (32/nkv_pad) == real
        # group; no-op when nh==32. SDPA returns [1,1,32,hd]; keep only the real heads.
        if self._nh_pad != nh:
            q_rot = ttnn.pad(q_rot, [(0, 0), (0, 0), (0, self._nh_pad - nh), (0, 0)], 0.0)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rot, self._kp[layer_idx], self._vp[layer_idx],
            cur_pos_tensor=self._cur_pos_dev, page_table_tensor=self._page_table,
            scale=1.0 / math.sqrt(hd), program_config=self._sdpa_pcfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)   # [1,1,nh_pad,hd]

        if self._nh_pad != nh:
            attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, nh, hd_p])   # real heads only

        # flatten heads -> [1, nh*hd_p] (head-major matches O-proj weight), broadcast to TILE rows
        attn_flat = ttnn.reshape(attn, [1, nh * hd_p])
        attn_t = ttnn.repeat(attn_flat, ttnn.Shape([TILE, 1]))
        return self._o_proj(attn_t, lw)

    def _mlp(self, normed_tt, layer_idx: int):
        """MLP using ttnn.matmul for large weight matrices.

        Handles both SwiGLU (gate+up+down) and 2-proj (up+down, gate_w=None).
        Gate and up projections run on device. Activation + elementwise multiply
        also run on device — no PCIe round-trip.

        Uses pre-allocated scratch buffers via output_tensor= when supported by ttnn
        to avoid mmap/munmap cycling on intermediate tensors.
        """
        import ttnn
        lw   = self._layers[layer_idx]
        # (#3 Phase C) listed -> matrix activation (canonical silu/gelu/gelu_tanh/relu2);
        # novel -> _cfg.activation (raw HF act string). Both are matched below.
        act  = self._entry.activation if self._listed else self._cfg.activation
        mot  = self._matmul_ot
        muot = self._mul_ot

        if lw.gate_w is None:
            # 2-proj MLP: act(up(x)+up_b) → down(·)+down_b  (GPTNeoX, BLOOM, Starcoder2, OLMo)
            up_ot = self._up_scratch_tt if mot else None
            up_tt = _matmul_safe(normed_tt, lw.up_w, self._device, output_tensor=up_ot)
            if lw.up_b is not None:                 # bias before activation
                up_biased = ttnn.add(up_tt, lw.up_b)
                if up_ot is None:
                    ttnn.deallocate(up_tt)
                up_tt, up_ot = up_biased, None      # own the fresh tensor now
            if act == "silu":
                act_tt = ttnn.silu(up_tt)
            elif act in _GELU_ACTS:
                act_tt = ttnn.gelu(up_tt, fast_and_approximate_mode=(act in _GELU_TANH_ACTS))
            else:
                act_tt = ttnn.relu(up_tt)
            if up_ot is None:
                ttnn.deallocate(up_tt)
            result = _matmul_safe(act_tt, lw.down_w, self._device)
            ttnn.deallocate(act_tt)
            if lw.down_b is not None:
                res_biased = ttnn.add(result, lw.down_b)
                ttnn.deallocate(result)
                result = res_biased
            return result

        # SwiGLU: act(gate(x)) * up(x) → down
        gate_ot = self._gate_scratch_tt if mot else None
        up_ot   = self._up_scratch_tt   if mot else None
        gate_tt = _matmul_safe(normed_tt, lw.gate_w, self._device, output_tensor=gate_ot)
        up_tt   = _matmul_safe(normed_tt, lw.up_w,   self._device, output_tensor=up_ot)

        if lw.gate_b is not None:                   # presence-driven; no-op for Llama-style
            t = ttnn.add(gate_tt, lw.gate_b)
            if gate_ot is None:
                ttnn.deallocate(gate_tt)
            gate_tt, gate_ot = t, None
        if lw.up_b is not None:
            t = ttnn.add(up_tt, lw.up_b)
            if up_ot is None:
                ttnn.deallocate(up_tt)
            up_tt, up_ot = t, None

        act_ot = self._activated_scratch_tt if muot else None
        if act == "silu":
            activated_tt = ttnn.multiply(ttnn.silu(gate_tt), up_tt,
                                         **({"output_tensor": act_ot} if act_ot else {}))
        elif act in _GELU_ACTS:
            activated_tt = ttnn.multiply(
                ttnn.gelu(gate_tt, fast_and_approximate_mode=(act in _GELU_TANH_ACTS)), up_tt,
                **({"output_tensor": act_ot} if act_ot else {}))
        else:  # relu2
            r = ttnn.relu(gate_tt)
            activated_tt = ttnn.multiply(ttnn.multiply(r, r), up_tt,
                                         **({"output_tensor": act_ot} if act_ot else {}))
            ttnn.deallocate(r)
        if gate_ot is None:
            ttnn.deallocate(gate_tt)
        if up_ot is None:
            ttnn.deallocate(up_tt)

        result = _matmul_safe(activated_tt, lw.down_w, self._device)
        if act_ot is None:
            ttnn.deallocate(activated_tt)
        if lw.down_b is not None:
            res_biased = ttnn.add(result, lw.down_b)
            ttnn.deallocate(result)
            result = res_biased
        return result

    def _setup_ondevice_lmhead(self):
        """Set up the on-device final-norm + lm_head + argmax path (issue #31), gated by env.

        DISPATCH_ONDEVICE_LMHEAD=1 keeps the whole final stage on the card: rms_norm →
        ttnn.matmul(normed, lm_head_w) → ttnn.argmax. Eliminates the per-token logits
        readback (the last big device→host transfer besides the 4-byte sampled id) and,
        when combined with DISPATCH_TRACE=1, folds into the captured decode trace so the
        full step is host-free.

        Eligibility (the nh==32 GQA/MHA validated targets — llama3, DeepSeek-Llama, Phi-3.5):
          - RMSNorm final norm (LayerNorm final-norm bias not yet uploaded to device).
        Non-tile-aligned vocab (e.g. granite 49155) IS supported: the uploaded weight's
        padding columns are zero (logit 0), which could win argmax if all real logits are
        negative, so a constant [vocab:vocab_p] = -inf mask is added before argmax. Validated
        on card (probe_ondevice_lmhead.py): bf16 matmul + fp32-acc gives logit cos-sim ~1.0
        vs CPU fp32; greedy picks differ only on genuine near-ties.
        """
        import os, ttnn
        self._ondevice_lmhead = False
        self._lmhead_pad_mask = None
        if os.environ.get("DISPATCH_ONDEVICE_LMHEAD", "0") != "1":
            return
        vocab   = int(self._lm_head_w_cpu.shape[0])          # [vocab, hidden]
        vocab_p = int(self._lm_head_w_tt.shape[-1])          # uploaded [hidden_p, vocab_p]
        # (#3 Phase C) listed -> derived lm_head_ondevice (norm_type != layernorm);
        # novel -> the introspection floor (not _uses_layernorm). Equivalent by construction.
        norm_ok = self._caps.lm_head_ondevice if self._listed else (not self._uses_layernorm)
        if not norm_ok:
            print("  On-device lm_head: NOT eligible (LayerNorm final norm not yet "
                  "supported on-device) -> CPU lm_head")
            return
        # HiFi4 + fp32 dest accumulation: maximize precision of the wide bf16 dot
        # products over a large vocab so greedy argmax matches the CPU fp32 reference.
        self._lmhead_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        # Unaligned vocab: mask the zero-weight padding columns so argmax can't pick them.
        if vocab_p != vocab:
            m = torch.zeros(TILE, vocab_p, dtype=torch.bfloat16)
            m[:, vocab:] = float("-inf")
            self._lmhead_pad_mask = ttnn.from_torch(
                m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._ondevice_lmhead = True
        print(f"  On-device lm_head: ENABLED (DISPATCH_ONDEVICE_LMHEAD=1; "
              f"norm+matmul+argmax on device, vocab={vocab}"
              + (f", {vocab_p - vocab} padding cols masked" if vocab_p != vocab else "") + ")"
              + (" + folded into decode trace" if getattr(self, "_traced", False) else ""))

    def _lm_head_graph(self, hidden_tt):
        """On-device final norm + lm_head matmul + argmax. Returns the argmax index tensor
        (device-resident). Trace-safe: rms_norm writes the reused _rmsnorm_buf_tt, matmul
        and argmax allocate fixed-address buffers captured by the trace (no host roundtrip).
        """
        import ttnn
        normed_tt = self._dispatcher.rmsnorm(hidden_tt, self._final_norm_w_tt,
                                             self._final_norm_sc_tt, self._rmsnorm_buf_tt)
        logits_tt = ttnn.matmul(normed_tt, self._lm_head_w_tt,
                                compute_kernel_config=self._lmhead_ckc)
        if self._lmhead_pad_mask is not None:
            logits_tt = ttnn.add(logits_tt, self._lmhead_pad_mask)
        return ttnn.argmax(logits_tt, dim=-1)

    def _lm_head_ondevice(self, hidden_tt) -> int:
        """Eager (non-traced) on-device lm_head — reads back only the 4-byte argmax id.

        DISPATCH_LMHEAD_DEBUG=1 additionally computes the CPU fp32 reference argmax from
        the same hidden state and logs any divergence with the fp32 logit gap between the
        two picks — a tiny gap means a benign bf16 near-tie, a large gap means a real
        ranking bug. Diagnostic only (the readback defeats the host-free goal).
        """
        import os, ttnn
        out_tt = self._lm_head_graph(hidden_tt)
        dev_idx = int(ttnn.to_torch(out_tt).flatten()[0])
        if os.environ.get("DISPATCH_LMHEAD_DEBUG", "0") == "1":
            hidden = self._cfg.hidden_size
            h_cpu = ttnn.to_torch(hidden_tt)[0, :hidden].float()
            rms = h_cpu.pow(2).mean().add(
                getattr(self._hf_cfg, "rms_norm_eps", 1e-5)).sqrt()
            normed = (h_cpu / rms) * self._final_norm_w_cpu
            ref_logits = normed @ self._lm_head_w_cpu.T
            ref_idx = int(ref_logits.argmax())
            if dev_idx != ref_idx:
                gap = float(ref_logits[ref_idx] - ref_logits[dev_idx])
                rng = float(ref_logits.max() - ref_logits.min())
                print(f"  [lmhead-debug] divergence: dev={dev_idx} cpu={ref_idx} "
                      f"fp32_gap={gap:.5f} (logit_range={rng:.2f}, rel={gap/rng:.2e})",
                      flush=True)
        return dev_idx

    def _lm_head(self, hidden_tt) -> int:
        """Final norm (device) + lm_head matmul + argmax (CPU). Returns next token id.

        lm_head runs on CPU in float32 for accuracy — bfloat16 argmax on device
        can mis-rank logits for large vocabularies due to precision loss. The
        on-device path (issue #31, DISPATCH_ONDEVICE_LMHEAD=1) uses fp32-accumulated
        bf16 matmul + ttnn.argmax instead; see _setup_ondevice_lmhead.
        """
        import ttnn
        if self._ondevice_lmhead:
            return self._lm_head_ondevice(hidden_tt)
        hidden = self._cfg.hidden_size
        if self._uses_layernorm:
            eps = getattr(self._hf_cfg, "layer_norm_eps",
                          getattr(self._hf_cfg, "rms_norm_eps", 1e-5))
            hidden_cpu = ttnn.to_torch(hidden_tt)[0, :hidden].float()
            normed_cpu = _layernorm_cpu(hidden_cpu, self._final_norm_w_cpu,
                                        self._final_norm_b_cpu, eps)
        else:
            normed_tt  = self._dispatcher.rmsnorm(hidden_tt, self._final_norm_w_tt,
                                                   self._final_norm_sc_tt,
                                                   self._rmsnorm_buf_tt)
            normed_cpu = ttnn.to_torch(normed_tt)[0, :hidden].float()   # (hidden,)
        logits = normed_cpu @ self._lm_head_w_cpu.T                 # (vocab,)
        return int(logits.argmax())

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self, hf_model):
        backbone, layers_attr = _find_text_backbone(hf_model)
        cfg = self._cfg
        layers_list = getattr(backbone, layers_attr)

        # Embedding — upload to device DRAM once; free CPU copy to save ~1-2 GB RAM.
        # For tied embeddings (Llama), lm_head.weight is the same CPU tensor but
        # the two device tensors need different layouts (ROW_MAJOR vs TILE+transposed),
        # so they are uploaded separately.
        import ttnn as _ttnn
        embed_module = _find_embed_tokens(backbone)
        embed_cpu = embed_module.weight.detach().bfloat16()       # [vocab, hidden]
        hidden_p  = _tile_align(cfg.hidden_size)
        if hidden_p != cfg.hidden_size:
            embed_cpu = torch.nn.functional.pad(embed_cpu, (0, hidden_p - cfg.hidden_size))
        self._embed_tt = _ttnn.from_torch(
            embed_cpu,
            dtype=_ttnn.bfloat16,
            layout=_ttnn.ROW_MAJOR_LAYOUT,
            device=self._device,
            memory_config=_ttnn.DRAM_MEMORY_CONFIG,
        )
        del embed_cpu
        # Keep a null reference so any stale access fails fast instead of silently using CPU
        self._embed_w_cpu = None

        # Embedding LayerNorm (BLOOM: word_embeddings_layernorm, applied to the embedding
        # before layer 0). CPU LayerNorm — None for every other arch (no-op).
        embed_ln = (getattr(backbone, "word_embeddings_layernorm", None)
                    or getattr(backbone, "embeddings_layernorm", None))
        self._embed_ln_w_cpu = embed_ln.weight.detach().float() if embed_ln is not None else None
        self._embed_ln_b_cpu = (embed_ln.bias.detach().float()
                                if embed_ln is not None and getattr(embed_ln, "bias", None) is not None
                                else None)

        # Behavioral flags (#3 Phase C). For a LISTED model these come from the matrix
        # entry (trusted, hardware-tuned); for a novel/unlisted model they fall back to the
        # HF/module introspection that is the universal floor. The matrix was populated to
        # match introspection, so listed models are byte-identical to the pre-#3 behavior.
        first_layer = layers_list[0]
        if self._listed:
            self._gemma_norm        = (self._entry.norm_type == "gemma_rms")
            self._uses_layernorm    = (self._entry.norm_type == "layernorm")
            self._parallel_residual = self._entry.parallel_residual
            self._embed_scale       = (math.sqrt(cfg.hidden_size)
                                       if self._entry.embed_scale == "sqrt_hidden" else 1.0)
        else:
            # Gemma-style (1+w) 4-norm pattern
            self._gemma_norm = (hasattr(first_layer, "pre_feedforward_layernorm") and
                                hasattr(first_layer, "post_feedforward_layernorm"))
            # GPTNeoX parallel residual: attn and MLP branch the same pre-norm hidden
            self._parallel_residual = getattr(self._hf_cfg, "use_parallel_residual", False)
            # LayerNorm (mean-sub + bias) vs RMSNorm: GPTNeoX/Pythia carry a norm bias
            first_norm = _find_layer_norms(first_layer)[0]
            self._uses_layernorm = (hasattr(first_norm, "bias") and first_norm.bias is not None)
            # Gemma family scales input embeddings by sqrt(hidden_size)
            text_cfg   = getattr(self._hf_cfg, "text_config", self._hf_cfg)
            model_type = getattr(text_cfg, "model_type", "")
            self._embed_scale = math.sqrt(cfg.hidden_size) if "gemma" in model_type.lower() else 1.0

        # Final norm
        final_norm = _find_final_norm(backbone)
        _fnw = getattr(final_norm, "weight", None)
        final_w = _fnw.detach().float() if _fnw is not None else None
        self._final_norm_w_cpu = ((1.0 + final_w) if self._gemma_norm else final_w) if final_w is not None else None
        self._final_norm_w_tt  = self._upload_norm_w(final_w, gemma_style=self._gemma_norm)
        self._final_norm_sc_tt = self._make_scaler_tt(cfg.hidden_size)
        _fnb = getattr(final_norm, "bias", None)
        self._final_norm_b_cpu = _fnb.detach().float() if _fnb is not None else None

        # lm_head (GPTNeoX names it embed_out)
        lm_head_mod = getattr(hf_model, "lm_head", None) or getattr(hf_model, "embed_out", None)
        if lm_head_mod is None:
            raise RuntimeError(f"Cannot find lm_head / embed_out on {type(hf_model).__name__}")
        self._lm_head_w_cpu = lm_head_mod.weight.detach().float()
        self._lm_head_w_tt  = self._upload_linear_w(self._lm_head_w_cpu)

        # Per-layer weights
        self._layers: List[_LayerWeights] = []
        n = len(layers_list)
        for i, layer in enumerate(layers_list):
            print(f"    layer {i+1}/{n}", end="\r")
            attn = _find_layer_attn(layer)
            mlp  = _find_layer_mlp(layer)
            n1, n2, n3, n4 = _find_layer_norms(layer)

            has_pre_ff  = n3 is not None
            has_post_ff = n4 is not None
            if has_pre_ff and has_post_ff:
                norm_style = "gemma"
            elif self._parallel_residual:
                norm_style = "gpt_neox"
            else:
                norm_style = "llama"
            gemma_norm  = (norm_style == "gemma")

            # QKV — handle fused and separate projections.
            # Per-head padding is required when head_dim is not a multiple of 32
            # (e.g. BLOOM head_dim=80) so the view in _attention stays valid.
            q_cpu, k_cpu, v_cpu = _split_qkv_weights(attn, cfg)
            q_cpu = _pad_qkv_per_head(q_cpu, cfg.num_heads,    cfg.head_dim)
            k_cpu = _pad_qkv_per_head(k_cpu, cfg.num_kv_heads, cfg.head_dim)
            v_cpu = _pad_qkv_per_head(v_cpu, cfg.num_kv_heads, cfg.head_dim)
            q_out_p = q_cpu.shape[1]   # num_heads    * hd_p (already aligned)
            k_out_p = k_cpu.shape[1]   # num_kv_heads * hd_p
            qkv_cat = torch.cat([
                _pad_to_tiles(q_cpu),
                _pad_to_tiles(k_cpu),
                _pad_to_tiles(v_cpu),
            ], dim=1)

            # MLP — handle SwiGLU, 2-proj, fused gate+up
            gate_raw, up_raw, down_raw = _find_mlp_weights(mlp, cfg)
            gate_w_tt = None if gate_raw is None else self._upload_linear_w(gate_raw)

            # Biases are presence-driven: loaded from the actual resolved modules
            # (dense_h_to_4h / dense_4h_to_h for GPTNeoX/BLOOM, not the hardcoded
            # up_proj name) and applied only when present. None for bias-free MLPs.
            gate_mod, up_mod, down_mod = _find_mlp_modules(mlp)
            gate_b_tt = self._upload_bias(
                getattr(gate_mod, "bias", None) if gate_mod is not None else None,
                cfg.intermediate_size)
            up_b_tt = self._upload_bias(
                getattr(up_mod, "bias", None) if up_mod is not None else None,
                cfg.intermediate_size)
            down_b_tt = self._upload_bias(
                getattr(down_mod, "bias", None) if down_mod is not None else None,
                cfg.hidden_size)

            # O-projection bias (GPTNeoX/BLOOM attention.dense.bias; None for Llama-style)
            o_mod  = _find_o_proj(attn)
            o_b_tt = self._upload_bias(getattr(o_mod, "bias", None), cfg.hidden_size)

            qkv_b_cpu = _get_qkv_bias(attn, cfg)

            lw = _LayerWeights(
                norm1_w  = self._upload_norm_w(getattr(n1, "weight", None), gemma_norm),
                norm1_sc = self._make_scaler_tt(cfg.hidden_size),
                norm2_w  = self._upload_norm_w(getattr(n2, "weight", None), gemma_norm),
                norm2_sc = self._make_scaler_tt(cfg.hidden_size),
                qkv_w = _to_tt(qkv_cat, self._device),
                qkv_b = qkv_b_cpu,
                q_end = q_out_p,
                k_end = q_out_p + k_out_p,
                o_w   = self._upload_linear_w(
                            _pad_o_proj_per_head(
                                _find_o_proj_weight(attn).detach().bfloat16(),
                                cfg.num_heads, cfg.head_dim)),
                gate_w = gate_w_tt,
                up_w   = self._upload_linear_w(up_raw),
                down_w = self._upload_linear_w(down_raw),
                gate_b = gate_b_tt,
                up_b   = up_b_tt,
                down_b = down_b_tt,
                o_b    = o_b_tt,
                norm3_w  = self._upload_norm_w(getattr(n3, "weight", None), gemma_norm) if has_pre_ff else None,
                norm3_sc = self._make_scaler_tt(cfg.hidden_size) if has_pre_ff else None,
                norm4_w  = self._upload_norm_w(getattr(n4, "weight", None), gemma_norm) if has_post_ff else None,
                norm4_sc = self._make_scaler_tt(cfg.hidden_size) if has_post_ff else None,
                norm_style = norm_style,
                q_norm_w = self._load_head_norm_w(attn, "q_norm", gemma_norm),
                k_norm_w = self._load_head_norm_w(attn, "k_norm", gemma_norm),
                norm1_w_cpu = n1.weight.detach().float() if self._uses_layernorm else None,
                norm1_b     = n1.bias.detach().float()   if (self._uses_layernorm and getattr(n1, "bias", None) is not None) else None,
                norm2_w_cpu = n2.weight.detach().float() if self._uses_layernorm else None,
                norm2_b     = n2.bias.detach().float()   if (self._uses_layernorm and getattr(n2, "bias", None) is not None) else None,
            )
            self._layers.append(lw)
        print()

    def _load_head_norm_w(self, attn, attr: str, gemma_style: bool):
        """Load per-head Q/K norm weight as a CPU float tensor, or None."""
        norm = getattr(attn, attr, None)
        if norm is None or not hasattr(norm, "weight"):
            return None
        w = norm.weight.detach().float()
        return (1.0 + w) if gemma_style else w

    def _upload_norm_w(self, w, gemma_style: bool = False):
        """Upload norm weight (hidden,) as (TILE, hidden_p) on device.

        w may be None for norms without learned parameters (e.g. OLMo
        uses elementwise_affine=False). In that case we upload ones so the
        kernel multiplies by 1 (no effect on the normalized output).

        Gemma-style norms use (1 + weight) instead of weight — bake the +1
        at load time so the kernel path stays the same.
        """
        if w is None:
            w_f = torch.ones(self._cfg.hidden_size, dtype=torch.float32)
        else:
            w_f = w.detach().float()
        if gemma_style:
            w_f = 1.0 + w_f
        w_1d = _pad_to_tiles(w_f.bfloat16())
        w_2d = w_1d.unsqueeze(0).expand(TILE, -1).contiguous()
        return _to_tt(w_2d, self._device)

    def _upload_linear_w(self, w: torch.Tensor):
        """Upload weight (out, in) transposed to (in_p, out_p) on device."""
        return _to_tt(w.detach().bfloat16().T.contiguous(), self._device)

    def _upload_bias(self, b: Optional[torch.Tensor], out_dim: int):
        """Upload bias as (TILE, out_p), or None when the module has no bias.

        Returning None (not a zero tensor) lets the forward pass skip the add
        entirely for bias-free models (Llama/Qwen/StableLM/OLMo/Phi-3) — keeping
        the generic bias support a true no-op for them.
        """
        if b is None:
            return None
        b_1d = _pad_to_tiles(b.detach().bfloat16())
        b_2d = b_1d.unsqueeze(0).expand(TILE, -1).contiguous()
        return _to_tt(b_2d, self._device)

    def _o_proj(self, inp, lw):
        """O-projection matmul + optional bias (GPTNeoX/BLOOM attention.dense.bias).

        Bias is presence-driven (lw.o_b is None for Llama-style attn) so this is a
        no-op for bias-free models on every attention path (CPU-readback/ondevice/paged).
        """
        import ttnn
        out = _matmul_safe(inp, lw.o_w, self._device)
        if lw.o_b is not None:
            biased = ttnn.add(out, lw.o_b)
            ttnn.deallocate(out)
            out = biased
        return out

    @staticmethod
    def _build_alibi_slopes(n_heads: int) -> torch.Tensor:
        """ALiBi per-head slopes (HF BLOOM scheme). slope_h = 2^(-8h/n) for h=1..n
        when n is a power of two; otherwise the nearest-lower-power-of-two set plus
        the interleaved extra slopes. Returns a (n_heads,) float tensor."""
        def pow2_slopes(n):
            start = 2.0 ** (-8.0 / n)
            return [start ** (i + 1) for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = pow2_slopes(n_heads)
        else:
            closest = 2 ** int(math.floor(math.log2(n_heads)))
            slopes = pow2_slopes(closest)
            extra = pow2_slopes(2 * closest)[0::2][: n_heads - closest]
            slopes = slopes + extra
        return torch.tensor(slopes, dtype=torch.float32)

    def _attention_cpu_alibi(self, q_cpu, layer_idx: int, kv_pos: int, lw):
        """CPU softmax attention with ALiBi positional bias (BLOOM).

        BLOOM has no RoPE and uses per-head ALiBi slopes that the shared-mask
        flash_attn kernel can't express; it's already off the device fast path
        (head_dim=80, LayerNorm), so attention runs on CPU from the K/V history
        mirrors. Softmax is shift-invariant per query row, so the absolute-key-pos
        bias (slope_h * key_pos) is equivalent to the relative ALiBi form.
        Projections (qkv/o-proj) stay on device.
        """
        cfg = self._cfg
        hd  = cfg.head_dim
        nh  = cfg.num_heads
        nkv = cfg.num_kv_heads
        L   = kv_pos + 1

        K = self._k_hist_cpu[layer_idx][:, :L, :].float()   # (nkv, L, hd)
        V = self._v_hist_cpu[layer_idx][:, :L, :].float()
        if nh != nkv:                                       # GQA expand (BLOOM is MHA)
            rep = nh // nkv
            K = K.repeat_interleave(rep, dim=0)
            V = V.repeat_interleave(rep, dim=0)
        q = q_cpu.float()                                   # (nh, hd)

        scale  = 1.0 / math.sqrt(hd)
        scores = torch.einsum("hd,hld->hl", q, K) * scale   # (nh, L)
        key_pos = torch.arange(L, dtype=torch.float32)
        scores = scores + self._alibi_slopes.view(nh, 1) * key_pos.view(1, L)
        attn = torch.softmax(scores, dim=-1)                # (nh, L)
        out  = torch.einsum("hl,hld->hd", attn, V)          # (nh, hd)

        # Pack into the _fa_out_tt layout [TILE, nh*hd_p]: row 0 = per-head outputs,
        # each head padded hd -> hd_p, matching the o-proj weight layout.
        hd_p  = _tile_align(hd)
        out_p = torch.zeros(nh, hd_p, dtype=torch.float32)
        out_p[:, :hd] = out
        fa = torch.zeros(TILE, nh * hd_p, dtype=torch.bfloat16)
        fa[0] = out_p.reshape(nh * hd_p).bfloat16()
        return self._o_proj(_to_tt(fa, self._device), lw)

    def _make_scaler_tt(self, dim: int):
        """Return (TILE, TILE) tile filled with 1/dim on device."""
        sc = torch.full((TILE, TILE), 1.0 / dim, dtype=torch.bfloat16)
        return _to_tt(sc, self._device)

    # ------------------------------------------------------------------
    # Attention scalars
    # ------------------------------------------------------------------

    def _build_attn_scalars(self):
        cfg = self._cfg
        scale = 1.0 / math.sqrt(cfg.head_dim)
        self._attn_scale_tt = _to_tt(
            torch.full((TILE, TILE), scale, dtype=torch.bfloat16), self._device)
        self._neg_inf_tt = _to_tt(
            torch.full((TILE, TILE), float("-inf"), dtype=torch.bfloat16), self._device)
        self._zero_tt = _to_tt(
            torch.zeros(TILE, TILE, dtype=torch.bfloat16), self._device)
        self._zero_head_tt = _to_tt(
            torch.zeros(TILE, _tile_align(cfg.head_dim), dtype=torch.bfloat16), self._device)
        self._ones_tt = _to_tt(
            torch.ones(TILE, TILE, dtype=torch.bfloat16), self._device)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    def _init_kv_cache(self):
        import ttnn
        cfg  = self._cfg
        n    = len(self._layers)
        nkv  = cfg.num_kv_heads
        hd_p = _tile_align(cfg.head_dim)
        ms_p = _tile_align(self._max_seq)

        # 4D device KV cache: [1, n_kv_heads, max_seq_p, head_dim_p]
        # Matches update_cache_for_token_ validation: input[1] == cache[1] == nkv
        self._k_dev = [
            ttnn.zeros([1, nkv, ms_p, hd_p], dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=self._device,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(n)
        ]
        self._v_dev = [
            ttnn.zeros([1, nkv, ms_p, hd_p], dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=self._device,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(n)
        ]
        print(f"  KV cache: on-device DRAM [1×{nkv}×{ms_p}×{hd_p}] × {n*2} buffers")

        # CPU-side KV history mirrors the device cache.
        # CPU attention reads from these directly, eliminating the device→CPU
        # slice+to_torch that caused 2×n_layers mmap/munmap calls per token.
        # Shape: (n_kv_heads, max_seq, head_dim) — matches the attention loop's
        # access pattern (K_hist[kv_head_idx]) so slices are contiguous.
        self._k_hist_cpu = [
            torch.zeros(nkv, self._max_seq, cfg.head_dim, dtype=torch.float32)
            for _ in range(n)
        ]
        self._v_hist_cpu = [
            torch.zeros(nkv, self._max_seq, cfg.head_dim, dtype=torch.float32)
            for _ in range(n)
        ]

    def _init_scratch_bufs(self):
        """Pre-allocate all fixed-shape device scratch tensors used in the hot path.

        Reusing these across tokens eliminates the mmap/munmap cycle that was
        measured at ~300-500 syscalls/token. All shapes are deterministic from
        the model config, so one set of buffers covers every layer and every token.

        Also probes whether ttnn.matmul / ttnn.add accept output_tensor= so the
        hot-path matmuls can write directly into pre-allocated device buffers.
        """
        import ttnn, inspect
        cfg  = self._cfg
        hid_p    = _tile_align(cfg.hidden_size)
        hd_p     = _tile_align(cfg.head_dim)
        q_out_p  = cfg.num_heads    * hd_p   # use tile-padded hd_p, not raw head_dim
        kv_out_p = cfg.num_kv_heads * hd_p
        qkv_p    = q_out_p + 2 * kv_out_p
        int_p    = _tile_align(cfg.intermediate_size)
        nkv      = cfg.num_kv_heads

        def z(*shape):
            return ttnn.zeros(list(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ms_p = _tile_align(self._max_seq)
        scale_val = 1.0 / math.sqrt(cfg.head_dim)

        # Existing rmsnorm output scratch
        self._rmsnorm_buf_tt        = z(TILE, hid_p)
        # QKV projection output
        self._qkv_scratch_tt        = z(TILE, qkv_p)
        # MLP gate, up, and activated intermediate (gate×up result)
        self._gate_scratch_tt       = z(TILE, int_p)
        self._up_scratch_tt         = z(TILE, int_p)
        self._activated_scratch_tt  = z(TILE, int_p)
        # Residual add output (hidden state accumulator across layers)
        self._hidden_scratch_tt     = z(TILE, hid_p)

        # Flash-attn constant tiles (allocated once, reused every layer/token)
        self._fa_scale_tt     = ttnn.from_torch(
            torch.full((TILE, TILE), scale_val, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._fa_ninf_tt      = ttnn.from_torch(
            torch.full((TILE, TILE), float("-inf"), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._fa_zero_tt      = z(TILE, TILE)
        self._fa_zero_head_tt = z(TILE, hd_p)
        self._fa_ones_tt      = ttnn.from_torch(
            torch.ones(TILE, TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Flash-attn output scratch: [TILE, N_heads * head_dim_p] — row-major, O-proj ready
        self._fa_out_tt       = z(TILE, cfg.num_heads * hd_p)

        # Pre-allocated CPU tensors reused each token (avoids torch.zeros() in hot loop)
        # Single-token embedding ID staging buffer — filled in _embed() each step
        self._token_id_cpu = torch.zeros(1, 1, dtype=torch.int64)
        # Position index staging for on-device RoPE cos/sin gather (issue #8)
        self._pos_idx_cpu = torch.zeros(1, 1, dtype=torch.int32)
        self._k_in_cpu  = torch.zeros(1, nkv, TILE, hd_p, dtype=torch.bfloat16)
        self._v_in_cpu  = torch.zeros(1, nkv, TILE, hd_p, dtype=torch.bfloat16)
        # Q for flash-attn: [N_heads*TILE, head_dim_p] (one TILE-block per head)
        self._q_cpu     = torch.zeros(cfg.num_heads * TILE, hd_p, dtype=torch.bfloat16)
        # Causal mask: [TILE, max_seq_p] — rebuilt cheaply each token
        self._mask_cpu  = torch.full((TILE, ms_p), float("-inf"), dtype=torch.bfloat16)

        # Probe whether ttnn ops accept output_tensor= / optional_output_tensor=.
        # inspect.signature doesn't see through nanobind wrappers, so use docstrings.
        # ttnn >= 0.71 uses optional_output_tensor for matmul; add/multiply use output_tensor.
        try:
            doc = ttnn.matmul.__doc__ or ""
            self._matmul_ot = "optional_output_tensor" in doc or "output_tensor" in doc
        except Exception:
            self._matmul_ot = False
        try:
            self._add_ot = "output_tensor" in (ttnn.add.__doc__ or "")
        except Exception:
            self._add_ot = False
        try:
            self._mul_ot = "output_tensor" in (ttnn.multiply.__doc__ or "")
        except Exception:
            self._mul_ot = False
        print(f"  Scratch bufs: matmul_ot={self._matmul_ot} add_ot={self._add_ot} mul_ot={self._mul_ot}")

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------

    def _build_rope_table(self):
        cfg     = self._hf_cfg
        dim     = self._cfg.head_dim
        theta   = getattr(cfg, "rope_theta", 10000.0)
        max_pos = self._max_seq

        # Partial RoPE (GPTNeoX/Pythia): only rotary_pct of head dims are rotated.
        # rotary_ndims = round(head_dim * rotary_pct) — must be even.
        # (#3 Phase C) listed -> matrix rotary_pct; novel -> HF-config introspection.
        if self._listed:
            rotary_pct = self._entry.rotary_pct
        else:
            rotary_pct = getattr(cfg, "partial_rotary_factor",
                                 getattr(cfg, "rotary_pct", 1.0))
        rotary_ndims  = round(dim * rotary_pct)
        if rotary_ndims % 2 != 0:
            rotary_ndims -= 1
        self._rotary_ndims = rotary_ndims   # used in _apply_rope

        # ALiBi models (BLOOM) have NO RoPE — they add a per-head linear positional
        # bias to the attention scores instead. Detect and disable rope; the per-head
        # slopes can't fold into the shared-mask flash_attn kernel, so BLOOM runs a CPU
        # softmax+ALiBi attention path (see _attention_cpu_alibi).
        text_cfg   = getattr(cfg, "text_config", cfg)
        model_type = getattr(text_cfg, "model_type", "")
        archs      = getattr(cfg, "architectures", None) or []
        self._uses_alibi = (model_type == "bloom"
                            or any("Bloom" in a for a in archs)
                            or bool(getattr(cfg, "alibi", False)))
        self._no_rope = self._uses_alibi
        if self._uses_alibi:
            self._rotary_ndims = 0
            self._alibi_slopes = self._build_alibi_slopes(self._cfg.num_heads)

        # Linear RoPE scaling: compress positions by dividing by scale factor.
        # Do NOT change theta -- scale the position indices instead.
        scale_factor = 1.0
        rope_cfg = getattr(cfg, "rope_scaling", None)
        if rope_cfg:
            if rope_cfg.get("rope_type") == "linear":
                scale_factor = rope_cfg.get("factor", 1.0)

        positions = torch.arange(max_pos, dtype=torch.float32) / scale_factor
        freqs     = 1.0 / (theta ** (torch.arange(0, rotary_ndims, 2).float() / rotary_ndims))
        angles    = torch.outer(positions, freqs)      # (max_pos, rotary_ndims/2)
        cos_half  = torch.cos(angles)                  # (max_pos, rotary_ndims/2)
        sin_half  = torch.sin(angles)

        # HuggingFace convention: expand to rotary_ndims by tiling [half, half].
        # Rotation uses grouped halves (i, i+rotary_ndims//2) not interleaved pairs.
        self._rope_cos = torch.cat([cos_half, cos_half], dim=-1)  # (max_pos, rotary_ndims)
        self._rope_sin = torch.cat([sin_half, sin_half], dim=-1)

        # On-device RoPE (issue #8): decide applicability and upload cos/sin tables.
        # Approach A — elementwise RoPE on DRAM tensors feeding the existing flash_attn,
        # eliminating the per-token QKV->CPU readback. Gated to the families it cleanly
        # covers; everything else keeps the CPU path unchanged.
        import ttnn as _ttnn
        hd   = self._cfg.head_dim
        hd_p = _tile_align(hd)
        no_bias     = all(lw.qkv_b is None for lw in self._layers)
        no_headnorm = all(lw.q_norm_w is None and lw.k_norm_w is None for lw in self._layers)
        full_rope   = (self._rotary_ndims == hd)
        introspect_eligible = bool(full_rope and hd_p == hd and no_bias and no_headnorm and not self._uses_layernorm)
        # (#3 Phase C) listed -> derived fast_path capability (which also folds in the
        # group|32 / nh<=32 SDPA-pad constraint from _setup_paged_decode); novel -> the
        # introspection floor. They agree for every listed fast-path model.
        eligible    = self._caps.fast_path if self._listed else introspect_eligible
        self._ondevice_attn_eligible = eligible
        # Opt-in (default OFF): the eager on-device path is CORRECT but slower than the
        # CPU-RoPE path because it trades one PCIe readback for ~25 host-dispatched device
        # ops/layer. The perf win lands once the decode step is trace-captured (#30), which
        # removes per-op host dispatch. Until then, keep it opt-in so baseline tok/s holds.
        import os
        self._ondevice_attn = eligible and os.environ.get("DISPATCH_ONDEVICE_ATTN", "0") == "1"
        # The paged trace-safe path (#30) also needs the on-device rope tables.
        _paged_want = os.environ.get("DISPATCH_PAGED_ATTN", "0") == "1" or os.environ.get("DISPATCH_TRACE", "0") == "1"
        if eligible and (self._ondevice_attn or _paged_want):
            self._rope_cos_dev = _ttnn.from_torch(
                self._rope_cos.bfloat16(), dtype=_ttnn.bfloat16, layout=_ttnn.ROW_MAJOR_LAYOUT,
                device=self._device, memory_config=_ttnn.DRAM_MEMORY_CONFIG)
            self._rope_sin_dev = _ttnn.from_torch(
                self._rope_sin.bfloat16(), dtype=_ttnn.bfloat16, layout=_ttnn.ROW_MAJOR_LAYOUT,
                device=self._device, memory_config=_ttnn.DRAM_MEMORY_CONFIG)
        if self._ondevice_attn:
            print("  On-device RoPE/attention: ENABLED (DISPATCH_ONDEVICE_ATTN=1)")
        elif eligible:
            print("  On-device RoPE/attention: eligible but OFF "
                  "(set DISPATCH_ONDEVICE_ATTN=1 to enable; faster once trace-captured, see #30)")
        elif self._listed and introspect_eligible:
            # Introspection alone would have marked this eligible, but the matrix entry
            # overrides it (e.g. gemma_rms norm / group∤32) -> stay on the CPU RoPE path.
            print(f"  On-device RoPE/attention: not eligible "
                  f"(matrix override: norm_type={self._entry.norm_type}, "
                  f"backend={self._caps.attn_backend}) -> CPU RoPE path")
        else:
            print(f"  On-device RoPE/attention: not eligible "
                  f"(full_rope={full_rope} hd_p==hd={hd_p == hd} no_bias={no_bias} "
                  f"no_headnorm={no_headnorm} rmsnorm={not self._uses_layernorm}) -> CPU RoPE path")

    def _apply_head_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply per-head RMSNorm. x: (n_heads, head_dim), w: (head_dim,) already (1+w) or w."""
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return (x / rms) * w

    def _apply_rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Apply RoPE to x (n_heads, head_dim) using HuggingFace convention.

        Supports partial RoPE (GPTNeoX/Pythia): only the first rotary_ndims dimensions
        are rotated; the remaining pass-through dimensions are left unchanged.
        """
        if position >= self._rope_cos.shape[0]:
            return x
        cos = self._rope_cos[position]   # (rotary_ndims,)
        sin = self._rope_sin[position]
        nd  = self._rotary_ndims
        x_rot_in = x[..., :nd]
        # Rotate: x_rot = [-x_rot_in[nd//2:], x_rot_in[:nd//2]]
        x_rot = torch.cat([-x_rot_in[..., nd // 2:], x_rot_in[..., :nd // 2]], dim=-1)
        rotated = x_rot_in * cos + x_rot * sin
        if nd == x.shape[-1]:
            return rotated
        # Concat rotated prefix with unrotated suffix
        return torch.cat([rotated, x[..., nd:]], dim=-1)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TTModelRunner inference")
    parser.add_argument("--model",      required=True, help="Local model path or HF repo")
    parser.add_argument("--prompt",     default="The capital of France is",
                        help="Prompt string")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-seq",    type=int, default=2048,
                        help="Max KV cache sequence length")
    parser.add_argument("--no-chat",    action="store_true",
                        help="Skip chat template (raw completion mode)")
    parser.add_argument("--unsafe",     action="store_true",
                        help="Allow community/unlisted (unverified) models without warning (#25)")
    args = parser.parse_args()

    import ttnn, os
    # Reserve a trace region when the trace-replay decode path (#30) is enabled.
    _tr = 134217728 if os.environ.get("DISPATCH_TRACE", "0") == "1" else 0
    device = ttnn.open_device(device_id=0, trace_region_size=_tr)
    try:
        runner = TTModelRunner(args.model, device, max_seq=args.max_seq, unsafe=args.unsafe)
        print(f"\nPROMPT: {args.prompt}")
        output = runner.generate(args.prompt, max_new_tokens=args.max_tokens,
                                 chat=not args.no_chat)
        print(f"OUTPUT: {output}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
