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
        # L1 overflow — fall back to CPU matmul (output_tensor not used on CPU path)
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
    - query_key_value (GPTNeoX, BLOOM, Falcon): (3*h, in) stacked evenly
    - qkv_proj (Phi-3):                         (q+k+v out, in) by head counts
    - wqkv (InternLM2):                          same layout as qkv_proj
    - c_attn (GPT-2 style):                      same as query_key_value
    """
    # Fused QKV — split by head counts then transpose
    fused = (getattr(attn, "query_key_value", None) or
             getattr(attn, "qkv_proj", None) or
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


def _find_o_proj_weight(attn):
    """Find output projection weight regardless of attribute name."""
    for attr in ("o_proj", "out_proj", "dense", "wo", "c_proj"):
        m = getattr(attn, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m.weight
    raise RuntimeError(f"Cannot find output projection in {type(attn).__name__}")


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
    gate_b:   object   # ttnn.Tensor (TILE, intermediate_p) -- zero if no bias
    up_b:     object
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
    ):
        import ttnn
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from tt_inference_server.dispatch.registry import detect_model_family
        from tt_inference_server.dispatch.dispatcher import KernelDispatcher

        self._device = device
        self._max_seq = max_seq
        self._lm_head_on_device = lm_head_on_device

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

        self._dispatcher = KernelDispatcher(device)

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
        hidden_tt = self._embed(token_id)
        for i in range(len(self._layers)):
            hidden_tt = self._layer_forward(hidden_tt, i, kv_pos)
        return self._lm_head(hidden_tt)

    def _embed(self, token_id: int):
        """Lookup embedding, apply optional scale, and upload to device as (TILE, hidden_p)."""
        h = self._embed_w_cpu[token_id].float() * self._embed_scale   # (hidden,)
        row = torch.zeros(TILE, _tile_align(self._cfg.hidden_size), dtype=torch.bfloat16)
        row[0, :self._cfg.hidden_size] = h.bfloat16()
        return _to_tt(row, self._device)

    def _layer_forward(self, hidden_tt, layer_idx: int, kv_pos: int):
        import ttnn
        lw = self._layers[layer_idx]

        def rmsnorm_buf():
            return self._rmsnorm_buf_tt

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
        else:
            # Llama / Qwen / Mistral pattern:
            #   hidden += attn(norm1(hidden))
            #   hidden += mlp(norm2(hidden))
            normed1_tt = self._dispatcher.rmsnorm(hidden_tt, lw.norm1_w, lw.norm1_sc, rmsnorm_buf())
            attn_out_tt = self._attention(normed1_tt, layer_idx, kv_pos)
            hidden_tt = ttnn.add(hidden_tt, attn_out_tt, **aot)

            normed2_tt = self._dispatcher.rmsnorm(hidden_tt, lw.norm2_w, lw.norm2_sc, rmsnorm_buf())
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
        qkv_cpu = ttnn.to_torch(qkv_tt)[0].float()   # (total_qkv_p,)
        if qkv_ot is None:
            ttnn.deallocate(qkv_tt)

        q_cpu = qkv_cpu[:lw.q_end].view(cfg.num_heads,    hd_p)[:, :hd].contiguous()
        k_cpu = qkv_cpu[lw.q_end:lw.k_end].view(cfg.num_kv_heads, hd_p)[:, :hd].contiguous()
        v_cpu = qkv_cpu[lw.k_end:].view(cfg.num_kv_heads, hd_p)[:, :hd].contiguous()

        # Optional per-head Q/K norm (Gemma 3, Qwen 3)
        if lw.q_norm_w is not None:
            q_cpu = self._apply_head_norm(q_cpu, lw.q_norm_w)
        if lw.k_norm_w is not None:
            k_cpu = self._apply_head_norm(k_cpu, lw.k_norm_w)

        # RoPE
        q_cpu = self._apply_rope(q_cpu, kv_pos)
        k_cpu = self._apply_rope(k_cpu, kv_pos)

        # Update CPU history mirrors (kept for debugging / fallback)
        self._k_hist_cpu[layer_idx][:, kv_pos, :] = k_cpu.float()
        self._v_hist_cpu[layer_idx][:, kv_pos, :] = v_cpu.float()

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
        return _matmul_safe(self._fa_out_tt, lw.o_w, self._device)

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
        act  = self._cfg.activation
        mot  = self._matmul_ot
        muot = self._mul_ot

        if lw.gate_w is None:
            # 2-proj MLP: act(up(x)) → down  (GPTNeoX, BLOOM, Starcoder2, OLMo)
            up_ot = self._up_scratch_tt if mot else None
            up_tt = _matmul_safe(normed_tt, lw.up_w, self._device, output_tensor=up_ot)
            if act == "silu":
                act_tt = ttnn.silu(up_tt)
            elif act in ("gelu_pytorch_tanh", "gelu_new", "gelu_fast", "gelu"):
                act_tt = ttnn.gelu(up_tt)
            else:
                act_tt = ttnn.relu(up_tt)
            if up_ot is None:
                ttnn.deallocate(up_tt)
            result = _matmul_safe(act_tt, lw.down_w, self._device)
            ttnn.deallocate(act_tt)
            return result

        # SwiGLU: act(gate(x)) * up(x) → down
        gate_ot = self._gate_scratch_tt if mot else None
        up_ot   = self._up_scratch_tt   if mot else None
        gate_tt = _matmul_safe(normed_tt, lw.gate_w, self._device, output_tensor=gate_ot)
        up_tt   = _matmul_safe(normed_tt, lw.up_w,   self._device, output_tensor=up_ot)

        act_ot = self._activated_scratch_tt if muot else None
        if act == "silu":
            activated_tt = ttnn.multiply(ttnn.silu(gate_tt), up_tt,
                                         **({"output_tensor": act_ot} if act_ot else {}))
        elif act in ("gelu_pytorch_tanh", "gelu_new", "gelu_fast", "gelu"):
            activated_tt = ttnn.multiply(ttnn.gelu(gate_tt), up_tt,
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
        return result

    def _lm_head(self, hidden_tt) -> int:
        """Final norm (device) + lm_head matmul + argmax (CPU). Returns next token id.

        lm_head runs on CPU in float32 for accuracy — bfloat16 argmax on device
        can mis-rank logits for large vocabularies due to precision loss.
        """
        import ttnn
        normed_tt  = self._dispatcher.rmsnorm(hidden_tt, self._final_norm_w_tt,
                                               self._final_norm_sc_tt,
                                               self._rmsnorm_buf_tt)
        hidden = self._cfg.hidden_size
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

        # Embedding (stays on CPU)
        self._embed_w_cpu = _find_embed_tokens(backbone).weight.detach().float()

        # Detect if this model uses Gemma-style (1+w) norms
        first_layer = layers_list[0]
        self._gemma_norm = (hasattr(first_layer, "pre_feedforward_layernorm") and
                            hasattr(first_layer, "post_feedforward_layernorm"))

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

        # lm_head
        self._lm_head_w_cpu = hf_model.lm_head.weight.detach().float()
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
            norm_style  = "gemma" if (has_pre_ff and has_post_ff) else "llama"
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
            if gate_raw is None:
                # 2-proj MLP: gate_w=None signals _mlp to skip gating
                gate_w_tt = None
                gate_b_tt = None
            else:
                gate_w_tt = self._upload_linear_w(gate_raw)
                gate_m    = getattr(mlp, "gate_proj", None)
                gate_b_tt = self._upload_bias(
                    getattr(gate_m, "bias", None) if gate_m else None,
                    cfg.intermediate_size)
            up_m   = getattr(mlp, "up_proj", None)
            up_b_tt = self._upload_bias(
                getattr(up_m, "bias", None) if up_m else None,
                cfg.intermediate_size)

            lw = _LayerWeights(
                norm1_w  = self._upload_norm_w(getattr(n1, "weight", None), gemma_norm),
                norm1_sc = self._make_scaler_tt(cfg.hidden_size),
                norm2_w  = self._upload_norm_w(getattr(n2, "weight", None), gemma_norm),
                norm2_sc = self._make_scaler_tt(cfg.hidden_size),
                qkv_w = _to_tt(qkv_cat, self._device),
                q_end = q_out_p,
                k_end = q_out_p + k_out_p,
                o_w   = self._upload_linear_w(_find_o_proj_weight(attn)),
                gate_w = gate_w_tt,
                up_w   = self._upload_linear_w(up_raw),
                down_w = self._upload_linear_w(down_raw),
                gate_b = gate_b_tt,
                up_b   = up_b_tt,
                norm3_w  = self._upload_norm_w(getattr(n3, "weight", None), gemma_norm) if has_pre_ff else None,
                norm3_sc = self._make_scaler_tt(cfg.hidden_size) if has_pre_ff else None,
                norm4_w  = self._upload_norm_w(getattr(n4, "weight", None), gemma_norm) if has_post_ff else None,
                norm4_sc = self._make_scaler_tt(cfg.hidden_size) if has_post_ff else None,
                norm_style = norm_style,
                q_norm_w = self._load_head_norm_w(attn, "q_norm", gemma_norm),
                k_norm_w = self._load_head_norm_w(attn, "k_norm", gemma_norm),
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
        """Upload bias as (TILE, out_p) or zeros if None."""
        if b is not None:
            b_1d = _pad_to_tiles(b.detach().bfloat16())
            b_2d = b_1d.unsqueeze(0).expand(TILE, -1).contiguous()
        else:
            b_2d = torch.zeros(TILE, _tile_align(out_dim), dtype=torch.bfloat16)
        return _to_tt(b_2d, self._device)

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
        q_out_p  = _tile_align(cfg.num_heads    * cfg.head_dim)
        kv_out_p = _tile_align(cfg.num_kv_heads * cfg.head_dim)
        qkv_p    = q_out_p + 2 * kv_out_p
        int_p    = _tile_align(cfg.intermediate_size)
        nkv      = cfg.num_kv_heads
        hd_p     = _tile_align(cfg.head_dim)

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

        # Linear RoPE scaling: compress positions by dividing by scale factor.
        # Do NOT change theta -- scale the position indices instead.
        scale_factor = 1.0
        rope_cfg = getattr(cfg, "rope_scaling", None)
        if rope_cfg:
            if rope_cfg.get("rope_type") == "linear":
                scale_factor = rope_cfg.get("factor", 1.0)

        positions = torch.arange(max_pos, dtype=torch.float32) / scale_factor
        freqs     = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        angles    = torch.outer(positions, freqs)      # (max_pos, dim/2)
        cos_half  = torch.cos(angles)                  # (max_pos, dim/2)
        sin_half  = torch.sin(angles)

        # HuggingFace convention: expand to full dim by tiling [half, half].
        # Rotation uses grouped halves (i, i+dim//2) not interleaved pairs (i, i+1).
        self._rope_cos = torch.cat([cos_half, cos_half], dim=-1)  # (max_pos, dim)
        self._rope_sin = torch.cat([sin_half, sin_half], dim=-1)

    def _apply_head_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply per-head RMSNorm. x: (n_heads, head_dim), w: (head_dim,) already (1+w) or w."""
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return (x / rms) * w

    def _apply_rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Apply RoPE to x (n_heads, head_dim) using HuggingFace convention."""
        if position >= self._rope_cos.shape[0]:
            return x
        cos = self._rope_cos[position]   # (dim,)
        sin = self._rope_sin[position]
        dim = x.shape[-1]
        # Rotate: x_rot = [-x[dim//2:], x[:dim//2]]
        x_rot = torch.cat([-x[..., dim // 2:], x[..., :dim // 2]], dim=-1)
        return x * cos + x_rot * sin


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
    args = parser.parse_args()

    import ttnn
    device = ttnn.open_device(device_id=0)
    try:
        runner = TTModelRunner(args.model, device, max_seq=args.max_seq)
        print(f"\nPROMPT: {args.prompt}")
        output = runner.generate(args.prompt, max_new_tokens=args.max_tokens,
                                 chat=not args.no_chat)
        print(f"OUTPUT: {output}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
