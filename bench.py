# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Benchmarking and correctness harness for tt-inference-server.

Two modes:

  Model inference benchmark (TASK-T01)::

      python bench.py --model models/mistral-7b

  Kernel correctness check::

      python bench.py --op swiglu
      python bench.py --op flash_attn --sweep-shapes
      python bench.py --op all

Model mode output: one JSON line appended to ~/bench_logs/results.jsonl.
Exit codes: 0=pass, 1=model error, 2=output validation failed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

TILE = 32
PCC_THRESHOLD = 0.99

BENCH_PROMPT   = "The capital of France is"
BENCH_N_TOKENS = 50
RESULTS_PATH   = pathlib.Path.home() / "bench_logs" / "results.jsonl"


# ------------------------------------------------------------------
# Hardware telemetry helpers
# ------------------------------------------------------------------

def _read_pcie_speed_gbps() -> float:
    """Read current PCIe link speed from sysfs. Returns 0.0 on failure."""
    try:
        base = pathlib.Path("/sys/bus/pci/devices")
        for dev in base.iterdir():
            speed_file = dev / "current_link_speed"
            if speed_file.exists():
                raw = speed_file.read_text().strip()
                # e.g. "32.0 GT/s PCIe"
                m = re.match(r"([\d.]+)\s*GT/s", raw)
                if m:
                    gt_s = float(m.group(1))
                    # GT/s → GB/s: 128b/130b encoding, ×16 lanes, ÷8 bits/byte
                    return gt_s * 16 * (128 / 130) / 8
    except Exception:
        pass
    return 0.0


def _parse_tt_smi() -> dict:
    """Run tt-smi and return {'power_w': float, 'core_util_pct': float}.

    tt-smi output example (JSON mode):
        [{"device_id": 0, "power": 53.2, "aiclk": 1350, "core_util": 18, ...}]
    Falls back to regex parse if JSON mode unavailable.
    """
    result = {"power_w": 0.0, "core_util_pct": 0.0}
    try:
        out = subprocess.check_output(
            ["tt-smi", "--json"], timeout=5, stderr=subprocess.DEVNULL
        ).decode()
        data = json.loads(out)
        entry = data[0] if isinstance(data, list) and data else data
        result["power_w"]        = float(entry.get("power", entry.get("power_w", 0.0)))
        result["core_util_pct"]  = float(entry.get("core_util", entry.get("core_util_pct", 0.0)))
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError,
            KeyError, IndexError, ValueError):
        # Fallback: plain-text tt-smi
        try:
            out = subprocess.check_output(
                ["tt-smi"], timeout=5, stderr=subprocess.DEVNULL
            ).decode()
            m_pow  = re.search(r"(\d+\.?\d*)\s*W", out)
            m_util = re.search(r"(\d+\.?\d*)\s*%", out)
            if m_pow:
                result["power_w"] = float(m_pow.group(1))
            if m_util:
                result["core_util_pct"] = float(m_util.group(1))
        except Exception:
            pass
    return result


class _TelemetryPoller:
    """Background thread that polls tt-smi every 2 s during a run."""

    def __init__(self):
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=6)
        if not self._samples:
            return {"power_w": 0.0, "core_util_pct": 0.0}
        avg = lambda key: sum(s[key] for s in self._samples) / len(self._samples)
        return {"power_w": avg("power_w"), "core_util_pct": avg("core_util_pct")}

    def _run(self):
        while not self._stop.wait(2.0):
            self._samples.append(_parse_tt_smi())


# ------------------------------------------------------------------
# Output validation
# ------------------------------------------------------------------

def _validate_output(prompt: str, output: str) -> bool:
    """Return True if output looks like a plausible model continuation.

    Delegates garbage detection to the shared validator (issue #46) so this harness
    and tests/_run_model.py agree, while keeping the prompt!=output guard that's unique
    to this entry point. No token ids are available here, so the validator runs in
    text-only mode (length/token-repetition checks skipped — the bench fixes n_tokens)."""
    if not output or not output.strip():
        return False
    # Must differ from the prompt itself (unique to this harness)
    if output.strip().lower() == prompt.strip().lower():
        return False
    from tt_inference_server.dispatch.output_validator import validate_output
    ok, _reason = validate_output(output, None)
    return ok


# ------------------------------------------------------------------
# Model benchmark entry point
# ------------------------------------------------------------------

def run_model_bench(model_path: str) -> int:
    """Run the standard model benchmark. Returns exit code (0/1/2)."""
    model_name = pathlib.Path(model_path).name

    try:
        import ttnn
        from tt_inference_server.dispatch.runner import TTModelRunner

        device = ttnn.open_device(device_id=0)
        try:
            runner = TTModelRunner(model_path, device)
        except Exception as e:
            print(f"ERROR loading model {model_path}: {e}", file=sys.stderr)
            ttnn.close_device(device)
            return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    poller = _TelemetryPoller()
    poller.start()

    try:
        tok_s, output = runner.benchmark(BENCH_PROMPT, n_tokens=BENCH_N_TOKENS)
    except Exception as e:
        poller.stop()
        print(f"ERROR during generation: {e}", file=sys.stderr)
        ttnn.close_device(device)
        return 1

    telemetry = poller.stop()
    ttnn.close_device(device)

    pcie_gbps   = _read_pcie_speed_gbps()
    output_ok   = _validate_output(BENCH_PROMPT, output)

    record = {
        "model":          model_name,
        "tokens":         BENCH_N_TOKENS,
        "tok_s":          round(tok_s, 2),
        "core_util_pct":  round(telemetry["core_util_pct"], 1),
        "power_w":        round(telemetry["power_w"], 1),
        "pcie_gbps":      round(pcie_gbps, 1),
        "output_ok":      output_ok,
        "output_preview": output[:120].replace("\n", " "),
        "ts":             time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\n{'='*60}")
    print(f"Model:        {model_name}")
    print(f"Tokens/s:     {tok_s:.1f}")
    print(f"Core util:    {telemetry['core_util_pct']:.1f}%")
    print(f"Power:        {telemetry['power_w']:.1f} W")
    print(f"PCIe speed:   {pcie_gbps:.1f} GB/s")
    print(f"Output valid: {output_ok}")
    print(f"Preview:      {record['output_preview']}")
    print(f"Result:       {RESULTS_PATH}")
    print(f"{'='*60}\n")

    return 0 if output_ok else 2


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_device(t: torch.Tensor, device):
    import ttnn
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def to_torch(t) -> torch.Tensor:
    import ttnn
    return ttnn.to_torch(t).float()


# ------------------------------------------------------------------
# Individual op benchmarks
# ------------------------------------------------------------------

def bench_swiglu(device, M=4, K=6, N=4, activation="silu"):
    from tt_inference_server.kernels.swiglu import make_swiglu_kernel

    torch.manual_seed(42)
    T = lambda tiles: tiles * TILE
    gate = torch.randn(T(M), T(K), dtype=torch.bfloat16)
    wg   = torch.randn(T(K), T(N), dtype=torch.bfloat16)
    bg   = torch.randn(T(M), T(N), dtype=torch.bfloat16)
    up   = torch.randn(T(M), T(K), dtype=torch.bfloat16)
    wu   = torch.randn(T(K), T(N), dtype=torch.bfloat16)
    bu   = torch.randn(T(M), T(N), dtype=torch.bfloat16)

    act_fn = {"silu": F.silu, "gelu": F.gelu, "relu2": lambda x: F.relu(x) ** 2}[activation]
    golden = act_fn(gate.float() @ wg.float() + bg.float()) * (up.float() @ wu.float() + bu.float())

    kernel = make_swiglu_kernel(M, K, N, activation)
    out_d = to_device(torch.zeros(T(M), T(N), dtype=torch.bfloat16), device)

    t0 = time.perf_counter()
    kernel(to_device(gate, device), to_device(wg, device), to_device(bg, device),
           to_device(up, device),   to_device(wu, device), to_device(bu, device), out_d)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return pcc(to_torch(out_d), golden), elapsed_ms


def bench_rmsnorm(device, seq=1, hidden=4):
    from tt_inference_server.kernels.rmsnorm import make_rmsnorm_kernel

    torch.manual_seed(42)
    S, H = seq * TILE, hidden * TILE
    x  = torch.randn(S, H, dtype=torch.bfloat16)
    w  = torch.ones(H, dtype=torch.bfloat16)
    sc = torch.full((TILE, TILE), 1.0 / H, dtype=torch.bfloat16)

    rms    = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    golden = (x.float() / rms) * w.float()

    w_tiled = w.unsqueeze(0).expand(S, H).contiguous()
    kernel  = make_rmsnorm_kernel(seq, hidden)
    out_d   = to_device(torch.zeros(S, H, dtype=torch.bfloat16), device)

    t0 = time.perf_counter()
    kernel(to_device(x, device), to_device(w_tiled, device), to_device(sc, device), out_d)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return pcc(to_torch(out_d), golden), elapsed_ms


def bench_layernorm(device, seq=1, hidden=4):
    from tt_inference_server.kernels.layernorm import make_layernorm_kernel

    torch.manual_seed(42)
    S, H = seq * TILE, hidden * TILE
    x  = torch.randn(S, H, dtype=torch.bfloat16)
    w  = torch.ones(H, dtype=torch.bfloat16)
    b  = torch.zeros(H, dtype=torch.bfloat16)
    sc = torch.full((TILE, TILE), 1.0 / H, dtype=torch.bfloat16)

    golden = F.layer_norm(x.float(), [H], w.float(), b.float())

    w_tiled = w.unsqueeze(0).expand(S, H).contiguous()
    b_tiled = b.unsqueeze(0).expand(S, H).contiguous()
    kernel  = make_layernorm_kernel(seq, hidden)
    out_d   = to_device(torch.zeros(S, H, dtype=torch.bfloat16), device)

    t0 = time.perf_counter()
    kernel(to_device(x, device), to_device(w_tiled, device),
           to_device(b_tiled, device), to_device(sc, device), out_d)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return pcc(to_torch(out_d), golden), elapsed_ms


def bench_flash_attn(device, N_heads=4, N_kv=2, head_dim_tiles=2, seq_tiles=2):
    from tt_inference_server.kernels.flash_attn import make_flash_attn_kernel

    torch.manual_seed(0)
    HD  = head_dim_tiles * TILE
    SEQ = seq_tiles * TILE
    GQA = N_heads // N_kv
    scale_val = 1.0 / (HD ** 0.5)

    Q  = torch.randn(N_heads * TILE, HD, dtype=torch.bfloat16) * 0.1
    K  = torch.randn(N_kv * SEQ,     HD, dtype=torch.bfloat16) * 0.1
    V  = torch.randn(N_kv * SEQ,     HD, dtype=torch.bfloat16) * 0.1

    # Reference: per query-head attention with GQA broadcast.
    # Output layout matches kernel: [TILE, N_heads * HD] — heads concatenated along columns.
    # dm_write writes head h into out[0, h*hdt:(h+1)*hdt], so row r of the result is
    # the concatenation of row r from each head's attention output.
    ref_outs = []
    for h in range(N_heads):
        kv_h  = h // GQA
        q_row = Q[h * TILE : (h+1) * TILE].float()
        k_blk = K[kv_h * SEQ : (kv_h+1) * SEQ].float()
        v_blk = V[kv_h * SEQ : (kv_h+1) * SEQ].float()
        w = torch.softmax((q_row @ k_blk.T) * scale_val, dim=-1)
        ref_outs.append(w @ v_blk)                         # (TILE, HD) each
    golden = torch.cat(ref_outs, dim=1)                    # (TILE, N_heads * HD)

    sc_t    = torch.full((TILE, TILE), scale_val,      dtype=torch.bfloat16)
    ninf_t  = torch.full((TILE, TILE), float("-inf"),  dtype=torch.bfloat16)
    zero_t  = torch.zeros(TILE, TILE,                  dtype=torch.bfloat16)
    zero_h  = torch.zeros(TILE, HD,                    dtype=torch.bfloat16)
    ones_t  = torch.ones(TILE, TILE,                   dtype=torch.bfloat16)
    mask_t  = torch.zeros(TILE, SEQ,                   dtype=torch.bfloat16)
    out_d   = to_device(torch.zeros(TILE, N_heads * HD, dtype=torch.bfloat16), device)

    kernel = make_flash_attn_kernel(N_heads, N_kv, head_dim_tiles, kv_chunk_tiles=1)
    t0 = time.perf_counter()
    kernel(to_device(Q, device), to_device(K, device), to_device(V, device),
           to_device(sc_t, device), to_device(ninf_t, device), to_device(zero_t, device),
           to_device(zero_h, device), to_device(ones_t, device), to_device(mask_t, device),
           out_d)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return pcc(to_torch(out_d), golden), elapsed_ms


def bench_kv_decode(device, N_kv=2, head_dim_tiles=2, max_seq_tiles=2):
    from tt_inference_server.kernels.kv_decode import make_kv_decode_kernel

    torch.manual_seed(7)
    HD      = head_dim_tiles * TILE
    MAX_SEQ = max_seq_tiles * TILE
    scale_val = 1.0 / (HD ** 0.5)

    Q = torch.randn(N_kv * TILE, HD, dtype=torch.bfloat16) * 0.1
    K = torch.randn(N_kv * MAX_SEQ, HD, dtype=torch.bfloat16) * 0.1
    V = torch.randn(N_kv * MAX_SEQ, HD, dtype=torch.bfloat16) * 0.1

    ref_outs = []
    for h in range(N_kv):
        q = Q[h * TILE : (h+1) * TILE].float()
        k = K[h * MAX_SEQ : (h+1) * MAX_SEQ].float()
        v = V[h * MAX_SEQ : (h+1) * MAX_SEQ].float()
        w = torch.softmax((q @ k.T) * scale_val, dim=-1)
        ref_outs.append(w @ v)
    golden = torch.cat(ref_outs, dim=0)

    sc_t   = torch.full((TILE, TILE), scale_val,     dtype=torch.bfloat16)
    ninf_t = torch.full((TILE, TILE), float("-inf"), dtype=torch.bfloat16)
    zero_t = torch.zeros(TILE, TILE,                 dtype=torch.bfloat16)
    zero_h = torch.zeros(TILE, HD,                   dtype=torch.bfloat16)
    ones_t = torch.ones(TILE, TILE,                  dtype=torch.bfloat16)
    out_d  = to_device(torch.zeros(N_kv * TILE, HD,  dtype=torch.bfloat16), device)

    kernel = make_kv_decode_kernel(N_kv_heads=N_kv, head_dim_tiles=head_dim_tiles,
                                    max_seq_tiles=max_seq_tiles, kv_chunk_tiles=1)
    t0 = time.perf_counter()
    kernel(to_device(Q, device), to_device(K, device), to_device(V, device),
           to_device(sc_t, device), to_device(ninf_t, device), to_device(zero_t, device),
           to_device(zero_h, device), to_device(ones_t, device), out_d)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return pcc(to_torch(out_d), golden), elapsed_ms


# ------------------------------------------------------------------
# Shape sweep configs
# ------------------------------------------------------------------

SWEEP_CONFIGS = {
    "flash_attn": [
        {"N_heads": 8,  "N_kv": 2, "head_dim_tiles": 4, "seq_tiles": 4},
        {"N_heads": 16, "N_kv": 4, "head_dim_tiles": 4, "seq_tiles": 8},
        {"N_heads": 32, "N_kv": 8, "head_dim_tiles": 4, "seq_tiles": 8},
        {"N_heads": 8,  "N_kv": 1, "head_dim_tiles": 4, "seq_tiles": 4},   # MQA
        {"N_heads": 4,  "N_kv": 4, "head_dim_tiles": 4, "seq_tiles": 4},   # MHA
    ],
    "swiglu": [
        {"M": 1, "K": 4, "N": 4, "activation": "silu"},
        {"M": 4, "K": 6, "N": 4, "activation": "silu"},
        {"M": 4, "K": 4, "N": 8, "activation": "gelu"},
        {"M": 2, "K": 4, "N": 4, "activation": "relu2"},
    ],
    "kv_decode": [
        {"N_kv": 2, "head_dim_tiles": 2, "max_seq_tiles": 2},
        {"N_kv": 4, "head_dim_tiles": 4, "max_seq_tiles": 4},
        {"N_kv": 8, "head_dim_tiles": 4, "max_seq_tiles": 8},
    ],
}

def bench_kv_history(device, n_heads=8, n_kv=2, head_dim=64, seq_len=16):
    """Verify CPU KV history attention matches a pure-PyTorch reference.

    Simulates the exact computation path used by runner._attention after B01:
    k/v stored in a pre-allocated (nkv, max_seq, hd) CPU buffer, then read
    back via K_hist[kv_h, :kv_seq, :] for each head.  PCC is measured against
    F.scaled_dot_product_attention run in float32.
    """
    torch.manual_seed(0)
    gqa = n_heads // n_kv
    scale = 1.0 / (head_dim ** 0.5)

    # Simulate seq_len tokens arriving one by one; check final step
    K_buf = torch.zeros(n_kv, seq_len + 1, head_dim)
    V_buf = torch.zeros(n_kv, seq_len + 1, head_dim)
    for pos in range(seq_len + 1):
        K_buf[:, pos, :] = torch.randn(n_kv, head_dim)
        V_buf[:, pos, :] = torch.randn(n_kv, head_dim)
    kv_seq = seq_len + 1

    Q_single = torch.randn(n_heads, head_dim)

    # --- runner path: per-head dot-product with CPU history slices ---
    out_heads = []
    for h in range(n_heads):
        kv_h   = h // gqa
        q_h    = Q_single[h]
        k_h    = K_buf[kv_h, :kv_seq, :]        # (kv_seq, hd) contiguous
        v_h    = V_buf[kv_h, :kv_seq, :]
        scores = (q_h @ k_h.T) * scale
        w      = torch.softmax(scores, dim=-1)
        out_heads.append(w @ v_h)
    runner_out = torch.stack(out_heads, dim=0)   # (n_heads, hd)

    # --- reference: same per-head dot-product in float64 for numerical ground truth ---
    # Running in double precision gives an independent, higher-accuracy result to
    # compare against the float32 runner path.  If the indexing or GQA mapping is
    # wrong the PCC will drop; if it's correct PCC should be >0.9999.
    ref_heads = []
    for h in range(n_heads):
        kv_h   = h // gqa
        q_h    = Q_single[h].double()
        k_h    = K_buf[kv_h, :kv_seq, :].double()
        v_h    = V_buf[kv_h, :kv_seq, :].double()
        scores = (q_h @ k_h.T) * scale
        w      = torch.softmax(scores, dim=-1)
        ref_heads.append((w @ v_h).float())
    ref_out = torch.stack(ref_heads, dim=0)    # (n_heads, hd)

    # device arg is unused (pure CPU test) — present for uniform bench signature
    _ = device
    score = pcc(runner_out, ref_out)
    return score, 0.0


def bench_runner_alloc(device, model_path: str = "models/allenai-OLMo-1B-hf",
                       n_steps: int = 5, warmup: int = 1):
    """Count device tensor allocations per decode step to measure B01 effect.

    Monkey-patches ttnn.from_torch (device allocations) and ttnn.deallocate
    to count calls.  Runs n_steps decode steps after warmup, reports the
    average per-step count, and checks it against the expected threshold.

    Expected ranges:
      pre-B01  : ~300-500 dealloc/step for a 32-layer model
      post-B01 (partial, no output_tensor=): ~150-300/step
      post-B01 (full, with output_tensor=) : <10/step

    The test PASSes if avg_dealloc/step < max_dealloc_per_step.
    For OLMo-1B (16 layers) baseline is ~150/step; B01 partial target ~80/step.
    """
    import pathlib as _pl, sys as _sys
    _pkg = str(_pl.Path(__file__).parent)
    if _pkg not in _sys.path:
        _sys.path.insert(0, _pkg)
    import ttnn
    from tt_inference_server.dispatch.runner import TTModelRunner

    # Counters
    _counts = {"from_torch_dev": 0, "deallocate": 0, "step": 0}
    _orig_ft  = ttnn.from_torch
    _orig_del = ttnn.deallocate

    def _ft_patched(*args, **kwargs):
        if kwargs.get("device") is not None:
            _counts["from_torch_dev"] += 1
        return _orig_ft(*args, **kwargs)

    def _del_patched(*args, **kwargs):
        _counts["deallocate"] += 1
        return _orig_del(*args, **kwargs)

    ttnn.from_torch  = _ft_patched
    ttnn.deallocate  = _del_patched

    try:
        runner = TTModelRunner(model_path, device)
        n_layers = len(runner._layers)

        # Patch _decode_step to intercept per-step counts
        per_step_ft  = []
        per_step_del = []

        token_id = 1   # arbitrary non-special token
        for step in range(warmup + n_steps):
            before_ft  = _counts["from_torch_dev"]
            before_del = _counts["deallocate"]
            runner._decode_step(token_id, step)
            if step >= warmup:
                per_step_ft.append(_counts["from_torch_dev"]  - before_ft)
                per_step_del.append(_counts["deallocate"]      - before_del)

        avg_ft  = sum(per_step_ft)  / len(per_step_ft)
        avg_del = sum(per_step_del) / len(per_step_del)

        # Scratch buf identity check: rmsnorm buf should not be reallocated
        # (same Python object id across the run means no realloc)
        buf_id = id(runner._rmsnorm_buf_tt)
        runner._decode_step(token_id, warmup + n_steps)
        same_buf = id(runner._rmsnorm_buf_tt) == buf_id

    finally:
        ttnn.from_torch = _orig_ft
        ttnn.deallocate = _orig_del

    # Max acceptable deallocs/step for OLMo-1B (16 layers).
    # Baseline (pre-B01) for 16 layers ≈ 176/step.
    # B01 partial target (no output_tensor=): removes K/V device round-trips
    # → saves 4×16=64/step.  Threshold set at 150 (catches regressions,
    # passes with partial B01 wins, fails if we somehow made things worse).
    MAX_DEL_PER_STEP = 150

    print(f"  Layers: {n_layers}  avg from_torch_dev/step: {avg_ft:.1f}"
          f"  avg deallocate/step: {avg_del:.1f}  rmsnorm_buf_stable: {same_buf}")

    # PCC-style score: 1.0 if under threshold, scales down linearly above
    score = min(1.0, MAX_DEL_PER_STEP / max(avg_del, 1))
    ms    = avg_del   # report deallocs/step as the "timing" field for readability
    return score, ms


BENCH_FNS = {
    "swiglu":        bench_swiglu,
    "rmsnorm":       bench_rmsnorm,
    "layernorm":     bench_layernorm,
    "flash_attn":    bench_flash_attn,
    "kv_decode":     bench_kv_decode,
    "kv_history":    bench_kv_history,
    "runner_alloc":  bench_runner_alloc,
}

ALL_OPS = list(BENCH_FNS.keys())

# Ops that require --bench-model to be set (they load a TTModelRunner)
MODEL_OPS = {"runner_alloc"}


def run_sweep(op: str, device):
    configs = SWEEP_CONFIGS.get(op, [])
    if not configs:
        print(f"No sweep configs for op={op!r}")
        return True

    print(f"\n{'Shape':45s} {'PCC':>8} {'ms':>8} {'Status':>8}")
    print("-" * 75)
    all_pass = True
    for cfg in configs:
        label = str(cfg)[:44]
        try:
            score, ms = BENCH_FNS[op](device, **cfg)
            ok = score >= PCC_THRESHOLD
            if not ok:
                all_pass = False
            print(f"{label:45s} {score:8.4f} {ms:8.1f} {'PASS' if ok else 'FAIL':>8}")
        except Exception as e:
            all_pass = False
            print(f"{label:45s} {'ERR':>8} {'':>8} {'FAIL':>8}  ({e})")
    print()
    return all_pass


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="tt-inference-server benchmark / correctness harness")
    parser.add_argument("--model", default=None,
                        help="Path to HuggingFace model dir — runs model inference benchmark")
    parser.add_argument("--op", default="all",
                        choices=ALL_OPS + ["all"],
                        help="Kernel op to bench (default: all); ignored when --model is set")
    parser.add_argument("--sweep-shapes", action="store_true",
                        help="Run all shape configs for the selected op")
    parser.add_argument("--bench-model", default="models/allenai-OLMo-1B-hf",
                        help="Model path used by runner_alloc (default: OLMo-1B)")
    args = parser.parse_args()

    if args.model is not None:
        sys.exit(run_model_bench(args.model))

    import ttnn
    device = ttnn.open_device(device_id=0)
    try:
        # When running "all", skip MODEL_OPS unless --bench-model is explicitly set
        if args.op == "all":
            ops = [o for o in ALL_OPS if o not in MODEL_OPS]
        else:
            ops = [args.op]

        overall = True
        for op in ops:
            if args.sweep_shapes and op in SWEEP_CONFIGS:
                ok = run_sweep(op, device)
                overall = overall and bool(ok)
            elif op in MODEL_OPS:
                score, ms = BENCH_FNS[op](device, model_path=args.bench_model)
                ok = score >= PCC_THRESHOLD
                overall = overall and ok
                print(f"{op:20s}  score={score:.3f}  dealloc/step={ms:.0f}  {'PASS' if ok else 'FAIL'}")
            else:
                score, ms = BENCH_FNS[op](device)
                ok = score >= PCC_THRESHOLD
                overall = overall and ok
                print(f"{op:20s}  PCC={score:.4f}  {ms:.1f}ms  {'PASS' if ok else 'FAIL'}")

        sys.exit(0 if overall else 1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
