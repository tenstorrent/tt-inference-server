# Forge LLM memory observations on P150

> Working notes from 2026-05-27/28 experiments. **Not a final analysis** — captures what's been verified, what's been inferred, and what still needs a sweep. Intended as a starting point for a dedicated investigation session.

## Hardware reference

- **P150** (Blackhole): 32 GB HBM total, ~34 GB allocatable across **8 DRAM banks ≈ 4.27 GB each**.
- All experiments below: 1 chip / `tensor_parallel_size=1`.

## The 12 GB hardcode

`vllm_tt/worker.py:236` (inside the forge wheel, inside the docker image):

```python
total_memory_size = 12 * 1024**3  # m["bytes_limit"]  ← hardcoded
usable_memory_size = int(total_memory_size * self.cache_config.gpu_memory_utilization)
tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)  # profiled is currently 0
```

So vLLM's reported "KV cache memory" = `total_memory_size × gpu_memory_utilization`. With the 12 GB hardcode, the KV pool ceiling is ~10.8 GB regardless of actual device DRAM.

**Workaround in flight (merged on followup branch):** `TT_KV_POOL_GB` env var, applied via a startup-script sed in `tt-media-server/run_uvicorn.sh`. Sets the literal to whatever value we want. Upstream tt-xla fix will replace the hardcode with the real device DRAM value; same effective behavior with a different math knob (gmu adjusted down).

## On-device memory components (estimates)

For Llama-3.1-8B on P150, ~34 GB budget:

| Component | Estimated size | How we know |
|---|---|---|
| **Weights** | **~8 GB** (not 16 GB) | Inferred from OOM math: at `TT_KV_POOL_GB=24, gmu=0.9, b=4, 16K`, total allocated reached ~33.4 GB. That only fits if weights < 16 GB. Suggests forge/tt-mlir packs weights to bfp8-ish on-device while exposing bfloat16 at the PyTorch API level. **Needs direct verification.** |
| **KV pool** | `total_memory_size × gmu` | Confirmed by formula match. At `(32 × 0.3)` = 9.6 GB the reported `GPU KV cache size: 78,624 tokens × 128 KB/token = 10 GB` ✓ |
| **Trace activation buffers** | **single largest tensor: ~7 GB at b×seq=128K** | Measured by OOM signature (v5/v6). Compiled graph reserves max-shaped intermediates. Grows linearly-ish with `b × seq`. |
| **vLLM bookkeeping** (block tables, prefix cache scratch, etc.) | ~1–2 GB | Standard overhead |

The **bytes-per-token for KV** is fixed by model architecture:
- Llama-3.1-8B (32 layers × 8 KV heads × 128 head_dim × 2 K+V × 2 bytes): **128 KB/token**
- Llama-3.2-3B (28 layers × 8 KV heads × 128 head_dim × 2 K+V × 2 bytes): **112 KB/token**

## Experimental matrix

All on `Llama-3.1-8B-Instruct` on P150 unless noted.

| Tag | `TT_KV_POOL_GB` | `gmu` | `MAX_NUM_SEQS` | `MAX_MODEL_LENGTH` | `b × seq` | KV pool | Reported concurrency | Result |
|---|---|---|---|---|---|---|---|---|
| Production baseline | (unset) | 0.6 | 16 | 4K | 64K | 7.2 GB | 14.4× | ✅ works |
| Production 3B | (unset) | 0.7 | 16 | 8K | 128K | 8.4 GB | ~9.4× | ✅ works in practice (vLLM preempts as needed) |
| **OOM v1** | 24 | 0.9 | 4 | 16K | 64K | **21.6 GB** | 10.8× | ❌ OOM at trace — KV pool too greedy |
| **v2 (first 16K success)** | 16 | 0.6 | 2 | 16K | 32K | 9.6 GB | 4.80× | ✅ /health at +10 min, inference OK |
| **v3** | 16 | 0.6 | 4 | 16K | 64K | 9.6 GB | 4.80× | ✅ /health at +11.5 min, inference OK |
| **v4 (equivalence test)** | 32 | 0.3 | 4 | 16K | 64K | 9.6 GB | 4.80× | ✅ identical KV alloc to v3 — confirms `(32, 0.3) ≡ (16, 0.6)` |
| **OOM v5** | 32 | 0.4 | 8 | 16K | 128K | 12.8 GB | 6.40× | ❌ OOM at trace — 7 GB tensor fragmentation |
| **OOM v6** | 32 | 0.3 | 8 | 16K | 128K | 9.6 GB | 4.80× | ❌ OOM at trace — same fragmentation despite more headroom |

## Three big findings

### 1. KV pool size is governed purely by `total_memory_size × gpu_memory_utilization`

Verified empirically: v3 `(16 × 0.6 = 9.6 GB)` and v4 `(32 × 0.3 = 9.6 GB)` produced identical `GPU KV cache size: 78,624 tokens` and `Maximum concurrency: 4.80×`. The hardcode is just one factor in a product. The upstream tt-xla fix that replaces the hardcode with real DRAM size + paired gmu adjustment will produce the **same effective behavior**.

### 2. `b × seq = 128K` is a fragmentation wall, not a budget wall

| What didn't fit (same in v5 and v6) | 7,516,192,768 B single tensor (7 GB), distributed as 896 MB per bank |
|---|---|
| v5 (gmu=0.4): per-bank free | 880 MB — slightly under target |
| v6 (gmu=0.3): per-bank free | 1351 MB — *more than* the required 896 MB! |
| **v6 largest free contiguous block per bank** | **556 MB** ← the actual blocker |

Even with 50% more total free memory in v6, the same tensor failed because the **largest contiguous block per bank** was too small. **Shrinking the KV pool doesn't help** — this is a tt-metal allocator fragmentation issue triggered by the compiled graph's max-shaped intermediates at b×seq=128K.

**Implication: anywhere on the 128K isobar fails the same way:**

- `b=8 × 16K = 128K` — confirmed failure
- `b=16 × 8K = 128K` — predicted same failure (not directly tested, but same b×seq product)
- `b=4 × 32K = 128K` — predicted same failure

### 3. The viable configs for 8B on P150 form a `b × seq ≤ ~64K` boundary

For Llama-3.1-8B specifically:

| Config | `b × seq` | Status |
|---|---|---|
| b=16 × 4K | 64K | ✅ production |
| b=8 × 8K | 64K | unverified but should work (same `b × seq`) |
| **b=4 × 16K** | **64K** | ✅ **verified (v3/v4)** |
| b=8 × 16K | 128K | ❌ allocator fragmentation |
| b=16 × 8K | 128K | ❌ same (predicted) |
| b=16 × 16K | 256K | ❌ KV alone exceeds device |

## Llama-3.2-3B is structurally different

Smaller weights (~6 GB at bfloat16 / ~3 GB at bfp8) and smaller per-token KV (112 KB vs 128 KB). Currently runs at `b=16 × 8K`. To push to 16K:

| Config | KV pool needed | Other budget | Verdict |
|---|---|---|---|
| b=16 × 16K | 16 × 16K × 112 KB = **28 GB** | weights + activations also ↑↑ | ❌ KV alone exceeds device |
| b=4 × 16K | 7 GB | similar regime to 8B v4 | likely ✅ (untested) |
| b=8 × 16K | 14 GB | b×seq=128K — fragmentation wall? | TBD — smaller weights may give more headroom |

**Worth a dedicated sweep.** 3B's smaller weight footprint gives meaningfully more activation budget room — the b×seq=128K fragmentation wall *may* not apply, or may apply at a different boundary.

## What "Maximum concurrency" reported by vLLM means

It's `pool_size / (max_seq × bytes_per_token)`. **It only describes the KV pool — not the activation budget.** vLLM happily reports e.g. 6.4× concurrency at the v5 config, then OOMs during warmup because activations don't fit. **Reported concurrency is a necessary but not sufficient predictor of "fits".**

Practical rule:
- `MAX_NUM_SEQS ≤ reported_concurrency` → KV is happy
- `b × seq ≤ ~64K` for 8B Llama on P150 → activations are happy
- **Both have to be true.**

## Open questions for a focused sweep

The current data is from chasing specific configs end-to-end. A proper sweep would tease apart:

1. **Direct device memory measurement.** Can we hook `tt-smi -s` or another telemetry call to get a real-time DRAM breakdown during engine init? Right now we infer from OOM messages and forward math.
2. **Is the on-device weight footprint really ~8 GB for 8B Llama?** Or 16 GB? Or something between? Probes:
   - Does forge / tt-mlir / pjrt-plugin-tt have a config knob that controls dtype on device?
   - Compare a forced-bfloat16 vs default load and see if total allocated changes.
3. **Where exactly is the 7 GB tensor?** Is it MLP up-projection scratch? Attention intermediate? Some compiler-introduced double-buffer? Knowing this would tell us whether tt-mlir has a `--reduce-max-buffer-size` style flag we can pass.
4. **Does `enable_prefix_caching=True` (currently on) inflate the activation budget?** Disabling it may free some HBM but cost throughput on common-prefix workloads.
5. **Can we parameter-sweep around the 128K isobar?**
   - `b=6 × 16K = 96K` — between known-good and known-bad
   - `b=8 × 12K = 96K` — same point on the isobar but different shape
   - Tells us whether the wall is at 96K, 100K, 128K, etc.
6. **What's the equivalent boundary for 3B?** Could be `b × seq ≤ 96K` or higher given smaller weights.
7. **What's the equivalent boundary for Falcon3-7B and Qwen3-8B?** Their KV-per-token may differ from Llama-3.1-8B's 128 KB.
8. **Does compile-graph splitting help?** vLLM's `compile_ranges_split_points` currently splits at `b × seq` boundaries. Could a smaller split reduce the worst-case tensor and dodge fragmentation?

## What works today on P150 (forge, 1 chip)

Conservative table, all verified end-to-end at `/health=200` + inference:

| Model | Config | Notes |
|---|---|---|
| Falcon3-7B-Instruct | b=16 × 4K | gmu=0.6, no TT_KV_POOL_GB knob |
| Llama-3.1-8B-Instruct | b=16 × 4K | gmu=0.6, no TT_KV_POOL_GB |
| Llama-3.1-8B-Instruct (long-context) | **b=4 × 16K** | TT_KV_POOL_GB=16, gmu=0.6 (or 32, 0.3) — verified locally as v3/v4, not yet in CI |
| Llama-3.2-3B-Instruct | b=16 × 8K | gmu=0.7, no TT_KV_POOL_GB |
| Qwen3-8B | b=16 × 4K | gmu=0.6, no TT_KV_POOL_GB |

The TT_KV_POOL_GB knob is merged and ready to use, but no model_spec entry in `cnn.yaml` opts into it yet — that's intentional for the first merge.

## How to reproduce a run

```bash
docker run -d \
  --name tt-llama-3-1-8b-16k-v3 \
  --device=/dev/tenstorrent/1 \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v volume_id_forge_vllm_plugin-Llama-3.1-8B-Instruct:/home/container_app_user/cache_root \
  -v /tmp/run_uvicorn_patched.sh:/home/container_app_user/app/server/run_uvicorn.sh:ro \
  -p 8005:8000 \
  --user container_app_user \
  -w /home/container_app_user/app \
  -e TT_KV_POOL_GB=16 \
  -e GPU_MEMORY_UTILIZATION=0.6 \
  -e MAX_MODEL_LENGTH=16384 \
  -e MAX_NUM_SEQS=4 \
  -e MODEL=Llama-3.1-8B-Instruct \
  -e MESH_DEVICE=P150 \
  # ... (rest of standard env from inspect)
  "$IMG" \
  /bin/bash -c 'cd ${APP_DIR}/server/ && source venv-worker/bin/activate && source ./run_uvicorn.sh --skip-venv'
```

For OOM signature interpretation: each bank size is **4.27 GB**, total 8 banks = 34.18 GB allocatable. `largest free block` is what matters for single-tensor allocations.
