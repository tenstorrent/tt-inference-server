# Forge vs TTNN: long-context / batched-inference capability gap

> Discussion starter — empirical data from QB2 P150 + the tt-inference-server `model_spec` catalog as of 2026-05-28.

## Why this matters

We have two production paths for serving the same HF LLMs on Tenstorrent silicon:

- **tt-transformers** (TTNN-based, hand-tuned per model)
- **forge / vllm_tt** (tt-xla / pjrt-plugin / tt-mlir compiled, HF-model-generic via vLLM)

Forge is the strategic stack — it's how we plan to scale model coverage without hand-tuning each one. But today, on the **same chip running the same model**, forge's max-context × max-batch capacity is roughly **16–32× lower** than tt-transformers. That gap is not silicon-limited; it is framework-limited. This doc lays out the data + what we think drives it.

## The data (same model, same chip)

Source: `MODEL_SPECS` in `workflows/model_spec.py` (post-YAML refactor on `main`). Each cell is `max_context × max_concurrency = tokens-in-flight`.

| Model | tt-transformers on N150 (12 GB) | tt-transformers on P150 (32 GB) | **forge on P150 (32 GB)** | Gap (same P150 chip) |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 32K × 32 = 1.0M | **65K × 32 = 2.1M** | **4K × 16 = 64K** | **~32×** |
| Llama-3.2-3B-Instruct | 131K × 32 = 4.2M | (uses same wormhole numbers — no p150 ttnn entry yet) | **8K × 16 = 128K** | **~32×** |
| Qwen3-8B | 32K × 32 = 1.0M | (no p150 ttnn entry yet) | **4K × 16 = 64K** | **~16×** (estimate) |
| Falcon3-7B-Instruct | (no ttnn entry) | (no ttnn entry) | **4K × 16 = 64K** | — |

Two especially uncomfortable comparisons:

- **Llama-3.1-8B on N150 (12 GB)** does 1.0M tokens-in-flight via tt-transformers. **Forge on P150 (32 GB) does 64K.** Smaller chip beats bigger chip by ~16× because of the framework choice.
- **Llama-3.2-3B on N150** does 4.2M via tt-transformers. **Forge on P150 does 128K.** Same story.

## Where the gap comes from

Rough decomposition. Each factor is multiplicative; estimates are within an order of magnitude.

| Factor | Forge today | tt-transformers | Approx KV-budget impact |
|---|---|---|---|
| **KV cache dtype / layout** | Standard vLLM PagedAttention, bfloat16, generic tile padding | Custom tile-aligned KV, sometimes packed dtypes (bfp8/bfp4), sharded across L1+DRAM | 2–4× per dim |
| **Trace-capture activation buffers** | tt-mlir captures the full forward graph; scratch sized for `max_seq × max_batch` | Hand-tuned graph with surgical activation reuse | 2–4× of HBM lost to scratch |
| **vLLM scheduler overhead** | PagedAttention block table, prefix cache buffers, swap-out scratch — all live in HBM | Custom continuous-batch loop with minimal scaffolding | ~1–2 GB HBM overhead |
| **Padding / alignment** | XLA buffer manager pads to standard shapes | Tile-friendly head_dim, KV-head packing, no power-of-2 padding waste | 1.5–2× per dim |
| **Device memory reporting** | Was capped to 12 GB regardless of actual chip; fix in flight to report real DRAM (32 GB on P150) | N/A (uses real device memory) | ~3× ceiling on KV pool (largely addressed) |

Multiplied together: a worst-case ~30× HBM-efficiency gap, which matches the empirical "32× tokens-in-flight" gap above.

The bottom row — device memory reporting — is the cheapest and is essentially fixed in the forge wheel chain. The other four rows are the real long-term work.

## Trajectory (rough)

| Horizon | What closes the gap |
|---|---|
| **Now** | Device memory reporting fix — KV pool sized against real 32 GB rather than hardcoded 12 GB. Useful, but only the bottom row of the table above. |
| **Months** | Trace-capture memory optimizations in tt-mlir (reducing scratch buffer sizing). Most leverage of any single change. |
| **Months → quarters** | Forge-side KV cache: tile-aware layout, packed dtypes (bfp8/bfp4), sharding strategy aligned with tt-metal. Closes 2–4×. |
| **Long-term** | Forge ↔ tt-transformers convergence on inference loop, or custom forge inference path that opts out of vLLM scheduler overhead where warranted. |

Even with the memory-reporting fix landed, the realistic forge ceiling on Llama-3.1-8B / P150 in the next month or two is **~16K seq, batch 2–4** — far short of tt-transformers' **65K × 32**.

## What this means for product / planning

Some honest framings:

- **Hardware isn't the bottleneck.** Customer asks like "can we do 16K Llama-3.1-8B on P150?" should be answered "yes, hardware-wise — but only via the tt-transformers path right now; the forge path needs framework work to match."
- **Forge model coverage matters more than per-model capability for the strategic case.** If forge unlocks 20 models that tt-transformers doesn't have, even at 1/30th the tokens-in-flight, that may be the right trade — but only if customers know which path serves which model.
- **The "tt-transformers numbers" are a useful capability ceiling.** When making forge perf-target promises, having the tt-transformers row in the same model_spec.py as a reachable-via-other-means upper bound is helpful framing.
- **Each row in the gap table is independently work-attributable.** Trace-capture optimization, KV dtype packing, scheduler overhead — these belong to different teams. Useful for triaging where to invest.

## Open questions for discussion

1. **Which row gives the biggest ROI right now?** Tracecapture optimizations look like ~2–4× per change, but the work is in tt-mlir, not forge. Is there a coordinated plan?
2. **Should the model_spec.py forge entries reflect the realistic per-config ceiling, or the aspirational target?** Currently they're set to what locally validates as "warmup completes" — which is conservative. Some customers may interpret these as commitments.
3. **Is there a story for "use tt-transformers for the headline LLM list, forge for everything else"?** That's roughly the implicit state today; would worth explicit branding/messaging.
4. **For long-context evals (longbench, RULER, etc.) on forge LLMs:** is the right answer "wait for the framework" or "skip these on forge"? Either way, customer-facing.
5. **What's the right benchmark to track this gap closing over time?** Suggest: tokens-in-flight on a fixed reference model (e.g. Llama-3.1-8B) on a fixed device (P150), reported per forge wheel release.

## Sources / how to reproduce

- Forge numbers: live config in `workflows/model_specs/cnn.yaml` + empirical health checks on QB2 P150.
- tt-transformers numbers: live config in `workflows/model_specs/llm.yaml` (post-YAML refactor); cross-checked against `MODEL_SPECS` map at runtime.
- "Tokens-in-flight" = `max_context × max_num_seqs` per `DeviceModelSpec`.
- Re-derive the comparison table at any time:
  ```python
  from workflows.model_spec import MODEL_SPECS
  for mid, ms in MODEL_SPECS.items():
      if 'Llama-3.1-8B-Instruct' in mid:
          d = ms.device_model_spec
          print(f"{mid:65s} {d.device.name:8s} ctx={d.max_context:7d} cc={d.max_concurrency:3d}")
  ```
