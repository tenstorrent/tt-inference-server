# Forge single-chip P150 LLMs — CI runtime & throughput effort

**Status:** living document. Tracks the ongoing effort to get the forge single-chip
P150 LLM **release flow (evals + benchmarks)** to fit the **6h CI budget** while
holding accuracy near reference. Organized in stages; each stage records the config
delta, results (timing + accuracy + a benchmark throughput sweep), and what the data
says is holding us back.

**Tracking issue:** [#4131](https://github.com/tenstorrent/tt-inference-server/issues/4131)
("Forge reasoning evals exceed the 6h CI budget") · parent [#3787](https://github.com/tenstorrent/tt-inference-server/issues/3787) · related tt-xla#5034 (forge 1.2.0 ~40% decode regression).

---

## The problem

Forge LLM nightlies are **cancelled at the 6-hour CI cap during the eval stage** — not
failing a check, just running out of wall-clock. The suite is dominated by **long-output
reasoning evals** (`r1_gpqa_diamond`, `mmlu_pro`, both at `max_gen_toks=12288`), and the
forge stack can't generate that token volume in time because **both throughput levers are
low**:

1. **Per-user decode throughput** ≈ 8–13 tok/s.
2. **Concurrency** was capped low (batch 4) and, when raised, is **KV-cache-memory bound**.

Aggregate eval throughput ≈ `effective_batch × per-user`, and both factors are small, so a
multi-hour reasoning-token budget can't clear 6h. The durable fix is **more of both** — plus
capping the runaway 12k-token generations that trip the per-request streaming timeout.

### Test setup / methodology

- **Model used for the deep-dive:** `Qwen3-8B` (forge-vllm-plugin), single P150-class chip
  (QB2 Blackhole p300c). Qwen3-8B + Qwen3-4B are the worst offenders (reasoning models,
  long chains).
- **How measured:** `run.py --workflow release --limit-samples-mode ci-nightly` (the *same*
  flow the nightly runs) pointed at a locally-launched `tt-media-server` via
  `--server-url http://127.0.0.1 --service-port 8004`. No Docker; local uvicorn picks up
  source edits immediately. **CI uses the prebuilt forge image → code changes need an image
  rebuild to reach CI.**
- **"agg tok/s"** in the sweep tables = `vllm bench serve` *Output token throughput*
  (aggregate across all concurrent requests, not per-user). In the sweep, concurrency is
  **capped by KV memory as ISL grows** (32 → 18 → 9 → 4 → 2), so the high-ISL rows are both
  prefill-dominated *and* less parallel — that's why agg tok/s falls down the table even
  though decode itself isn't getting slower.

---

## Stage 0 — Baseline (b4 / 16K, bf16 KV, no chunked prefill)

The pre-effort nightly config.

| Knob | Value |
|---|---|
| batch (max_concurrency / max_num_seqs) | **4** |
| max context | **16384** |
| KV cache dtype | bf16 |
| chunked prefill | none |
| optimization_level | 0 |
| sampling | CPU |

**Results** — #4131 evidence, nightly [27106578855](https://github.com/tenstorrent/tt-shield/actions/runs/27106578855) (2026-06-07), Qwen3-8B:

| Phase | Wall-clock |
|---|---|
| Server startup (trace capture) | ~23 min |
| `r1_gpqa_diamond` (40 samples) | **~3h48m** — 13/40 hit the 3000s streaming timeout → partial output → `RuntimeError` |
| `mmlu_pro` (608 req) | only ~14% done (83/608) when **cancelled at 6h cap** |
| **Total eval** | **~13h extrapolated — never finished** |

- Per-user decode ≈ 8–10 tok/s (Qwen3-8B); Falcon3-7B ≈ 12.85.
- **Benchmark sweep table: not captured** at Stage 0 (CI was cancelled in the eval stage;
  the benchmark stage never ran clean). Only the per-user decode figures above survive.

**Takeaway:** at batch-4 the eval token volume is ~13h of wall-clock — it was never going to
fit 6h. Both levers (batch and per-user tok/s) needed to move.

---

## Stage 1 — b32 + 64K + bfp8-KV + chunked-prefill (gmu 0.15)

*Branch `kmabee/forge_llm_chunked_prefill`, **pre** tt-mlir uplift.*

**Config delta vs Stage 0:**

| Knob | Stage 0 → Stage 1 |
|---|---|
| batch | 4 → **32** |
| max context | 16384 → **65536** (Qwen3 capped to its native **40960**) |
| KV cache dtype | bf16 → **bfp8** (`experimental_kv_cache_dtype=bfp_bf8`) |
| chunked prefill | none → **on** (`prefill_chunk_size=2048`, on-device chunked SDPA) |
| optimization_level | 0 → **1** |
| sampling | CPU → **device** |
| enable_trace | — → **on** |
| gpu_memory_utilization | — → **0.15** |
| max_num_batched_tokens | — → context × batch (≈ **1.3M**) |

### Eval result — the #4131 "b32 helps, but doesn't fit 6h" answer

| Eval | b4 baseline (Stage 0) | b32 (Stage 1) |
|---|---|---|
| r1_gpqa_diamond | ~3h48m | **~1h00m (~3.8× faster)** |
| mmlu_pro | cancelled @6h, 14% done | **completed, ~6h49m** |
| Total eval | ~13h (never finished) | **~7h49m (finished)** |

**Full release run** (Qwen3-8B, `qwen3_8b_release_b32_cinightly.log`): total **11h52m**,
`run.py` exit=1.

| Stage | Wall-clock | Accuracy |
|---|---|---|
| Evals | ~7h49m (**rc=1**, 109 streaming timeouts) | gpqa 12.5% (r0.19), mmlu_pro 53.95% (r0.82) |
| — r1_gpqa_diamond | ~1h00m | 12.5% |
| — mmlu_pro | ~6h49m | 53.95% |
| Benchmarks | ~3h51m | ✅ |

### Stage 1 benchmark sweep (Qwen3-8B, gmu 0.15)

| cfg | concurrency | ISL | OSL | n | agg tok/s | TTFT ms |
|----:|----:|------:|-----:|----:|----------:|--------:|
| 1 | 1 | 128 | 128 | 8 | 12.44 | 356 |
| 2 | 32 | 128 | 128 | 256 | 74.03 | 2,999 |
| 3 | 1 | 128 | 1024 | 4 | 12.02 | 883 |
| 4 | 32 | 128 | 1024 | 128 | 89.12 | 2,455 |
| 5 | 1 | 1024 | 128 | 4 | 6.61 | 8,869 |
| 6 | 32 | 1024 | 128 | 128 | 8.25 | 164,144 |
| 7 | 1 | 2048 | 128 | 4 | 4.01 | 20,759 |
| 8 | 18 | 2048 | 128 | 72 | 4.73 | 187,928 |
| 9 | 1 | 4096 | 128 | 4 | 2.14 | 43,001 |
| 10 | 9 | 4096 | 128 | 36 | 2.34 | 125,766 |
| 11 | 1 | 8192 | 128 | 2 | 1.47 | 64,025 |
| 12 | 4 | 8192 | 128 | 8 | 1.21 | 175,563 |
| 13 | 1 | 16384 | 128 | 2 | 0.71 | 147,366 |
| 14 | 2 | 16384 | 128 | 4 | 1.24 | — |

**Commentary:**
- **Win:** mmlu_pro now *completes* (was cancelled at the cap); gpqa ~3.8× faster.
- **Per-user decode ≈ 12 tok/s** (cfg 1); 32-way concurrency reaches only ~74–89 tok/s
  aggregate (cfg 2/4) — **~6–7×, not 32×.** At **gmu 0.15 the concurrency is KV-starved**:
  the KV pool only fits a fraction of 32 sequences, so most of the batch waits.
- Falling agg tok/s down the table is **prefill-poisoning** (each row emits only 128 output
  tokens but TTFT climbs to ~188s at high ISL) + the **memory-capped concurrency** (32→2).
- **Still broken:** evals exit `rc=1` (109 timeouts on the 12k-token generations); total
  11h52m ≫ 6h; mmlu_pro (6h49m) is the long pole.
- **Validates** that chunked-prefill + bfp8-KV are working (zero OOM across the run; long
  contexts fit) — they are *not* the bottleneck. The bottleneck is decode throughput ×
  effective concurrency.

---

## Stage 2 — raise gmu + fp32_dest_acc + rebased tt-xla (tt-mlir uplift)

**Config delta vs Stage 1:**

| Knob | Stage 1 → Stage 2 |
|---|---|
| gpu_memory_utilization | 0.15 → **0.30** (0.35 OOMs at trace-capture — fragmentation) |
| fp32_dest_acc_en | (plugin default, fp32) → **false** (bf16 dest accum → smaller matmul buffers) |
| max_num_batched_tokens | ≈1.3M (ctx×batch) → **65536** (batch × prefill_chunk) — *but see note: capped to 2048 by the plugin either way* |
| tt-xla branch | pre-uplift → **rebased on latest (tt-mlir uplift)** |

> **`fp32_dest_acc_en=false` is the lever** that frees enough DRAM to raise gmu — it was the
> missing knob vs the tt-xla benchmark sweep (which sets it false by default). The
> `max_num_batched_tokens` change is **likely a no-op**: when `prefill_chunk_size` is set (it is),
> the tt-xla plugin caps `max_num_batched_tokens` to `prefill_chunk_size` (2048) for both the
> scheduler budget *and* the prefill activation buffer — so 65536 and the old 1.3M both collapse
> to a 2048-token prefill buffer (see "Chunked prefill: who actually sets it"). The real prefill-
> DRAM reduction comes from **`prefill_chunk_size` (chunking)**, which both Stage 1 and Stage 2
> already had. So Stage 1 was stuck at gmu 0.15 purely for lack of `fp32_dest_acc_en=false`.
> (Caveat: gmu 0.35 OOM'd *before* `fp32_dest_acc_en=false` existed; we only tested 0.30 after
> adding it, so 0.35 may now work too — untested. Confirm via the `--log-level info` relaunch.)

**Full release run** (Qwen3-8B, `qwen3_8b_release_b32_gmu03_cinightly.log`): total **9h38m**,
`run.py` exit=1.

### Stage 1 → Stage 2 comparison

| | Stage 1 (gmu 0.15, pre-uplift) | **Stage 2 (gmu 0.30, fp32=false, rebased)** | Δ |
|---|---|---|---|
| **Total wall-clock** | 11h52m | **9h38m** | **−2h14m (~19%)** |
| Evals stage | 7h49m | **6h00m** | −1h49m |
| — r1_gpqa_diamond | 1h00m | ~57m | ~flat |
| — mmlu_pro | 6h49m | **5h04m** | −1h45m |
| Benchmarks | 3h51m | 3h26m | −25m |
| Streaming timeouts | 109 | **51** | ~halved |
| gpqa accuracy | 12.5% (r0.19) | **22.5% (r0.35)** | ↑ |
| mmlu_pro accuracy | 53.95% (r0.82) | **61.68% (r0.93)** | ↑ |

### Stage 2 benchmark sweep (Qwen3-8B, gmu 0.30)

| cfg | concurrency | ISL | OSL | n | agg tok/s | TTFT ms | vs Stage 1 |
|----:|----:|------:|-----:|----:|----------:|--------:|---|
| 1 | 1 | 128 | 128 | 8 | 13.28 | 356 | +7% |
| 2 | 32 | 128 | 128 | 256 | 77.47 | 3,025 | +5% |
| 3 | 1 | 128 | 1024 | 4 | 13.30 | 357 | +11% |
| 4 | 32 | 128 | 1024 | 128 | **180.68** | 539 | **~2×** (was 89.12) |
| 5 | 1 | 1024 | 128 | 4 | 6.87 | 8,822 | +4% |
| 6 | 32 | 1024 | 128 | 128 | 12.70 | 109,463 | +54% |
| 7 | 1 | 2048 | 128 | 4 | 4.11 | 20,725 | ~flat |
| 8 | 18 | 2048 | 128 | 72 | 4.74 | 187,709 | ~flat |
| 9 | 1 | 4096 | 128 | 4 | 2.18 | 42,633 | ~flat |
| 10 | 9 | 4096 | 128 | 36 | 2.36 | 124,657 | ~flat |
| 11 | 1 | 8192 | 128 | 2 | 1.50 | 62,940 | ~flat |
| 12 | 4 | 8192 | 128 | 8 | 1.23 | 157,085 | ~flat |
| 13 | 1 | 16384 | 128 | 2 | 1.10 | 89,524 | +55% |
| 14 | 2 | 16384 | 128 | 4 | 1.24 | 89,733 | ~flat |
| 15 | 1 | 32768 | 128 | 1 | 4.35 | 1,335 | (outlier — n=1, TTFT implausibly low; treat as suspect) |

**Commentary:**
- The headline: **cfg 4 (con32, osl 1024) doubled, 89 → 181 tok/s.** The speedup is mostly
  **better concurrency utilization** — gmu 0.30 gives the KV-cache headroom for far more of
  the 32 sequences to decode concurrently. At gmu 0.15 they were KV-starved.
- **Single-user decode barely moved** (~12.4 → ~13.3 tok/s, +7–11%). That ~10% is the
  rebased-branch's raw per-user gain — modest.
- Fewer timeouts (109 → 51) → more generations complete → **accuracy up** (mmlu_pro now
  **0.93 of GPU reference**, gpqa nearly doubled).
- **Still:** evals `rc=1` (51 timeouts persist); **9h38m still > 6h**; mmlu_pro (5h04m) still
  the long pole; acceptance still FAIL (but much closer).

---

## Why concurrency doesn't scale linearly (TPOT decomposition)

A natural question — "we run batch 32, why isn't aggregate 32× single-user?" The per-token
metric **TPOT** (mean time per output token) decomposes it cleanly. Governing identity:

> **aggregate tok/s = (effective concurrent decoders) × (per-user tok/s)**, where
> per-user tok/s = `1000 / TPOT_ms`.

Sub-metrics from the Stage 2 sweep (gmu 0.30):

| cfg | con | ISL | OSL | agg tok/s | TPOT ms | per-user tok/s | effective concurrency (agg ÷ per-user) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 128 | 128 | 13.28 | 73 | 13.7 | 1.0 |
| 2 | 32 | 128 | 128 | 77.47 | **385** | 2.6 | ~30 |
| 3 | 1 | 128 | 1024 | 13.30 | 74 | 13.5 | 1.0 |
| 4 | 32 | 128 | 1024 | 180.68 | **113** | 8.9 | ~20 |

The identity reproduces every row to ~2%.

- **cfg2 (con32, short output) → only 5.8× agg.** Not because few requests run concurrently
  (~30 of 32 *do*), but because **per-user TPOT collapses 5.3×** (73 → 385 ms) from prefill
  competing with decode for the per-step token budget. (Mechanism note: vLLM scheduler-level
  chunked prefill is actually **on** — the tt-xla plugin forces `enable_chunked_prefill=True`
  whenever `prefill_chunk_size` is set, overriding the `=False` the server passes; see "Chunked
  prefill: who actually sets it" below. So it's not "prefill blocks decode" — it's the inverse:
  with chunking on, prefill tokens are **co-scheduled into decode steps** (per-step budget ≈
  `prefill_chunk_size`=2048), and short outputs + many requests (n=256) mean a large fraction
  of every step is prefill work, inflating each decode token's latency.) Net: `32 × (1/5.3) ≈ 6×`.
- **cfg4 (con32, longer output) → ~14× agg.** Longer generations amortize prefill over far
  more decode steps, so TPOT degrades only 1.5× (74 → 113 ms). Net much better scaling.
- **Why never 32×:** two independent losses, both would have to vanish. (a) Batched decode is
  never free — even at best TPOT degrades ~1.5× (bigger batch = more KV/attention reads,
  sampling, host overhead per step). (b) Effective concurrency < 32 — requests start/finish
  staggered (ramp-up + drain + early-EOS), so the average simultaneously-decoding count is
  ~20–30, not a steady 32.

**One-liner:** *throughput = concurrency × per-user speed; adding 32 streams doesn't give 32×
because on short outputs the decode batch is constantly interrupted by prefills (per-user speed
−5×), and even on long outputs batched decode is ~1.5× slower per user with only ~20–30 streams
truly overlapping.*

> **Refinement (clean ignore-eos follow-up):** the cfg2→cfg4 trend reads as "longer output =
> better scaling," but that's only true up to a point. A controlled test forcing *full-length*
> output (`ignore_eos`, single lockstep wave — see `test_osl_concurrency.sh`) found a **sweet
> spot**: effective concurrency peaks at **osl≈1024 (~20.7×, 274 tok/s)** and **falls at
> osl=2048 (~12.6×, 161 tok/s)**. Past the sweet spot, decode becomes **KV-length-bound** —
> each step attends over a longer KV cache, and at high batch the aggregate KV-read bandwidth
> saturates (osl=2048/con32 took 3.4× the wall for 2× the tokens). Single-user is nearly
> length-insensitive (−3%); the penalty is a batch effect. **Implication:** the 12k-token
> reasoning evals sit *far* past the sweet spot — the worst regime for concurrency — which is
> an independent, super-linear argument for capping `max_gen_toks` (lever #1 below).

---

## Chunked prefill: who actually sets it (and the `max_num_batched_tokens` cap)

There are **two distinct "chunked prefill" features** that are easy to conflate:

- **tt-xla `prefill_chunk_size`** (additional_config) — *op-level / on-device chunked SDPA*:
  splits the attention math *within* a prefill so the buffer fits. **This is the DRAM lever.**
- **vLLM `enable_chunked_prefill`** (scheduler-level) — splits prefills across steps and mixes
  prefill + decode tokens in one batch.

The server passes `enable_chunked_prefill=False` to `AsyncEngineArgs`
(`tt_model_runners/vllm_runner.py`), **but that value is overridden by the tt-xla plugin.**
In `vllm_tt/platform.py` (`check_and_update_config`), for non-MLA generative models, when
`prefill_chunk_size > 0` (we set 2048) it does:

```python
budget = max(min(scheduler_config.max_num_batched_tokens, prefill_chunk_size), floor)  # min(65536, 2048) = 2048
scheduler_config.enable_chunked_prefill = True       # forces it ON, ignoring the server's False
scheduler_config.max_num_batched_tokens  = budget    # = 2048
```

and `model_runner.py` then sizes the prefill activation / precompile buckets from it:
`prefill_chunk_budget = min(max_num_batched_tokens, max_model_len) = min(2048, 40960) = 2048`.

**Consequences:**
1. **vLLM chunked prefill is ON** for every forge LLM with `prefill_chunk_size` set — nothing is
   missing on the tt-inference-server side; the literal `enable_chunked_prefill=False` is dead
   code for our config (worth deleting for clarity).
2. **`max_num_batched_tokens` above `prefill_chunk_size` is a no-op** — it's capped to 2048 for
   both the per-step budget and the prefill DRAM buffer. Setting it to 65536 vs 1.3M makes no
   memory difference; the prefill-DRAM reduction comes from `prefill_chunk_size` (chunking),
   which bounds the buffer to the chunk instead of `max_model_len` (plugin docstring:
   *"bounding peak prefill DRAM by the chunk size rather than max_model_len"*).
3. **Verify at runtime:** relaunch with `--log-level info` (now the launch-script default) and
   grep the server log for `[TT] Chunked prefill: capping max_num_batched_tokens N -> 2048` and
   the engine-config `enable_chunked_prefill` / `max_num_batched_tokens` values.

---

## What's holding us back (cross-stage)

1. **Per-user decode throughput is the hard floor (~13 tok/s single-user).** Concurrency
   multiplies *aggregate* but the eval workload is **per-user-bound**: each reasoning request
   generates up to 12,288 tokens serially. At ~8–13 tok/s that's ~16–25 min *per request*,
   and the 3000–3600s per-request streaming timeout fires on the long tail → `rc=1` + partial
   outputs that depress accuracy. **Raising batch does not shorten an individual generation.**
2. **mmlu_pro is the long pole** (5h+), driven by 608 requests × up to 12k reasoning tokens.
3. **Concurrency is KV-memory-bound** at high ISL (32 → 2). gmu tuning is the multiplier and
   has real headroom (0.15 → 0.30 nearly doubled cfg-4 aggregate), but 0.35 OOMs.

## Levers / suggestions (data-driven, roughly by impact)

1. **Cap `max_gen_toks` on the reasoning evals** (currently 12288 for gpqa & mmlu_pro). This
   directly kills the long-tail generations that trip the timeout and dominate wall-clock —
   the single biggest CI-time lever available *today*, at a known accuracy/length trade-off.
2. **Recover per-user decode throughput** (tt-xla#5034, the forge 1.2.0 ~40% regression). The
   rebased branch recovered only ~10%. This is the most *durable* lever — it shortens every
   generation directly and reduces timeouts.
3. **Stop the timeouts** so evals exit clean (no `rc=1`, no partial-output accuracy hits):
   either #1/#2 above, or raise `request_processing_timeout` beyond 3000s (#3950 already went
   1000→3000s; insufficient at b32 because higher batch lowers per-user tok/s).
4. **Keep tuning gmu / concurrency** — the multiplier. 0.30 is the validated known-good for
   Qwen3-8B@40960; 0.35 OOMs. Worth a per-model gmu sweep.
5. **Stopgap:** lower the ci-nightly sample fraction for forge reasoning evals (gpqa is
   `limit=0.2`, mmlu_pro `0.05`).

## Config reference (per stage, Qwen3-8B)

| Knob | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| batch / max_num_seqs | 4 | 32 | 32 |
| max context | 16384 | 40960 | 40960 |
| KV dtype | bf16 | bfp8 | bfp8 |
| prefill_chunk_size | — | 2048 | 2048 |
| optimization_level | 0 | 1 | 1 |
| sampling | CPU | device | device |
| enable_trace | off | on | on |
| gpu_memory_utilization | — | 0.15 | 0.30 |
| fp32_dest_acc_en | (fp32) | (fp32) | false |
| max_num_batched_tokens | — | ~1.3M | 65536 |
| tt-xla | — | pre-uplift | rebased (tt-mlir uplift) |
| **Total release wall-clock** | — (cancelled @6h) | **11h52m** | **9h38m** |

## Reproduce

```bash
# 1. server (any free chip; gmu 0.30 known-good for Qwen3-8B)
cd ~/tt-inference-server/tt-media-server
TT_VISIBLE_DEVICES=<chip> GPU_MEMORY_UTILIZATION=0.30 ./launch_qwen3_8b.sh 2>&1 | tee server.log

# 2. release flow against it (same as nightly)
cd ~/tt-inference-server
VLLM_API_KEY=your-secret-key OPENAI_API_KEY=your-secret-key MODEL_SPECS_ENV=dev \
python run.py --model Qwen3-8B --device p150 --impl forge-vllm-plugin \
  --workflow release --server-url http://127.0.0.1 --service-port 8004 \
  --limit-samples-mode ci-nightly
```

## Open items / next stages

- **Stage 3 (candidate):** `max_gen_toks` cap experiment on gpqa/mmlu_pro — measure CI-time vs
  accuracy trade-off. Expected to be the move that gets under 6h.
- Per-model gmu sweep (0.30 → ? before OOM) for each of the 5 single-chip forge LLMs.
- Re-run the other 4 models (Llama-3.1-8B, Llama-3.2-3B, Falcon3-7B, Qwen3-4B) through the
  same staged comparison; numbers above are Qwen3-8B only.
- Commit status: chunked-prefill/bfp8 + context caps + max_num_batched_tokens/gmu are committed
  on `kmabee/forge_llm_chunked_prefill`; `fp32_dest_acc_en` + final gmu default pending.

---

## Stage 3 attempt — gmu 0.35 (FAILED, not viable) + large-ISL compile findings

**gmu 0.35 is not usable for the server workload.** It warms up (single-prompt) but **OOM-crashes
the EngineCore under real b32 eval load**: the lazy batch-32 / ctx-1024 **decode trace capture**
(~1.5 GiB scratch) hits at runtime with only ~362 MiB/bank free at 0.35 → `TT_FATAL: Out of Memory`
→ EngineCore dead → whole release run failed in ~16 min. At **gmu 0.30** the same captures fit, so
the 9h38m Stage-2 run completed. **gmu 0.30 is the ceiling.** `min_context_len` 32→128 (which
dropped 2 small prefill buckets and let 0.35 *warm up*) did **not** give enough runtime headroom.

Root cause of the OOM is *lazy trace/graph capture*, not a config knob: decode traces are captured
per `(batch, context-bucket)` on first runtime use; warmup (1 short seq) only captures the
batch-1/short trace, so the batch-32/large-context captures happen mid-eval and OOM at 0.35.

Confirmed NOT the cause (ruled out by diffing server vs tt-xla standalone, both at gmu 0.35):
vLLM engine config (byte-identical), `VLLM_ENABLE_V1_MULTIPROCESSING` (both pass), `additional_config`
(identical), `TT_KV_POOL_GB` (inert on this branch — worker auto-detects DRAM).

**4-row re-bench (gmu0.35/minctx128) showed no short-context gain** — expected: those rows are
KV-not-limited, and per-user decode (TPOT ~73 ms single-user) is unchanged. gmu/concurrency only
help long-context fit, not short-context tok/s.

### Large-ISL (16K) recompile + warmup gap (see HANDOFF doc)

A real ISL=16384 request triggers a **runtime recompile** of the chunked-prefill cached-prefix /
page-table-gather graphs that warmup never precompiled → minutes of compile → client 300 s timeout
→ abort → `AscendScheduler` `FINISHED_ABORTED` crash (separate bug; gist + tt-xla issue filed).
Standalone tt-xla runs the *same* ISL in warmup so it's fast (1.1 s prefill TTFT). Fix: precompile
the chunked path at max ISL (dummy, at production batch). Details + repro in
`HANDOFF_largeisl_compile_investigation.md`.

### Where the eval-time lever actually is (unchanged conclusion)
Per-user decode throughput (~8–13 tok/s) + the 12k-token reasoning generations remain the 6h-budget
bottleneck. gmu is maxed at 0.30. Durable levers: recover per-user decode (tt-xla#5034) and/or cap
`max_gen_toks` on the reasoning evals.
