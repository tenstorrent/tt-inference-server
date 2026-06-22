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
  - **It *includes* prefill time.** `Output token throughput = total_output_tokens ÷ benchmark
    duration`, and the benchmark duration is the end-to-end wall (first request sent → last
    response received), which spans each request's prefill/TTFT. So **agg tok/s is NOT a
    decode-only rate** — on the OSL=128 rows the wall is mostly prefill, which is why agg tok/s
    badly understates decode there (e.g. cfg6: 16,384 output toks ÷ 1,239 s wall = ~13 tok/s,
    even though per-user *decode* is ~12–14 tok/s and there are 32 of them). The decode-only rate
    is `1000 / TPOT` per user (see the TPOT decomposition section); aggregate *decode* would be
    `effective_concurrency × 1000/TPOT`, both larger than the prefill-diluted agg tok/s.
- **Per-row timing / "wall s"** in the sweep tables = the `Benchmark duration (s)` reported by
  `vllm bench serve` for that row (logged once per `Running benchmark …: N/M` block). Row-to-row
  wall (incl. setup/teardown) comes from those blocks' timestamps; the per-phase totals come from
  `release_log_summary.py`.

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

> **Update (Stage 4 below):** the decode lever *did* move — the tt-xla #4278 incremental-counts
> fix (+ `repetition_penalty=1.0` + seed-drop) cut mmlu_pro ~55% and eliminated its timeouts. The
> remaining eval blocker (gpqa) turned out to be **KV-oversubscription**, not decode rate — fixed
> by capping concurrency, not by more timeout.

---

## Stage 4 — #4278 incremental-counts fix + rep_penalty=1.0 + seed-drop (decode unblocked)

*Branch `kmabee/forge_llm_chunked_prefill.debug`. Log `local_release_qwen3_8b_p8009_20260620_173102.log`
(2026-06-20). Server: `launch_qwen3_8b.sh` defaults (gmu 0.30), full Qwen3-8B on P150, port 8009;
release flow via `--server-url` (no Docker).*

**Config delta vs Stage 2.** Server/memory knobs are unchanged (gmu 0.30, b32, 40960, bfp8-KV,
chunked prefill, opt1, device sampling, enable_trace). The deltas are all in the *decode/sampling
path*:

| Knob | Stage 2 → Stage 4 |
|---|---|
| tt-xla wheel | rebased uplift → **+ #4278 incremental output-token-count fix** (commit `863947492`): rep-penalty counts folded incrementally (O(Δ)) instead of rescanned per step (O(N²)) → **flat decode** at rep ≥ 1.0 |
| eval `repetition_penalty` | 1.1 → **1.0** on gpqa + mmlu_pro `gen_kwargs` (runner default later moved to 1.0 too) |
| per-request seed | honored (seeded slow-path) → **dropped** for Forge (tt-xla#4539): lm-eval's `seed=1234` no longer forces the ~19 MB-CPU-noise/step path |
| gpqa request timeout | 3600 → **7200** (one-off experiment in this run; since reverted) |

### E2E timing (this run)

| Phase | Wall-clock | start (UTC) | notes |
|---|---|---|---|
| **Total release** | **8h09m** | 17:31 | `run.py` exit=1 (acceptance FAIL) |
| Evals | 4h33m | 17:31 | |
| — r1_gpqa_diamond | 2h07m | | 24 timeouts, 0.225 — **inflated by the 7200s experiment** |
| — mmlu_pro | 2h16m | | **0 timeouts**, 0.6316 |
| Benchmarks | 3h35m | 22:05 | ~13m warmup + 15-run text sweep + 6-run structured |
| Reports | <1m | 01:40 | |

Trace-capture warmups: ~4m47s (eval-stage server check) + ~13m07s (benchmark-stage).

### Stage 2 → Stage 4 comparison

| | Stage 2 (gmu 0.30, rebased) | **Stage 4 (+#4278, rep=1.0, seed-drop)** | Δ |
|---|---|---|---|
| **Total wall-clock** | 9h38m | **8h09m** | **−1h29m (~15%)** |
| Evals stage | 6h00m | **4h33m** | −1h27m |
| — r1_gpqa_diamond | ~57m | 2h07m | +1h10m (artifact of 7200s timeout; at 3600s ≈ ~57m) |
| — mmlu_pro | 5h04m | **2h16m** | **−2h48m (~55%)** |
| Benchmarks | 3h26m | 3h35m | +9m (~flat) |
| Streaming timeouts | 51 (both evals) | **24** (all gpqa; mmlu_pro **0**) | mmlu eliminated |
| gpqa accuracy | 22.5% (r0.35) | 22.5% (r0.35) | flat |
| mmlu_pro accuracy | 61.68% (r0.93) | **63.16% (r~0.95)** | ↑ |

**The win is mmlu_pro: 5h04m → 2h16m with its timeouts eliminated** — the #4278 flat-decode fix
(+ rep=1.0 + seed-drop) paying off on sustained long-output decode, exactly the path the O(N²)
penalty rescans and the seeded-noise transfer used to tax. **gpqa did not improve:** its wall rose
(7200s experiment) with the *same* 24 timeouts and *same* 0.225 score → gpqa is **not** timeout- or
decode-rate-bound but **KV-oversubscription bound** at conc 32 (32 × long-generation KV ≫ the
~139k-token pool → preemption thrash). That points to a concurrency cap, not more timeout.

### Stage 4 benchmark sweep (Qwen3-8B, gmu 0.30)

| cfg | con | ISL | OSL | n | agg tok/s | TTFT ms | wall s | vs Stage 2 |
|----:|----:|------:|-----:|----:|----------:|--------:|-------:|---|
| 1 | 1 | 128 | 128 | 8 | 12.22 | **1,224.8** | 80 | TTFT ↑ (was 356) ⚠ |
| 2 | 32 | 128 | 128 | 256 | 78.18 | 3,410 | 394 | ~flat |
| 3 | 1 | 128 | 1024 | 4 | 12.72 | 1,223 | 63 | TTFT ↑ (was 357) ⚠ |
| 4 | 32 | 128 | 1024 | 128 | **120.68** | 1,744 | 246 | **−33% (was 180.68)** ⚠ |
| 5 | 1 | 1024 | 128 | 4 | 6.79 | 9,071 | 75 | ~flat |
| 6 | 32 | 1024 | 128 | 128 | 12.53 | 106,754 | 1,239 | ~flat |
| 7 | 1 | 2048 | 128 | 4 | 3.98 | 21,777 | 129 | ~flat |
| 8 | 18 | 2048 | 128 | 72 | 4.56 | 180,009 | 1,988 | ~flat |
| 9 | 1 | 4096 | 128 | 4 | 2.21 | 43,876 | 219 | ~flat |
| 10 | 9 | 4096 | 128 | 36 | 2.32 | 128,005 | 1,905 | ~flat |
| 11 | 1 | 8192 | 128 | 2 | 1.63 | 64,495 | 157 | ~flat |
| 12 | 4 | 8192 | 128 | 8 | 1.28 | 208,006 | 793 | ~flat |
| 13 | 1 | 16384 | 128 | 2 | 1.07 | 91,773 | 218 | ~flat |
| 14 | 2 | 16384 | 128 | 4 | 1.21 | 92,551 | 404 | ~flat |
| 15 | 1 | 32768 | 128 | 1 | 3.89 | 4,768 | 33 | (outlier — n=1, TTFT implausibly low; ≈ Stage 2's 4.35) |

**`wall s` = vLLM `Benchmark duration (s)`** per row — the measured request→response window for that
config. *Source:* the release log, where each `Running benchmark Qwen3-8B: N/15` block reports a
`Benchmark duration (s)` line (and the row-start timestamps give the row-to-row wall). **Sum of the
15 rows ≈ 7,943 s ≈ 2h12m.** With per-row setup/teardown the text-sweep wall is ~2h24m; the full
**benchmark *phase* is ~3h35m** = ~27m warmup-bench (the `1/1` pass) + ~2h24m text sweep + ~34m of
the 6-run structured sweep. That ~3.5h phase is the figure quoted elsewhere.
(Note: cfg15's `agg tok/s`/TTFT are an `n=1` outlier — TTFT 4.8 s for a 32,768-token prefill is
implausibly low, likely a prefix-cache/measurement artifact; treat it as suspect, same as Stage 2.)

### Stage 4 structured-output sweep (6 runs) — the other benchmark block

Runs *after* the 15-row text sweep: 3 datasets × structured-output ratio {1.0, 0.0}, all
`max_concurrency=4`, `OSL=128`, `n=100`. `struct out` = `structured_output_ratio` (1.0 = every
request forced to emit schema-valid JSON/grammar; 0.0 = free text on the same prompts).

| # | dataset | struct out | con | OSL | n | agg tok/s | TTFT ms | wall s |
|--:|---|--:|--:|--:|--:|----------:|--------:|-------:|
| 1 | json | 1.0 | 4 | 128 | 100 | 46.04 | 2,133 | 278 |
| 2 | json | 0.0 | 4 | 128 | 100 | 45.68 | 2,133 | 280 |
| 3 | json-unique | 1.0 | 4 | 128 | 100 | 44.96 | 2,160 | 285 |
| 4 | json-unique | 0.0 | 4 | 128 | 100 | 45.14 | 2,135 | 284 |
| 5 | xgrammar_bench | 1.0 | 4 | 128 | 100 | 25.94 | 5,113 | 489 |
| 6 | xgrammar_bench | 0.0 | 4 | 128 | 100 | 36.20 | 1,392 | 345 |

**Sum ≈ 1,960 s ≈ 32.7m** (wall incl. per-run setup ≈ 34m). The `json`/`json-unique` rows are ~flat
(~45 tok/s) whether structured output is on or off; **`xgrammar_bench` at ratio 1.0 (run 5) is the
slow outlier** (489 s, 25.9 tok/s, TTFT 5.1 s) — grammar-constrained decode overhead.

### Benchmark *phase* breakdown (~3h35m total)

Putting the two sweeps + warmup together explains the full benchmark stage:

| sub-phase | wall | note |
|---|--:|---|
| warmup + `1/1` warmup-bench | ~27m | benchmark-stage trace-capture + one warmup run |
| 15-run text sweep | ~2h24m | Σ row durations ~2h12m + per-row overhead |
| 6-run structured sweep | ~34m | grammar/JSON-constrained, con4/n100/OSL128 |
| **total benchmark phase** | **~3h35m** | (= 44% of the 8h09m e2e) |

So the ~3.5h "benchmarks" phase is dominated by the **text sweep (~2/3 of it)**; the structured
sweep is ~16% and warmup ~13%. Trimming the text sweep (Tier-3) is the largest benchmark-time lever.

**Commentary:**
- The sweep is **OSL ≈ 128 dominated → prefill-bound**, so it does *not* surface the #4278 decode
  win (that needs long sustained output, which the evals have and the sweep doesn't). Per-user decode
  rows (cfg 1/5/7/9/11/13) are ~flat vs Stage 2.
- **Two regressions vs Stage 2, both flagged by acceptance:**
  - **cfg1/cfg3 single-stream short-ISL TTFT 356 → ~1224 ms** → fails `ttft_check` (thresholds
    300/60/30 ms). Consistent across both cfgs; likely the small-ISL trace-capture / chunked-prefill
    behavior on the WIP `.debug` branch (see `DEBUG_chunked_prefill_batch_budget.md`).
  - **cfg4 (con32, osl1024) 180.68 → 120.68 tok/s (−33%)** — the Stage-2 headline win partially
    regressed; also branch / chunked-prefill-WIP suspect. Worth a clean re-measure.
- Acceptance still **FAIL** (exit=1): gpqa accuracy (0.225 vs ref 0.641) + benchmark ttft/tput
  checks (ttft 1224.8 ms; tput_user 13.8 vs 18.5/37 thresholds).

### Takeaway / next

1. **#4278 + rep=1.0 + seed-drop is a real, durable decode win** — first lever to move long-output
   eval wall without trading accuracy (mmlu_pro halved, its timeouts gone).
2. **gpqa is the remaining eval blocker, and it's KV-bound, not timeout-bound.** Fix = **Tier-1
   concurrency cap** (gpqa `max_concurrent` 32 → 8 so the working set fits the 139k pool) — applied
   to `eval_config.py`; expected to drop gpqa timeouts to ~0.
3. **Benchmark TTFT / cfg4 regressions** want the chunked-prefill batch-budget fix
   (`DEBUG_chunked_prefill_batch_budget.md`) — improves single-stream TTFT and `ttft_check`.
4. With gpqa at 3600s + conc 8 and mmlu already fast, a clean re-run should land **well under** the
   prior 8–9h, making benchmarks (~3.5h) the largest remaining block → candidate for sweep trimming.

---

## Stage 5 — Tier-1: gpqa concurrency cap 32 → 8 (validated)

*Standalone gpqa re-run, `eval_gpqa_conc8_20260622_154653.log`, against the warm full-model server
(gmu 0.30, port 8009). lm-eval `--concurrency 8`, limit 0.2 (40 docs), `max_gen_toks=12288`,
`repetition_penalty=1.0`, timeout 3600s.*

**Root cause (from the KV-pool analysis):** gpqa generates up to 12,288 tokens; at conc 32 the
`32 × (prompt + gen)` working set vastly exceeds the **~139k-token KV pool** → vLLM
preempts/recomputes → per-user decode collapses → >half the docs miss the 3600s streaming timeout
→ truncated output → wrong answers. Capping concurrency so the active set fits removes the thrash.

| metric | conc 32 (Stage 4) | **conc 8 (Tier-1)** |
|---|---:|---:|
| streaming timeouts | 24 / 40 | **0** |
| exact_match | 0.225 | **0.60** (~0.94 of GPU ref 0.641; ≈ published 0.62) |
| wall (40 docs) | 2h07m (7200s timeout) | **1h20m** (3600s timeout) |

**Three wins from one config line:** timeouts → 0, accuracy **0.225 → 0.60** (acceptance FAIL →
likely PASS), and wall **−40%** (no thrash + no waiting out long timeouts). The gpqa "accuracy
failure" in the release flow was a **timeout/truncation artifact, not a model deficiency.** Landed
as `max_concurrent=8` on the gpqa task in `eval_config.py` (commit on
`kmabee/forge_llm_chunked_prefill.debug`).

**mmlu_pro deliberately left at conc 32:** at conc 32 it shows **0 timeouts** (Stage 4) — its
shorter effective generations keep the working set within the KV pool, so it is *not* thrashing.
Lowering its concurrency would only cost parallelism (308 docs → ~4× more waves at conc 8) and
**slow it down**. The same KV diagnostic gives **opposite prescriptions**: cap gpqa (oversubscribed),
leave mmlu_pro (fits). mmlu_pro's lever is per-user decode (tt-xla#5034) / `max_gen_toks`, not
concurrency.

**Projected e2e with Tier-1:** gpqa ~1h20m + mmlu_pro ~2h16m ≈ **~3.5h evals (0 timeouts)** +
benchmarks ~3.5h ≈ **~7h**, with benchmarks now the dominant block → Tier-3 (sweep trim) is the
next wall-clock lever.
