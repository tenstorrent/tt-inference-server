# Qwen3-8B forge release run @ b32 — does it fix the #4131 eval-time blowout?

**Date:** 2026-06-12
**Run:** `run.py --workflow release --model Qwen3-8B --impl forge-vllm-plugin --limit-samples-mode ci-nightly`
against a local server (`--server-url http://127.0.0.1 --service-port 8004`), b32 / 40960-ctx /
bfp8-KV / prefill_chunk_size=2048 / opt=1 / device-sampling / trace.
**Log:** `qwen3_8b_release_b32_cinightly.log`
**Baseline:** #4131, nightly run 27106578855 (batch-4).

## Bottom line

b32 + bfp8-KV + chunked-prefill **helps materially but does NOT fit the 6h CI budget**.
The eval stage still runs ~7h49m and the per-request streaming timeouts (#4131 secondary failure)
persist. The bottleneck remains **per-user decode throughput**, which b32 trades away for aggregate.

## Timing (total run: ~11h52m, exit=1)

| Stage | Wall-clock | Result |
|---|---|---|
| Warmup (trace capture) | ~8 min | ok |
| **Evals** | **~7h49m** (05:23→13:12) | **rc=1** (timeouts) |
| — r1_gpqa_diamond (~40 samples) | ~1h00m | rc=1, 12.5% |
| — mmlu_pro (0.05 limit, 608 req) | ~6h49m | completed, 53.95% |
| Benchmarks | ~3h51m (13:12→17:03) | ✅ completed |
| Reports/acceptance | <1 min | ✅ completed (verdict FAIL) |

`run.py` exit=1 because the evals workflow returned 1 (lm_eval exited non-zero after streaming
timeouts), even though it scored most samples.

## Eval: b32 vs b4 (#4131)

| Eval | b4 baseline | b32 this run | Delta |
|---|---|---|---|
| r1_gpqa_diamond | ~3h48m | ~1h00m | **~3.8× faster** |
| mmlu_pro | cancelled @6h cap, ~14% done | **completed, ~6h49m** | now finishes |
| Total eval | ~13h extrapolated (never finished) | **~7h49m (finished)** | ~1.7× + completes |

- **Win:** mmlu_pro now completes (was cancelled at the cap before).
- **Still broken:** 109 streaming `TimeoutError`s across the eval → partial outputs → rc=1. Long
  12,288-token reasoning generations at low per-user decode (~8 tok/s under b32) blow the 3600s
  per-request timeout. Total eval still > 6h.

## Accuracy (acceptance = FAIL, but status EXPERIMENTAL / no enforced tiers → soft)

| task | score | ratio→gpu_ref | ratio→published | check |
|---|---|---|---|---|
| r1_gpqa_diamond | 12.50 | 0.19 (64.14) | 0.20 (62.00) | FAIL — depressed by timeout/partial outputs |
| mmlu_pro | 53.95 | 0.82 (66.07) | 0.95 (56.73) | FAIL on 0.05 tol, but ~0.95 of published |

## Benchmark throughput sweep (15 pts) — aggregate Output tok/s

| conc | ISL | OSL | tok/s | TTFT ms |
|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 12.4 | 356 |
| 32 | 128 | 128 | 74.0 | 2,999 |
| 1 | 128 | 1024 | 12.0 | 883 |
| 32 | 128 | 1024 | 89.1 | 2,455 |
| 1 | 1024 | 128 | 6.6 | 8,869 |
| 32 | 1024 | 128 | 8.3 | 164,144 |
| 1 | 2048 | 128 | 4.0 | 20,759 |
| 18 | 2048 | 128 | 4.7 | 187,928 |
| 1 | 4096 | 128 | 2.1 | 43,001 |
| 9 | 4096 | 128 | 2.3 | 125,766 |
| 1 | 8192 | 128 | 1.5 | 64,025 |
| 4 | 8192 | 128 | 1.2 | 175,563 |
| 1 | 16384 | 128 | 0.7 | 147,366 |

Notes: concurrency is **capped by KV memory** as ISL grows (32→18→9→4→2), not fixed at 32. The
falling tok/s at high ISL is mostly the metric being prefill-poisoned (128 output tokens ÷
prefill-dominated duration; TTFT climbs to ~188s). Single-user decode ≈ 12 tok/s; 32-way concurrency
only reaches ~74–89 tok/s aggregate (~6–7×, not 32× — memory-bandwidth bound).

## Structured-output benchmark (6 pts, 100 prompts, conc 4, OSL 128) — all ran, no failures

| dataset | so-ratio | tok/s |
|---|---|---|
| json | 1.0 | 53.8 |
| json | 0.0 | 53.9 |
| json-unique | 1.0 | 45.5 |
| json-unique | 0.0 | 45.1 |
| xgrammar_bench | 1.0 | 24.4 |
| xgrammar_bench | 0.0 | 50.0 |

(Note: `xgrammar_bench so=0.0` completed here — the empty-chunks issue from
`FINDINGS_xgrammar_bench_ratio0_empty_chunks` did not fail this run.)

## Chunked-prefill / bfp8-KV validation

Zero `TT_FATAL` / DRAM-bank / OOM errors across the entire ~11.9h run at b32 (incl. 16K/32K-ISL
benchmark points). Knobs confirmed active in server config (`prefill_chunk_size=2048`,
`experimental_kv_cache_dtype=bfp_bf8`). The change works as designed; it is **not** the lever for
the eval slowness.

## Recommendation for #4131

b32 (concurrency) alone is insufficient. Need **higher per-user decode throughput** (tt-xla#5034
forge 1.2.0 ~40% regression is the big lever) AND/OR a **`max_gen_toks` cap** on the reasoning evals
to stop 12k-token generations from blowing the per-request timeout. Concurrency raises aggregate but
lowers per-user, so the long-output reasoning tasks don't benefit proportionally.
