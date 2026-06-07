# Qwen3-32B release benchmark exceeds the 6h CI cap (output over-generation, not decode)

Part of the forge-TP-p300x2 umbrella. Qwen3-32B release benchmark
([tt-shield release run 5204](https://github.com/tenstorrent/tt-shield/actions/runs/27054514200))
hit the **6h job cap** and was cancelled. A controlled measurement on a live server shows the
**decode is healthy** — the 6h is from benchmark requests **generating ~8× the requested output
length**, not from slow decode or a concurrency pathology.

## Controlled measurement (gemma server, batch-16/4K, p300x2)
Per-request decode is flat and aggregate scales linearly — **no conc-1 / partial-batch penalty:**

| conc | per-request tok/s | aggregate tok/s |
|---|---|---|
| 1 | 5.04 | 5.04 |
| 2 | 4.94 | 9.87 |
| 4 | 5.03 | 20.1 |
| 8 | 5.06 | 40.5 |
| 16 | 5.05 | 80.8 |

A warm `vllm bench serve` (conc-1) against the same server agrees: **TPOT ≈ 205 ms (~5 tok/s)**.

## Root cause
In CI, **TPOT was healthy too** (~168–216 ms ≈ 5–6 tok/s on every config) — the "0.3 tok/s"
seen earlier was `vllm bench serve`'s *whole-config output-throughput*, not the decode rate.
The 6h came from per-config **durations of 12–38 min × ~13 configs**. From the CI bench summaries,
`output_throughput × duration ÷ requests`:
- conc-1, osl=**128**, n=8 → 5.85 × 1444 / 8 ≈ **~1056 output tokens/request**
- conc-16, osl=128, n=128 → 66.81 × 1909 / 128 ≈ **~996 tokens/request**

→ **Qwen3-32B produced ~1000 tokens/request despite the requested osl=128** (~8×). It's
Qwen-specific: a warm `vllm bench serve` on **gemma** honored osl (≈128 tok/request). The ~1024-ish
value suggests the requested length isn't honored by the Qwen forge server (falls to a ~1024 default)
and/or Qwen3 **reasoning/thinking** output isn't disabled for benchmark requests.

So: 6h ≈ 13 configs × ~1000 tok/req (≈8× over-generation) at a healthy ~5 tok/s — **not** decode,
conc-1, or batch. **gemma is not affected** (it honors osl; its CI failure was only the tokenizer).

## Options
| | Effect | Note |
|---|---|---|
| **Primary: make Qwen honor benchmark osl** — disable Qwen3 thinking for bench requests and/or ensure `max_tokens` is enforced | cuts each config ~8× → whole sweep ~45 min | small, Qwen-specific; the actual root cause |
| Skip conc-1 when `max_concurrency>1` | ~halves the config count | proportional help, not the root cause |
| Lower cnn.yaml `max_context` for these models | fewer isl/osl shapes | proportional help |

## Open verification (before filing)
Confirm ~1000 tok/request on a Qwen server (run a `vllm bench serve` against Qwen, or read a saved
`benchmarks_output/*.json`), and determine whether it's Qwen3 thinking-not-disabled vs `max_tokens`
not enforced. Withdrawn: the earlier "engine partial-batch decode fix" — the controlled sweep shows
decode is healthy.
