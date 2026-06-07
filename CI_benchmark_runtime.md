# Forge-TP release benchmark exceeds the 6h CI cap (conc-1 at batch-N)

Part of the forge-TP-p300x2 umbrella. Qwen3-32B release benchmark (tt-shield release run 5204) hit the
**6h job cap** and was cancelled; gemma will too once its tokenizer (the Benchmark-uplift item) is
fixed. Goal: a **small interim** so CI doesn't hang QB2 for 6h, plus a pointer to the proper fix.

## Root cause
The release sweep (`benchmark_config._expand_text_sweep_params`) runs **both conc-1 and
conc-`max_concurrency`** for every isl/osl pair. At the batch-16 config, per-request rate is wildly
concurrency-dependent (a single request still runs the full batch-16 graph):
- **conc-16:** ~26s / 128 tok per request, 16 in parallel → a 128/128 config ≈ 3–4 min.
- **conc-1:** ~383s / 128 tok (~0.3 tok/s) → 128/128 ≈ **50 min**; 128/**1024** ≈ **~3.8 h**.

→ The **conc-1 runs are essentially the entire 6h**, and they're unrepresentative (we serve at 16).

## Options
| | Type | Effect | Effort / Risk |
|---|---|---|---|
| **A. Skip conc-1 when `max_concurrency>1`** (drop the `1` in the concurrency list) | workaround | removes ~all the 6h; keeps representative conc-16 coverage | ~3 lines in `benchmark_config.py`; shared file but only gemma/Qwen have batch>1 today → tiny blast radius |
| B. Lower cnn.yaml `max_context` for these models | workaround | drops long-osl (osl=1024) pairs; per-model, no shared code | small; under-reports context; doesn't fix conc-1 alone |
| C. Evals-only in CI for now | workaround | zero benchmark cost | needs per-model workflow knob; no perf coverage |
| D. Raise tt-shield timeout to 8h | ✗ | avoids the cancel only | **rejected** — still hangs QB2 8h |
| E. Engine: don't run the full batch-N graph for a single request | **proper fix** | removes the pathology | large (tt-xla/vllm_tt), long-term |
| F. Per-model benchmark profiles in the catalog | proper fix | clean per-model control | medium |

## Recommendation
**A** (skip conc-1 for `max_concurrency>1`), optionally + **B** if conc-16-only is still tight. Small,
low-risk, keeps the meaningful benchmark. Track **E** as the real fix.

## Note
Independent of the Benchmark-uplift item: that lets the benchmark *start* (tokenizer); this keeps it
*bounded* (runtime).
