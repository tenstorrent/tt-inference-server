# [Tracking] Forge TP LLMs on p300x2 (gemma-4-31b-it + Qwen3-32B) end-to-end in CI

Umbrella for getting the two 4-chip tensor-parallel Forge LLMs on **p300x2 / QB2**
(`gemma-4-31b-it`, `Qwen3-32B`) through nightly + release CI at `EXPERIMENTAL`.
Mirrors the single-chip forge umbrella (#3787). Work splits into focused sub-issues
(one PR each) so the initial servable/eval-able support can land without waiting on
the harder benchmark items.

**Image:** `…/tt-media-inference-server-forge:aed6177…_f752cce_79853177306`
(tt-forge `1.3.0.dev20260605003323`). **Config:** batch-16 / 4K, env-driven via
`dev/cnn.yaml`. Branch: `kmabee/gemma4_31b_it_forge`. Detail: `PERF_gemma4_31b_it_forge.md`.

## Current CI status
| Model | serve | evals | benchmarks |
|---|---|---|---|
| gemma-4-31b-it | ✅ | ✅ (ifeval 0.89 local) | ❌ tokenizer crash → **#2** |
| Qwen3-32B | ✅ | ✅ | ⏱️ 6h cap → **#3** |

## Sub-issues (one PR each)
- **#1 — Initial Forge-TP support (serving + evals).** dev-catalog specs, runner (env-var tunable),
  constants (TP topology), forge wheel, ifeval, perf-ref placeholder, nightly matrix. Ships both models
  servable + eval-able at `EXPERIMENTAL`. → `CI_issue1_initial_support.md`. **Can merge first.**
- **#2 — Benchmark client vllm/transformers uplift** (vllm 0.19.1 / transformers 5.5.1). Prereq for
  benchmarks; wide/shared change, qualified vs existing models. → `CI_issue2_benchmarks_vllm_uplift.md`.
- **#3 — Forge-TP benchmark runtime / 6h cap.** conc-1 ~16–50× slow at batch-16 → Qwen 6h timeout.
  → `CI_issue3_benchmark_runtime.md`.

## Follow-ups (checkboxes, not PRs yet)
- [ ] Fill gemma/Qwen eval published+gpu reference scores from first clean nightly (currently None).
- [ ] Replace placeholder perf-reference targets with measured numbers.
- [ ] Decide prod-catalog promotion (currently dev-only, `EXPERIMENTAL`).

## Out of scope
- `EXPERIMENTAL → FUNCTIONAL` perf/accuracy promotion (separate, like #3954).
- Root-cause engine fix for conc-1-at-batch-N inefficiency (tracked in #3 as the proper fix).

## Evidence
Build tt-shield `on-dispatch-build-media-server` #268 ✅ · release gemma #5203 (serve+eval ✅, bench ❌
→#2) · release Qwen #5204 ⏱️ cancelled @6h (→#3) · local p01t05: both serve ✅ at 4K/16.
