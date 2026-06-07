# [Tracking] Forge TP LLMs on p300x2 (gemma-4-31b-it + Qwen3-32B) end-to-end in CI

Umbrella for getting the two 4-chip tensor-parallel Forge LLMs on **p300x2 / QB2**
(`gemma-4-31b-it`, `Qwen3-32B`) through nightly + release CI at `EXPERIMENTAL`.
Mirrors the single-chip forge umbrella (tracking issue 3787). Work splits into focused
sub-issues (one PR each) so the initial servable/eval-able support can land without
waiting on the harder benchmark items.

**Image:** `…/tt-media-inference-server-forge:aed6177…_f752cce_79853177306`
(tt-forge `1.3.0.dev20260605003323`). **Config:** batch-16 / 4K, env-driven via
`dev/cnn.yaml`. Branch: `kmabee/gemma4_31b_it_forge`. Detail: `PERF_gemma4_31b_it_forge.md`.

## Current CI status
| Model | serve | evals | benchmarks |
|---|---|---|---|
| gemma-4-31b-it | ✅ | ✅ (ifeval 0.89 local) | ❌ tokenizer crash → see Benchmark-uplift |
| Qwen3-32B | ✅ | ✅ | ⏱️ 6h cap → see Benchmark-runtime |

## Sub-issues (one PR each)
- **Initial support — serving + evals** (`CI_initial_support.md`): dev-catalog specs, runner
  (env-var tunable), constants (TP topology), forge wheel, ifeval, perf-ref placeholder, nightly
  matrix. Ships both models servable + eval-able at `EXPERIMENTAL`. **Can merge first.**
- **Benchmark client vllm/transformers uplift** (`CI_benchmarks_vllm_uplift.md`): vllm 0.19.1 /
  transformers 5.5.1. Prereq for benchmarks; wide/shared change, qualified vs existing models.
- **Benchmark runtime / 6h cap** (`CI_benchmark_runtime.md`): Qwen3-32B benchmark over-generates
  output (~1000 tok/req vs requested 128) → 6h timeout. Decode is healthy (controlled measurement);
  gemma honors the length and is not affected.

## Follow-ups (checkboxes, not PRs yet)
- [ ] Fill gemma/Qwen eval published+gpu reference scores from first clean nightly (currently None).
- [ ] Replace placeholder perf-reference targets with measured numbers.
- [ ] Decide prod-catalog promotion (currently dev-only, `EXPERIMENTAL`).

## Out of scope
- `EXPERIMENTAL → FUNCTIONAL` perf/accuracy promotion (separate tracking effort).

## Evidence
[tt-shield build run 268](https://github.com/tenstorrent/tt-shield/actions/runs/27053465897) ✅ · [gemma release run 5203](https://github.com/tenstorrent/tt-shield/actions/runs/27054513861) (serve+eval ✅, benchmark ❌ → Benchmark-uplift) ·
[Qwen release run 5204](https://github.com/tenstorrent/tt-shield/actions/runs/27054514200) ⏱️ cancelled at 6h (→ Benchmark-runtime) · local p01t05: both serve ✅ at 4K/16.
