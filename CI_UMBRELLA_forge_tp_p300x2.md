# [Tracking] Forge TP LLMs on p300x2 (gemma-4-31b-it + Qwen3-32B) end-to-end in CI

Umbrella for getting the two 4-chip tensor-parallel Forge LLMs on **p300x2 / QB2** (`gemma-4-31b-it`, `Qwen3-32B`) through nightly + release CI at `EXPERIMENTAL`. Mirrors the single-chip forge umbrella (tracking issue 3787). Work splits into focused sub-issues so the initial servable/eval-able support can land without waiting on the harder benchmark items.

**Image:** `…/tt-media-inference-server-forge:aed6177…_f752cce_79853177306` (tt-forge `1.3.0.dev20260605003323`). **Config:** batch-16 / 4K, env-driven via `dev/cnn.yaml`. Branch: `kmabee/gemma4_31b_it_forge`. Detail: `PERF_gemma4_31b_it_forge.md`.

## Current CI status
| Model | serve | evals | benchmarks |
|---|---|---|---|
| gemma-4-31b-it | ✅ | ✅ (ifeval 0.89 local) | ❌ tokenizer crash → see Benchmark-uplift |
| Qwen3-32B | ✅ | ✅ | ⏱️ 6h cap → see Benchmark-runtime |

## Sub-issues / breakdown
Three independent work items. Two are tiny/self-contained; one (uplift) is the heavy, cross-cutting pole.

| Item | Sub-issue | Scope | Files | Risk / blast radius | Depends on |
|---|---|---|---|---|---|
| **A — Benchmark output cap** | _TBD_ | Force `max_tokens=osl` in text bench-serve requests so output is bounded (fixes Qwen 6h cap; general correctness) | `benchmarking/run_benchmarks.py` (1 hunk) | Tiny; only changes behavior for models that were over-generating | none |
| **B — Initial Forge-TP support** | _TBD_ | Both models servable + eval-able at `EXPERIMENTAL`: dev-catalog specs, env-var-tunable runners, TP-topology constants, forge wheel, ifeval (downsampled), perf-ref placeholder, nightly matrix | `dev/cnn.yaml`, `tt_model_runners/*`, `constants.py`, `forge_runners/requirements.txt`, `eval_config.py`, `model_performance_reference.json`, `models-ci-config.json` | Additive, new models only | none (benchmark step stays red until A+C) |
| **C — Benchmark client uplift** | _TBD_ | vllm `0.13.0 → 0.19.1`, transformers `→5.5.1` (gemma-4 tokenizer prereq) | `requirements/benchmarks-vllm.txt`, `workflows/workflow_venvs.py` | **Wide** — shared client venv for *all* benchmarked LLMs; needs cross-model qualification | none (longest pole) |

All three are independent and can land in any order — **B can land immediately** (serving + evals go green; the benchmark step stays known-red until A + C land), while **C** is the long pole (broad cross-model re-qualification).

## Follow-ups
- [ ] Fill gemma/Qwen eval published+gpu reference scores from first clean nightly (currently None).
- [ ] Replace placeholder perf-reference targets with measured numbers.
- [ ] Decide prod-catalog promotion (currently dev-only, `EXPERIMENTAL`).

## Out of scope
- `EXPERIMENTAL → FUNCTIONAL` perf/accuracy promotion (separate tracking effort).

## Evidence
[tt-shield build run 268](https://github.com/tenstorrent/tt-shield/actions/runs/27053465897) ✅ · [gemma release run 5203](https://github.com/tenstorrent/tt-shield/actions/runs/27054513861) (serve+eval ✅, benchmark ❌ → Benchmark-uplift) · [Qwen release run 5204](https://github.com/tenstorrent/tt-shield/actions/runs/27054514200) ⏱️ cancelled at 6h (→ Benchmark-runtime) · local p01t05: both serve ✅ at 4K/16.
