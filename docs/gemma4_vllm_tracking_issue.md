# [Tracking] Gemma 4 (12B / 31B) vLLM serving + benchmarking on QB2 (P300X2)

**Labels:** `model:gemma4`, `hardware:qb2`, `vllm`, `tracking`
**Hardware:** QB2 — 2× P300 (P300X2), 4-chip TP mesh as P150x4
**Stack:** tt-metal `a4967d5f39d`, vLLM `9d88cd5`

---

## Motivation

We need Gemma 4 (`google/gemma-4-12b-it` and `google/gemma-4-31b-it`) serving correctly and
benchmarked through the `tt-inference-server` vLLM path on QB2, so we can (a) publish
performance numbers against the tt-metal standalone reference and (b) qualify the config for
nightly/CI. Initial bring-up surfaced correctness, performance, and stability issues that block
hitting the headline targets, so this issue tracks the work end-to-end and the blockers that
need owners.

## Goal

Serve Gemma 4 12B and 31B under vLLM on QB2 with correct output and document/benchmark the
target operating points (batch × ISL), then close the gaps preventing the long-context and
high-batch targets.

## Scope

**In scope**
- vLLM serving of Gemma 4 12B and 31B on QB2 (eager prefill + traced decode).
- `vllm bench serve` benchmarking across the target batch × ISL grid.
- Async scheduling enablement for Gemma4 and measuring its impact.
- Root-causing and documenting the large-ISL device hang.
- Characterizing the KV-cache ceiling and the max servable context/concurrency.
- Defining the path to long-context (128k) and high-batch (32) targets.

**Out of scope (separate issues / follow-ups)**
- The actual fabric/HW fix for the FABRIC_1D NOC read-response deadlock (needs a fabric owner).
- Landing traced prefill for the Gemma4 hybrid-attention path (tracked as its own work item).
- Re-tuning the device KV allocation / re-profiling for 128k & batch-32 (config decision).
- Other model families.

## Target operating points

| Batch | ISL | 12B | 31B |
|---:|---:|---|---|
| 1 | 4096 | required | required |
| 1 | 128k (131072) | required | required |
| 1 | 256k (262144) | required | required |
| 32 | 4096 | required | required |

## Current status (2026-06-25)

- ✅ Correct decode; stable: **12B ≈ 22.6 tok/s/u (44 ms TPOT)**, **31B ≈ 13.7 tok/s/u (73 ms TPOT)**.
- ✅ Captured `B1@4096` and `B1@16384` (12B) / up to `B1@8192` (31B).
- ✅ Async scheduling found disabled, fixed in tt-metal → **~9% faster decode** (branch `dwadhwani/gemma4-vllm-async-decode`).
- 🚫 `B1@128k`, `B1@256k`, `B32@4096` **physically unservable** on current config — hard **23,168-token KV-cache ceiling**.
- 🔴 Large-ISL eager prefill **hangs the device unrecoverably** — triage-confirmed **fabric NOC read-response deadlock** on the ethernet cores.

> Full numbers, logs, and root-cause writeup: see the follow-up comment / `docs/gemma4_vllm_target_benchmarks.md`.

## Acceptance criteria

- [ ] **Correctness:** 12B and 31B produce coherent output under vLLM on QB2 (no gibberish) at the benchmarked points.
- [ ] **Benchmarks:** `B1@4096` captured with TTFT + tok/s/u for both models, recorded against the tt-metal reference.
- [ ] **Async:** async scheduling enabled for Gemma4, verified engaged (`async_scheduling=True`), with a measured decode delta; change landed via PR and shipped in the image.
- [ ] **KV ceiling:** the max servable context (per model) and max concurrency @4096 are documented, with the numeric reason the 128k / batch-32 targets don't fit.
- [ ] **Hang:** the large-ISL device hang has a documented, evidence-backed root cause (triage dump attached) and a concrete mitigation/owner.
- [ ] **Long-context / high-batch:** either `B1@128k` and `B32@4096` are captured **or** a documented plan exists (KV re-allocation + traced prefill) with the changes required to reach them.
- [ ] **Reproducibility:** exact server-launch + benchmark commands documented so anyone can re-run.

## Task breakdown

- [x] Bring up Gemma 4 12B/31B under vLLM on QB2 with correct output
- [x] Benchmark `B1@4096` (both models) + full batch-1 sweep
- [x] Find + fix async scheduling disabled for Gemma4 (tt-metal branch)
- [x] Measure async ON vs sync decode improvement
- [x] Capture max-context point that fits KV (`B1@16384` 12B, `B1@8192` 31B)
- [x] Root-cause the large-ISL device hang (tt-triage dump → fabric deadlock)
- [x] Document the 23,168-token KV ceiling + max servable context/concurrency
- [ ] Open PR for the async-decode fix and rebuild the image
- [ ] File the fabric-deadlock issue and assign a fabric owner
- [ ] File / implement traced prefill for the Gemma4 hybrid path
- [ ] Decide + implement KV re-allocation to unlock >16k context and batch-32
- [ ] Capture `B1@128k` and `B32@4096` (or record the documented plan if still blocked)

## Blockers

1. **KV-cache ceiling (23,168 tokens)** — config decision needed on raising `num_gpu_blocks_override` / shrinking `trace_region_size` and re-profiling.
2. **Fabric NOC read-response deadlock** on large eager prefill — needs a fabric/HW owner; independent of the model.

## References

- Benchmark numbers + root-cause writeup: `docs/gemma4_vllm_target_benchmarks.md`
- Discovery notes / hang timeline: `docs/gemma4_vllm_benchmark_status.md`
- Triage dump (fabric hang proof): `workflow_logs/triage_capture/tt-triage-20260625-195755.log`
- Async-decode fix: tt-metal branch `dwadhwani/gemma4-vllm-async-decode` (`772a878478f`)
