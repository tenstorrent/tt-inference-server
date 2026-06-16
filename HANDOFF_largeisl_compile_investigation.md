# HANDOFF — Investigate the large-ISL runtime recompile (warmup precompile gap)

**Date:** 2026-06-16 · **Owner handoff** · Stack: tt-inference-server (forge vLLM plugin) on tt-xla,
single Blackhole P150-class chip (device 0). Branches: tt-inference-server
`kmabee/forge_llm_chunked_prefill`; tt-xla `kmabee/chunked_prefill_isue_4986_explore.rebase`.

Shareable summary (problem + fixes): https://gist.github.com/kmabeeTT/119d0a738832300e431ff271d237ad3e

## TL;DR

A real large-ISL request (e.g. **ISL=16384**) on the forge server triggers a **runtime recompile**
of chunked-prefill graphs that warmup never precompiled → minutes of compile → client 300 s timeout
→ abort → server crash. The **same shape runs fine standalone in tt-xla (~1.1 s prefill TTFT)**
because the benchmark warms up with the exact production shape. **Fix: precompile the chunked-prefill
path at the max benchmark ISL** (dummy, at production batch). A separate `AscendScheduler` bug turns
the timeout into a full server crash.

## What was found (root cause)

`_precompile_backbone` (`vllm_tt/model_runner.py`) precompiles prefill **query buckets** up to
`prefill_chunk_budget = min(max_num_batched_tokens, max_model_len)`. With chunked prefill that budget
is **capped to `prefill_chunk_size` (2048)** → only `[128,256,512,1024,2048]` are precompiled.

A prompt > 2048 is split into 2048-token chunks. The **cached-prefix chunked-SDPA / page-table-gather
graphs** for chunks 2..N (attending over the *growing* cached KV) are **not enumerated by the
precompile** → they JIT at runtime on the first long-prompt request.

Evidence (decisive): in the tt-xla standalone 16K run, the engine init (`_precompile_backbone`)
emitted **126** MLIR compiles, and the **same-shape warmup iteration emitted +18 more** — those 18
are the chunked-prefill graphs the precompile misses. The server's crash log shows it recompiling the
*same* shapes at runtime: `tensor<32x2048x4096>` (2048-token prefill chunk, batch-32),
`tensor<32x1x4096>` (decode), and `gather.34` (page-table gather).

### Why standalone tt-xla is fine but the server isn't
- **Standalone benchmark:** `benchmark_vllm` runs `warmup_iterations=1` with the **exact production
  shape** → the 18 chunked-prefill graphs compile during warmup → measured request is fully cached.
- **tt-inference-server:** `VLLMForgeRunner.warmup` runs a **single short prompt** (`"Hello, it's me"`,
  10 tokens) → never exercises the chunked path → first real long request recompiles live.

### Scope / what does NOT recompile
- ISL ≤ 2048 never recompiles — prompts pad up to a precompiled query bucket (128→128, 1050→2048…).
- Only ISL > 2048 (the chunked path) recompiles, and **once per new shape** (not per request).
- It is a **chunked-prefill side-effect**: *without* chunked prefill, `prefill_chunk_budget =
  max_model_len`, so the full prefill bucket is precompiled at init (no runtime recompile — but then
  you hit the giant single-shot prefill buffer / OOM that chunked prefill was added to avoid).
- **Decode** is a separate lazy-capture axis: decode traces are captured per `(batch, context-bucket)`
  on first use (this is what OOM-crashed the gmu-0.35 run as sequences grew past ctx-1024).

## How we tested ISL=16K in tt-xla (reproduce this)

The stock benchmark sets ISL = tokenized length of a short `DEFAULT_PROMPT`, so the qual-sweep's
"seq len" was only `max_model_len` (capacity) with a ~17-token prompt — **not a real large input.**
To test a true 16K input we added an exact-ISL hook (3 small edits — already in the tt-xla worktree):

- `tests/benchmark/benchmarks/vllm_benchmark.py`: added `isl: int = 0` to `VLLMBenchmarkConfig`;
  `benchmark_vllm` builds `{"prompt_token_ids": [100]*isl}` when `isl>0`.
- `tests/benchmark/test_vllm_benchmarks.py`: reads `TT_BENCHMARK_ISL` → passes to the config.

Run script: **`~/tt-xla/run_qwen3_8b_isl16k.sh`** — Qwen3-8B, `TT_BENCHMARK_ISL=16384`,
max_model_len=40960, gmu=0.30, chunked prefill 2048, opt=1, trace, bfp8 KV, `TTXLA_LOGGER_LEVEL=DEBUG`,
device 0, **batch=1** (b32×16K won't fit KV; b1 matches the server's `--concurrent 1` test). Reproduce:
```bash
cd ~/tt-xla && bash run_qwen3_8b_isl16k.sh        # or: BATCH=4 ./run_qwen3_8b_isl16k.sh
```
**Result:** PASS, **TTFT 1124 ms**, decode 17 tok/s, **0 OOM**, init 378 s. 144 MLIR dumps total
(126 init + 18 in the warmup iteration). Logs: `~/tt-xla/qwen3_8b_isl16k_<ts>/`.

The forge server hitting the same 16K (`test_all_llm_servers.sh --concurrent 1 --isl 16384`) never
finished in 300 s and crashed — log: `tt-media-server/run_qwen3_8b_rebase_0p30_debug_min_128.log`.

IR comparison artifact (server-vs-standalone graph diffs): `IR_DIFF_server_vs_standalone.md`.

## Suggested fix (to investigate / implement)

**Precompile the chunked-prefill path at the max benchmark ISL (e.g. 32K).** One pass through the
chunked path at 32K compiles the full cached-context ladder (`{0, 2K, …, 30K}`), a superset of every
shorter ISL → covers all ISL ≤ 32K. Requirements:
1. **Dummy-precompile path, not a real generation** — a real b32×32K generate OOMs (KV doesn't fit);
   `_dummy_run` doesn't allocate KV, so extend `_precompile_backbone` to step the chunked-prefill
   dummy through the cached-context range (it currently only dummies the max-blocks state once).
2. **At the production batch (b32)** — graphs are batch-keyed (`32x2048x4096`); a b1 warmup won't cover b32.
3. **Also cover decode traces** at the large context buckets (separate `(batch, ctx-bucket)` axis).

Open questions for the investigator:
- Exact keying of the cached-prefix graphs (per cached-context length? per chunk index? padded
  bucket?) — determines how many distinct graphs a 32K pass compiles. Confirm via a `TT_BENCHMARK_ISL`
  sweep (e.g. 4K/8K/16K/32K) counting MLIR dumps per run.
- Whether the fix belongs in `_precompile_backbone` (tt-xla plugin, benefits all consumers) or in
  `VLLMForgeRunner.warmup` (tt-inference-server, warm representative ISLs). Plugin-side is cleaner.

## Separate but related: AscendScheduler abort crash
Aborting a request **mid-chunked-prefill** crashes the EngineCore
(`RuntimeError: Invalid request status: FINISHED_ABORTED`, `ascend_scheduler.py:311`) — base
`finish_requests()` doesn't purge it from `skipped_waiting`. This is what turns the slow recompile
into a full outage. Full writeup + CPU-only deterministic repro:
`~/tt-xla/ISSUE_finished_aborted_scheduler_crash.md` (file as a tt-xla issue).

## Key files / artifacts
- tt-xla benchmark edits + `run_qwen3_8b_isl16k.sh` (ISL hook).
- `~/tt-inference-server/FORGE_LLM_CI_PERF_EFFORT.md` (full stage-by-stage perf log).
- `~/tt-inference-server/IR_DIFF_server_vs_standalone.md`.
- `~/tt-xla/ISSUE_finished_aborted_scheduler_crash.md`.
- gist: https://gist.github.com/kmabeeTT/119d0a738832300e431ff271d237ad3e
- server crash log: `tt-media-server/run_qwen3_8b_rebase_0p30_debug_min_128.log`.
