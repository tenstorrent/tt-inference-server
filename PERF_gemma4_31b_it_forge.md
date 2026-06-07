# gemma-4-31b-it (Forge, tensor-parallel) — serving + performance findings

Source-of-truth notes for opening a ticket to add **gemma-4-31b-it** to
tt-inference-server + tt-shield CI. Captures the end-to-end bring-up and a
flag sweep that recovers the tt-xla benchmark's ~9 tok/s.

> **NOTE (2026-06-07):** the **Current status** section immediately below is
> authoritative. The bullets/sections after it are the original 2026-06-05/06
> bring-up snapshot (512 seq / conc-1 on the 1.2.0 image), kept for history.

## Current status — 2026-06-07 (gemma-4-31b-it + Qwen3-32B, forge TP on p300x2)

Both models are wired into the **dev** catalog (`workflows/model_specs/dev/cnn.yaml`)
and run via the release flow. CI (nightly + on-dispatch) runs `run.py --dev-mode`
**unconditionally** (tt-shield `workflow_run-tests-with-inference-server.yml:327`),
so it reads **dev**, not prod. Serving dims are **env-driven** (cnn.yaml `env_vars`,
not `constants.py`) — mirrors the single-chip forge LLMs; `constants.py` holds only
the TP mesh topology, and `model` comes from the `MODEL` env.

**Docker image (both models):**
`ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:aed6177ab424aa93f447118aac5a7b8ab1cafdbc_f752cce_79853177306`
- **tt-forge wheel: `1.3.0.dev20260605003323`** (built 2026-06-05; same build as
  `pjrt-plugin-tt` / `vllm_tt`). Built by tt-shield `on-dispatch-build-media-server`
  run #268 from branch `kmabee/gemma4_31b_it_forge`@`f752cce2`
  (`forge-media-inference-server`). Tag = `<tt-metal aed6177>_<inf-server f752cce>_<ci 79853177306>`.
- Predecessors: `e03b231b…_46a7c96_79778041239` (1.3.0, no branch changes);
  `97aea20e…_fd9296b_79608294867` (`tt-forge 1.2.0.dev20260530002932` — original
  bring-up + the perf sweeps documented below).

**Serving config (both; env-driven via `dev/cnn.yaml` `env_vars`):** `MAX_MODEL_LENGTH=4096`,
`MAX_NUM_SEQS=16` (→ auto `max_num_batched_tokens=65536`), `GPU_MEMORY_UTILIZATION=0.9`,
`OPTIMIZATION_LEVEL=0`, `ENABLE_TRACE=true`, `CPU_SAMPLING=false`, `bfp_bf8` weights,
`TT_MESH_GRAPH_DESC_PATH`=p300_x2. → batch-16 / 4K on the 4-chip `(1,4)` TP mesh.

### gemma-4-31b-it  (`id_forge-vllm-plugin_gemma-4-31b-it_p300x2`, weights `google/gemma-4-31B-it`, EXPERIMENTAL)
| | Result |
|---|---|
| Local serve (new image, p01t05) | ✅ warms up + serves at 4K/16 |
| Local smoke eval | ✅ `ifeval` prompt_level_strict_acc **0.89** |
| Local smoke benchmark | ❌ `vllm bench serve` client crash — `AttributeError: 'list' object has no attribute 'keys'` (`transformers/tokenization_utils_base.py:1210`, gemma-4 tokenizer `extra_special_tokens` is a list) |
| CI release [#5203] | ❌ failed @62min — serve ✓, eval ✓, all 14 benchmark runs ❌ (same tokenizer crash) |
| Evals registered | `ifeval`, downsampled CI_NIGHTLY 0.1 / SMOKE 0.01; published/gpu refs TBD (first TT user) |
| Perf reference | p300x2 128/128 conc1 placeholder (borrowed Qwen3-32B targets) |

**Next / fixes (gemma):** (1) **[BLOCKER] benchmark-client tokenizer bug** — `extra_special_tokens`
list; likely a `transformers` pin/patch in the benchmark venv. (2) after the client works, re-check
benchmark runtime (conc-1 slowness — see Qwen). (3) fill eval reference scores from first clean nightly.

### Qwen3-32B  (`id_forge-vllm-plugin_Qwen3-32B_p300x2`, weights `Qwen/Qwen3-32B`, EXPERIMENTAL)
*(separate vLLM Qwen3-32B specs for galaxy/t3k remain in `prod/llm.yaml` + `dev/llm.yaml`, untouched.)*

| | Result |
|---|---|
| Local serve (new image) | ✅ warms up + serves at 4K/16 |
| Local smoke eval | ✅ `r1_aime24` ran (0.33 on ~3 smoke samples — functional, not a real accuracy figure) |
| Local smoke benchmark | ✅ **functional, no crash** — but conc-1 ≈ **0.3 tok/s** |
| CI release [#5204] | ⏱️ cancelled @ 6h job cap — serve ✓, eval ✓, benchmark did **not** finish |

**Root cause of the 6h:** conc-1 benchmark runs are ~16× slower than aggregate at a batch-16 config
(a single request still runs the batch-16 graph: ~0.3 tok/s vs ~9 aggregate). The release sweep runs
**conc=1 for every isl/osl pair**, so those dominate → full matrix > 6h.

**Next / fixes (Qwen):** no clean in-branch lever — `timeout-minutes` is **not** dispatch-exposable
(`on-dispatch.yml` never passes it; reusable-workflow default 360). Options: (a) tt-shield workflow
change to expose/raise `timeout-minutes` per model; (b) `benchmark_config.py`: **skip conc-1 for
forge-TP** (kills the slowest runs + ~halves the matrix — shared code, codeowner review); (c)
re-dispatch with `workflow=evals` for a green eval run while benchmarks are handled separately.
Also decide: keep batch-16/4K (from the gemma sweep) vs revert Qwen to 512/conc-1 (never swept on Qwen).

### Plumbing verified end-to-end (both)
`--dev-mode` → dev spec → env_vars (`4096/16/0.9/trace`) flow into the container → new image (reused via
the `on-dispatch` `docker-image` input, skipping rebuild). Local default `run.py` (no `--dev-mode`) uses
**prod**, so prod must not be relied on for CI. `[#5203]: https://github.com/tenstorrent/tt-shield/actions/runs/27054513861`
`[#5204]: https://github.com/tenstorrent/tt-shield/actions/runs/27054514200`

---

- **Model:** `google/gemma-4-31B-it` (loaded internally; host spec uses lowercase
  `gemma-4-31b-it` to match `ModelNames.GEMMA_4_31B_IT`).
- **Device:** QB2 = 2× P300 = `p300x2`, 4 chips, `(1,4)` 1D tensor-parallel mesh.
- **Serve config:** 512 seq len, concurrency 1, `bfp_bf8` weights.
- **Image (current):** `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:e03b231b1de926cd8f9a0e1a2d39dd1df599f7a7_46a7c96_79778041239`
  (`tt-forge` 1.3.0.dev20260605003323). **NOTE:** all numbers below were measured
  on the prior build `...97aea20e..._fd9296b_79608294867` (`tt-forge`
  1.2.0.dev20260530002932, `vllm 0.19.1`, `torch-xla 2.9.0+git8d31cb3`); the spec
  has since been bumped to the 1.3.0 image but **not yet re-validated on it**.
- **Runner:** server selects `vllm_forge_gemma4_31b`
  (`ModelRunners.VLLMForge_GEMMA4_31B`); mesh `(1,4)`, `max_model_length=512`,
  `max_num_seqs=1`, `max_num_batched_tokens=2560`.

## Bring-up status (2026-06-05/06)

Verified **end-to-end serving** on the healthy box `qb2-120-p01t05`
(10.32.48.15) — 4-chip connected mesh, fabric `Degree histogram: {2:4}`,
weights load, compile, `Model warmup completed`, Uvicorn serving, coherent
chat output. (The earlier dev box `p01t06` was blocked only by a hardware
eth-link fault between its two P300 cards — see `QB2_P300_ETH_FABRIC_HANG_REPORT.md`.)

## Performance sweep — recovering the benchmark's 9 tok/s

The tt-xla benchmark `test_vllm_tp_benchmark[gemma4-31b-it-tp]` reports ~9
tok/s; the as-shipped tt-inference-server runner served ~4.5. Swept the runner's
`additional_config` flags (decode tok/s, bs1, 96 tokens, 512 seq len, p300x2):

| Config | trace | cpu_sampling | weights | greedy t/s | non-greedy t/s |
|---|---|---|---|---|---|
| baseline (as-shipped) | off | True (CPU) | bfp8 | 4.77 | 4.22 |
| +trace | **on** | True (CPU) | bfp8 | 8.14 | 7.19 |
| **+trace +device-sampling** ⭐ | **on** | **False (device)** | bfp8 | **9.20** | 7.39 |
| device-sampling only | off | False (device) | bfp8 | 5.26 | 4.40 |
| benchmark-match | on | False | **bf16** | 7.99 | 6.65 |

### Conclusions
1. **`enable_trace=True` is the dominant lever:** +71% greedy (4.77 → 8.14) on its
   own. Decode-graph replay. Trace-capture **fits in DRAM at 512 seq len** on the
   4-chip mesh (both bfp8 and bf16; no OOM).
2. **On-device sampling (`cpu_sampling=False`) only pays off *with* trace:**
   +13% greedy on top of trace (8.14 → 9.20). Alone it's marginal (4.77 → 5.26),
   because until trace removes per-token dispatch overhead, host sampling isn't
   the bottleneck.
3. **Keep `bfp_bf8` weights:** bfp8 (9.20) **beats** bf16 (7.99). bf16 doubles
   per-token weight DRAM traffic. The benchmark's bf16 only reached ~9 because it
   ran at `max_model_len=128` (less attention work) vs the server's 512.
4. **`optimization_level` must stay 0:** `opt>=1` aborts in tt-mlir
   MemoryLayoutPropagation on the 1.2.0 wheel (tt-xla#4990). `TTConfig` also
   rejects `enable_trace=True + opt>=1 + cpu_sampling=False`, so the trace
   defaults are only valid at opt 0.
5. **Non-greedy ceiling (~7.4 tok/s):** device sampling barely helps temp>0 — the
   top_p/temperature path keeps a host round-trip in this `vllm_tt` build. The
   benchmark's "9 tok/s" is greedy, so greedy 9.2 is the apples-to-apples match.

### Recommended runner defaults (applied)
The TP runner (`tt-media-server/tt_model_runners/vllm_forge_gemma4_31b.py`) now
mirrors the single-chip `vllm_runner.py`: env-var tunable with measured-best
defaults — `ENABLE_TRACE=true`, `CPU_SAMPLING=false`, `OPTIMIZATION_LEVEL=0`,
weights `bfp_bf8`. Net: **~9.2 tok/s greedy (≈2× the as-shipped 4.5)**.

## Gotchas for CI integration
- **`fp32_dest_acc_en` is NOT a valid `TTConfig` kwarg in `vllm_tt` 1.2.0** — the
  tt-xla benchmark's `_config` passes it; reusing that config verbatim raises
  `TypeError: TTConfig.__init__() got an unexpected keyword argument
  'fp32_dest_acc_en'` at engine start. (Benchmark-vs-image version skew.)
- `TT_MESH_GRAPH_DESC_PATH` must be pinned to the `p300_x2` descriptor (the gemma
  runner does not auto-set it, and the image maps `p300x2 -> p150`). Already done
  as an `-e` override in `workflows/model_specs/prod/cnn.yaml`.
- `model_name` must be lowercase `gemma-4-31b-it` (case-sensitive `ModelNames`).
- HF cache (`~/.cache/huggingface`) is outside the mounted `cache_root`, so weights
  re-download (~8 min) every launch unless `--host-hf-cache` is passed.
- Needs the full 4-chip connected mesh (degree `{2:4}`); a 2-chip mesh is
  insufficient (31B OOMs / topology won't map).

## Concurrency × context-length sweep (batch = max_num_seqs, len = max_model_len)

Goal: find a good concurrency/context balance (target batch-32 / 8K). Swept with
the perf-tuned decode defaults (trace on, device sampling, bfp8), `GMU=0.9`,
`XLA_PARAMETER_WRAPPING_THREADSHOLD=100000`. p300x2, 4-chip mesh. `agg`/`per` =
aggregate / per-stream decode tok/s over `batch` concurrent greedy streams
(short prompt, 64 tokens).

| cell | batch | seq len | max_batched_tokens | status | KV tokens | KV conc @len | agg tok/s | per-stream |
|---|---|---|---|---|---|---|---|---|
| A | 4  | 4096 | 16384  | **OK**     | 34176 | 8.34× | **28.8** | 7.20 |
| B | 4  | 8192 | 32768  | **OK**     | 34176 | 4.17× | **27.5** | 6.88 |
| C | 16 | 4096 | 65536  | **OK**     | 34176 | 8.34× | **76.3** | 4.77 |
| D | 16 | 8192 | 131072 | **OOM**    | 34176 | 4.17× | — | — |
| E | 32 | 8192 | 262144 | not run (D failed) | — | — | — | — |

### Key findings
1. **`max_num_batched_tokens` must be `>= max_model_len * max_num_seqs`** in this
   `vllm_tt` — not just `>= max_model_len` like upstream. Asserted in
   `model_runner.py:285`. This is the no-batched-`paged_fill_cache` limitation
   (proper fix tracked in tt-xla #5032/#5030). **Consequence:** the prefill graph
   is built over the full `batch*len` token budget, so prefill activation scales
   with **batch × len** — that product is the real wall, not batch and len
   independently.
2. **The fit boundary is the prefill-budget size**, ~between **65536 (C, fits)**
   and **131072 (D, OOM)** tokens. D died in **trace capture**
   (`capture_model -> _precompile_backbone -> torch_xla.sync()`) with a TT_FATAL
   allocation failure on the 131072-token prefill backbone — i.e. real activation
   OOM (≈ batch·len·hidden·2), surfacing at the trace-capture step.
3. **KV pool is ~34176 tokens at GMU=0.9, independent of seq len** (it's a
   GMU-sized budget). Fine for these batches with short requests, but it caps
   *worst-case full-length* concurrency: 8.34× at 4K, 4.17× at 8K. To serve many
   simultaneous *full-length* sequences you'd raise GMU and/or need the KV pool >=
   batch*len.
4. **Throughput scaling:** batch 4 → 16 (at 4K) gave 28.8 → 76.3 tok/s aggregate
   (~2.65×) while per-stream fell 7.2 → 4.77 (batched-decode contention). 8K vs 4K
   costs only ~4% aggregate at batch-4.
5. **Prefill serialization caveat:** batch-16/4K steady-state decode is 76 tok/s,
   but wall time for 16 simultaneous arrivals was 222s vs 39s at batch-4 — the 16
   prefills appear to run largely one-at-a-time (TTFT contention), a real
   first-token-latency concern for bursty load even where decode batches fine.

### Recommended balance & open follow-ups
- **Best fitting config found: batch-16 / 4K** (~76 tok/s aggregate, trace on).
  For long context at lower concurrency, **batch-4 / 8K** (~27 tok/s).
- **batch-16 / 8K and batch-32 / 8K do not fit** with trace on at GMU=0.9 — blocked
  by the `batch*len` prefill-activation OOM during trace capture.
- **Untested lever:** retry D/E with **`ENABLE_TRACE=false`**. Per the trace
  seq-len cap (#4220), the capture-time OOM may clear with trace off, extending the
  servable envelope (at the cost of the ~2× decode speedup). Not yet run.
- The real unlock for high batch × long context is **batched `paged_fill_cache`**
  (tt-xla #5032/#5030), which removes the `max_num_batched_tokens >= batch*len`
  requirement so prefill activation stops scaling with batch.
- Dimension overrides used here (`MAX_NUM_SEQS`, `MAX_MODEL_LEN`,
  `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`) are sweep-only (bind-mounted
  `runner_dims.py`); production `max_model_length`/`max_num_seqs` live in the
  server-side `(MODEL, DEVICE)` config, not the runner.

## Reference
- tt-xla CI: `test_vllm_tp_benchmark[gemma4-31b-it-tp]` /
  `test_tensor_parallel_generation_bhqb_gemma4_31b`.
- Sweep artifacts: `~/gemma_sweep/` (sweep.py, sweep_dims.py, results.jsonl,
  dims_results.jsonl, runner_dims.py, logs).
- Related: `HANDOFF_gemma4_31b_it_forge.md`, `QB2_P300_ETH_FABRIC_HANG_REPORT.md`.
- Param-wrap / batched paged_fill_cache: tt-xla #5032, #5030; trace seq-len cap: #4220.
