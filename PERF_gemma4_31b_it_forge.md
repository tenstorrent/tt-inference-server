# gemma-4-31b-it (Forge, tensor-parallel) — serving + performance findings

Source-of-truth notes for opening a ticket to add **gemma-4-31b-it** to
tt-inference-server + tt-shield CI. Captures the end-to-end bring-up and a
flag sweep that recovers the tt-xla benchmark's ~9 tok/s.

- **Model:** `google/gemma-4-31B-it` (loaded internally; host spec uses lowercase
  `gemma-4-31b-it` to match `ModelNames.GEMMA_4_31B_IT`).
- **Device:** QB2 = 2× P300 = `p300x2`, 4 chips, `(1,4)` 1D tensor-parallel mesh.
- **Serve config:** 512 seq len, concurrency 1, `bfp_bf8` weights.
- **Image:** `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:97aea20e33246455fa81b1d2bc5270093415ee23_fd9296b_79608294867`
- **Wheels in image:** `tt-forge` / `pjrt-plugin-tt` / `vllm_tt` all
  `1.2.0.dev20260530002932`; `vllm 0.19.1`; `torch-xla 2.9.0+git8d31cb3`.
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

## Reference
- tt-xla CI: `test_vllm_tp_benchmark[gemma4-31b-it-tp]` /
  `test_tensor_parallel_generation_bhqb_gemma4_31b`.
- Sweep artifacts: `~/gemma_sweep/` (sweep.py, results.jsonl, per-config runners, logs).
- Related: `HANDOFF_gemma4_31b_it_forge.md`, `QB2_P300_ETH_FABRIC_HANG_REPORT.md`.
