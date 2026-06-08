# Initial Forge-TP support: gemma-4-31b-it + Qwen3-32B on p300x2 (serving + evals)

Part of the forge-TP-p300x2 umbrella. Adds both models as **servable + eval-able** at `EXPERIMENTAL` in CI. **Benchmarks are out of scope here** (blocked on the Benchmark-uplift and Benchmark-runtime items) — but serving and evals run, which unblocks perf work.

## Changes (done on branch `kmabee/gemma4_31b_it_forge`)
| Area | File |
|---|---|
| Model specs (dev catalog — CI reads dev) | `workflows/model_specs/dev/cnn.yaml` |
| Runner (env-var tunable: ENABLE_TRACE/CPU_SAMPLING/OPTIMIZATION_LEVEL; perf defaults trace-on/device-sampling/bfp8) | `tt-media-server/tt_model_runners/vllm_forge_{gemma4_31b,qwen_32b}.py` |
| Server config (TP mesh topology only; dims env-driven) | `tt-media-server/config/constants.py` |
| Forge wheel | `tt-media-server/tt_model_runners/forge_runners/requirements.txt` |
| Evals (ifeval, downsampled 0.1 / 0.01) | `evals/eval_config.py` |
| Perf-ref placeholder | `benchmarking/benchmark_targets/model_performance_reference.json` |
| Nightly matrix (FORGE/P300X2) | `.github/workflows/models-ci-config.json` |

## Must-know gotchas
- **CI reads `dev`** (`run.py --dev-mode` is hardcoded in tt-shield's run job) — not prod.
- `model_name` must be lowercase `gemma-4-31b-it` (case-sensitive `ModelNames`).
- Pin `TT_MESH_GRAPH_DESC_PATH` to the `p300_x2` descriptor (runner doesn't auto-set; image maps p300x2→p150).
- Dims env-driven via cnn.yaml `env_vars` (constants.py = TP topology only; model from `MODEL` env).
- Needs the full 4-chip connected mesh (degree `{2:4}`).

## Acceptance
- Nightly/release reaches **serving ✅ + evals ✅** for both models on p300x2.
- Benchmark step known-failing until the Benchmark-uplift (gemma tokenizer) and bounded by the Benchmark-runtime work; not gating this work item.

## Validated
Local p01t05 on the build image: both serve at 4K/16; gemma ifeval 0.89; Qwen eval functional.
