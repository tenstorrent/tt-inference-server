# Bringing a new model to tt-inference-server + tt-shield CI (forge / tt-xla)

A staged path for taking a model that runs in tt-xla and getting it served in tt-inference-server and running in nightly tt-shield CI with production-like configs (b32, high concurrency, long context). Stages 1, 2, and 4 are the required path; Stage 3 (local testing) is optional but recommended for fast iteration.

- **Stage 1 — Validate standalone in tt-xla** by adding a config to the vLLM benchmark suite (`tests/benchmark/test_vllm_benchmarks.py`) with the full production config.
- **Stage 2 — Integrate into tt-inference-server** — add everything CI reads: model registration, model spec, eval config, perf targets, nightly YAML.
- **Stage 3 — Local testing (optional)** — run the release/eval flow locally via `run.py`, against either its docker server or your own uvicorn server from a tt-xla checkout.
- **Stage 4 — CI testing** — validate through tt-shield On-Dispatch, then nightly.

Reference models to copy throughout: **Qwen3-8B** for a **single-chip (P150)** model, or **gemma-4-31b-it** for a **tensor-parallel / multi-chip** model (a TP forge LLM running on p300x2 / QB2). When in doubt, `grep -ril qwen3-8b` (or `gemma-4-31b`) and mirror every place it appears.

---

## Stage 1 — Validate in tt-xla standalone (vllm benchmark suite, production settings)

Goal: prove the model compiles and serves at the target production config on a single chip. Make note of the throughput and TTFT so you can compare against the same model once it's integrated into tt-inference-server. No tt-inference-server involved yet — this isolates the model/plugin from the serving wrapper.

### Production config knobs (the target)

| Knob | Value | Where (vLLM `--additional-config` key unless noted) |
|---|---|---|
| Batch | `--max-num-seqs 32` (b32) | vllm serve flag |
| Context | as high as possible, ideally = model max | `--max-model-len` |
| b1-prefill | on | `min_num_seqs: 1`, `prefill_batch_threshold: 16` |
| KV cache dtype | BFP8 | `experimental_kv_cache_dtype: bfp_bf8` |
| Weights dtype | BFP8 | `experimental_weight_dtype: bfp_bf8` |
| Optimization level | >= 1 | `optimization_level: 1` |
| Trace | on | `enable_trace: true` |
| Sampling | on-device | `cpu_sampling: false` |
| GPU mem util | as high as fits (max concurrency) | `--gpu-memory-utilization` (tune, see below) |
| Chunked prefill | on for long context | `--enable-chunked-prefill` + `prefill_chunk_size: 2048` |
| Const eval | on | `enable_const_eval: true` |

In the benchmark suite (`test_vllm_benchmarks.py`, the Stage 1 deliverable) the `--additional-config` keys map directly onto the `_config(...)` `additional_config` dict, and the CLI-flag rows are set via `_config(...)` args / `TT_BENCHMARK_*` env vars (below). The literal CLI flags shown above apply to a standalone `vllm serve` (the optional demo).

`max_num_batched_tokens`: with chunked prefill **on**, set it so the plugin right-sizes (the plugin caps it to `prefill_chunk_size * max_num_seqs`); with chunked prefill **off**, vLLM requires `max_num_batched_tokens >= max_model_len * max_num_seqs`.

### Add the benchmark config (this is the Stage 1 deliverable)

The vLLM benchmark suite lives in tt-xla at **`tests/benchmark/test_vllm_benchmarks.py`**. Add your model as a config entry (CI runs it) — **mimic the nearest existing test**: a `_config(...)` line in `SINGLE_DEVICE_CONFIGS` for single chip, or a `_tp_config(...)` line in `TP_CONFIGS` for tensor-parallel (mesh shape is board-dependent — copy an existing entry for the *same* hardware, don't assume `[2, 4]`).

Test locally with `pytest -k "<model-id>" -s`. CI (and you locally) sweep the production config via `TT_BENCHMARK_*` env vars rather than duplicating entries: `TT_BENCHMARK_MAX_MODEL_LEN` (default 128 — raise toward model max), `TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8`, `TT_BENCHMARK_GMU`, `TT_BENCHMARK_PREFILL_CHUNK_SIZE`, `TT_BENCHMARK_WEIGHT_DTYPE`, `TT_BENCHMARK_BATCH_SIZE`, `TT_BENCHMARK_TRACE`, `TT_BENCHMARK_CPU_SAMPLING`, `_BENCH_OPTIMIZATION_LEVEL`. (On a QB2 / multi-chip host targeting a single P150, also set the single-P150 env — see Appendix.)

Sampling caveat: `_config` currently force-sets `cpu_sampling=True` whenever `optimization_level>=1`, so you can't benchmark the production **device-sampling** path (opt1 + trace + device sampling) as-is — tracked in [tt-xla#5403](https://github.com/tenstorrent/tt-xla/issues/5403). In the meantime, delete these two lines in `_config` (in `tests/benchmark/test_vllm_benchmarks.py`) locally:

```python
        # TTConfig raises if enable_trace=True AND opt>=1 AND cpu_sampling=False
        additional["cpu_sampling"] = True
```

### Optional: standalone client + server demo in tt-xla

For a self-contained compile-run-interact demo, add a `client + server` example under `examples/vllm/` (reference **`examples/vllm/TinyLlama-1.1B-Chat-v1.0/`**, which exists in main). This lets you (and reviewers) compile the model, serve it standalone, and interact with it directly — handy for first-compile debugging before/alongside the benchmark config. Not required for CI; see that example for the pattern.

### Pass criteria for Stage 1

- Full model compiles and serves at b32 + target context + the config above (no DRAM OOM at trace capture).
- Coherent output (not gibberish — that means single-layer/misconfig).

---

## Stage 2 — Integrate into tt-inference-server

Goal: add everything the CI flow reads so the model can be served and evaluated unattended. Mirror Qwen3-8B in each file (`grep -ril qwen3` and add the equivalent). The canonical (non-forge-specific) reference for the model-spec / eval / perf-target steps is [`docs/add_support_for_new_model.md`](docs/add_support_for_new_model.md) — this guide adds the forge/tt-xla specifics on top.

### 1. Model registration (tt-media-server)

- `tt-media-server/config/constants.py`, `config/vllm_settings.py`, `config/settings.py` — register the model and its defaults. **Required for CI** (the served config resolves from here). `grep -ril qwen3` under `tt-media-server/` and add the equivalent for your model everywhere it appears. The additional-config keys are the same ones from Stage 1.
- **Single-chip vs tensor-parallel**: single-chip models (Qwen3-8B) use the default vLLM forge runner. A **TP / multi-chip model needs a dedicated runner module** — mirror `tt-media-server/tt_model_runners/vllm_forge_gemma4_31b.py` (sets `enable_tensor_parallel: True`) — plus a `device_mesh_shape` whose first dim is > 1 (tt-media-server derives `is_tensor_parallel` from `device_mesh_shape[0] > 1`; `constants.py` carries the TP mesh topology). **gemma-4-31b-it** is the reference TP LLM (spec in `workflows/model_specs/dev/cnn.yaml`, `(1, 4)` mesh on p300x2 / QB2).

### 2. Model spec

- `workflows/model_specs/dev/llm.yaml` — add your model's serving spec (batch, context, chunk size, dtypes, opt level, mesh/device, HF id). Use the **same production config you validated in Stage 1** — b32, highest seq len (ideally model max), BFP8 KV + weights, opt >= 1, trace, chunked prefill, and GMU as high as fits.

### 3. Eval config

- `evals/eval_config.py` — add `EvalTask` entries (e.g. `gpqa_diamond`, `mmlu_pro`) with `max_concurrent`, `timeout`, and any `gen_kwargs`. `run.py` reads this file from the **working tree at run time**, so local edits are live.
- Add the evals appropriate for your model (its task type / domain / expected capabilities). **If you're unsure which evals to run, ask the CSE team.**
- Evals in particular benefit from the **highest `max_concurrent` and `max_model_len`** the model sustains: evals push many prompts, so more concurrency drops wall-clock, and a larger context avoids truncating long eval prompts.

### 4. Perf targets

- `benchmarking/benchmark_targets/model_performance_reference.json` — add your model's target points on the ISL/OSL/concurrency curve, each with a **theoretical** `ttft_ms` / `tput_user` / `tput`. These are the pass/fail checkpoints Models CI measures against — **target/reference numbers, not "what we currently achieve."** See [`docs/add_support_for_new_model.md`](docs/add_support_for_new_model.md) (Step 4) for the format.
- If you don't have derived targets, base them on a **trustworthy comparable model** — same parameter size *and* same architecture / hardware class (QB2, Galaxy, etc.) — rather than guessing. If unsure, ask the CSE team.

### 5. Nightly CI config

- `.github/workflows/models-ci-config.json` — add your model entry (impl, runner label, device type, workflow, limits). Validated by `.github/workflows/validate-models-ci-config.yml` against `models-ci-config-schema.json` — run that validation before pushing.

---

## Stage 3 — Local testing (optional)

Goal: see that model can be served, run the ~15 min smoke release flow (`run.py`) on your box before pushing to CI. Optional, but the fastest way to iterate — especially against local tt-xla source changes.

`run.py` is the release/eval orchestrator CI runs (`--workflow release` = evals -> benchmarks -> spec_tests -> tests -> reports, or `--workflow evals`). It reads `evals/eval_config.py` and the model specs from the **working tree**, so uncommitted local edits are active. Pick how it gets a server:

- **`--server-url <url>` (+ `--service-port`) — the fast path, and the default target (`http://127.0.0.1:8000` if you pass no server flag).** Point it at a server you started yourself — e.g. a local tt-media-server (uvicorn) from your **live tt-xla checkout**. No wheel/docker wait, and your local tt-xla source edits are live. This is the reason to bother running the uvicorn server.
- **`--docker-server` — what CI uses.** `run.py` launches its own server in a docker image built from tt-forge wheels (nightly at best), so expect latency waiting for a wheel that contains your change.
- **`--local-server`** — `run.py` launches a server on the host from a prebuilt tt-metal python venv (an alternative to docker).

### Bring-your-own server (the optional uvicorn path)

A personal `launch_<model>.sh` makes this a one-liner — a local convenience, **not** needed for CI. Mirror an existing launcher ([gist with all three scripts](https://gist.github.com/kmabeeTT/f997b3cf1b59f7325e7d64ea273880d1)): [`launch_qwen3_8b.sh`](https://gist.github.com/kmabeeTT/f997b3cf1b59f7325e7d64ea273880d1#file-launch_qwen3_8b-sh) (single-chip P150), or [`launch_server_gemma4_31b.sh`](https://gist.github.com/kmabeeTT/f997b3cf1b59f7325e7d64ea273880d1#file-launch_server_gemma4_31b-sh) — the launcher used to bring up **gemma-4-31b-it** (TP, 4-chip 1x4 mesh) on QB2 / p300x2. The single-chip launcher sets the additional-config env (`MAX_MODEL_LENGTH`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `KV_CACHE_DTYPE`, `PREFILL_CHUNK_SIZE`, `MIN_NUM_SEQS`, `PREFILL_BATCH_THRESHOLD`, `OPTIMIZATION_LEVEL`, `ENABLE_TRACE`, `CPU_SAMPLING`, `DEVICE_IDS`) then runs `uvicorn main:app`:

```
cd ~/tt-xla && source venv/activate
cd ~/tt-inference-server/tt-media-server

# one-time: install the tt-media-server server deps into the (tt-xla) venv
# need to remove torch and torchaudio old deps from this file not compatible with forge
pip install -r requirements.txt

# requirements.txt misses a few — you'll typically also need:
pip install uvicorn fastapi colorama prometheus_client faster_fifo python-multipart

DEVICE_IDS=0 PORT=8019 ./launch_<model>.sh
# on QB2 / multi-chip targeting a single P150, extra env is needed — see Appendix
```

For a worked example of launching a model on a branch (to serve it) and reproducing evals both locally and on CI, see [this comment on issue #4431](https://github.com/tenstorrent/tt-inference-server/issues/4431#issuecomment-4840148344).

Verify it serves — the [helper scripts](#helper-scripts-for-interacting-with-a-hosted-server) at the bottom of this doc are handy here for poking at / benchmarking the hosted model — then run the flow against it. For a **quick end-to-end sanity check** use [`smoke_release_qwen3_8b.sh`](https://gist.github.com/kmabeeTT/f997b3cf1b59f7325e7d64ea273880d1#file-smoke_release_qwen3_8b-sh) (release flow at `--limit-samples-mode smoke-test` = smoke evals + one small benchmark, ~15 min on P150); use the ci-nightly release for a fuller run:

```
curl -s http://127.0.0.1:8019/v1/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Name one color."}],"max_tokens":16}'

./smoke_release_qwen3_8b.sh                                              # quick sanity: smoke evals + benchmark (~15m)
```

A smoke run's accuracy/perf pass-fail isn't meaningful (tiny sample counts) — `exit 0` just confirms the end-to-end release path (evals -> benchmarks -> spec_tests -> tests -> reports) works.

### Reproduce the exact CI environment locally

To rule out "works locally, fails in CI" drift, run the server inside the CI docker image instead of your venv: tt-shield's On-Dispatch "create docker image" job produces a prebuilt image (the exact env CI uses). Pull it, run the server in it, and point `run.py` at it via `--server-url`. The local editable venv can differ from the CI wheel (pinned to a tt-xla build number = commit; a fix committed after that commit is not in the wheel).

### Local gotchas

- **Pin the chip via `DEVICE_IDS`** (e.g. `DEVICE_IDS='(0)'`) — see the Appendix for how this differs from standalone tt-xla and for single-P150 targeting on QB2 / multi-chip hosts.
- **Teardown before every relaunch**, or an orphan EngineCore holds the device / port: `pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "uvicorn main:app"; fuser -k <port>/tcp`, then confirm `fuser /dev/tenstorrent/<n>` is empty. EngineCore is a `comm` name, so `pkill -f` may miss it — kill by PID if needed.
- Front-end `Application startup complete` appears **before** the EngineCore finishes compiling — readiness = a real generate returns text, not the front-end log line.

---

## Stage 4 — CI testing

Goal: the model runs unattended in nightly tt-shield CI with pass/fail on accuracy and perf. Validate first via On-Dispatch, then it rides nightly.

Trigger tt-shield's `on-dispatch.yml` manually (GitHub Actions → Run workflow) to exercise your model without waiting for nightly. Key inputs:

- `model`, `impl-of-model` = `forge-vllm-plugin`, `device-type` + `runner-label` for your hardware — single-chip P150 uses `device-type = p150` / `runner-label = tt-ubuntu-2204-p150b-stable`; a TP / multi-chip model uses its board's values (e.g. QB2 / p300x2).
- `workflow` = `release` (evals + benchmarks + tests) or `evals`.
- `inference-server-git-ref` = your tt-inference-server branch.
- `tt-forge-version-override` = a specific tt-xla wheel/build if you need one (pinned by build number = a tt-xla commit).
- `docker-image` = a prebuilt image (skips the build); the eval **client** runs from the branch checkout, not from the image.
- `run-full-evals` = `false` to use CI-mode downsampled limits (ci-nightly), `true` for the full set.

For a forge-only run, set the forge flags on and unset the media/blaze/vllm ones.

Example On-Dispatch run (TP / multi-chip): [gemma-4-31b-it on QB2](https://github.com/tenstorrent/tt-shield/actions/runs/27560515075) — a good reference for the inputs and outputs of a real forge TP model run (the single-chip P150 case uses the `p150` `device-type` / `runner-label` above).

Note: tt-shield scheduled Nightly will use "latest pre-release forge wheel" (a recently added feature) which is generated from the most recent tt-xla nightly. This is to reduce latency to get changes from tt-xla into tt-shield CI and not run the same thing every night.  This is different than default for On-Dispatch CI job which will use the tt-forge wheel pinned in requirements.txt.

---

## Tips & tricks

### Helper scripts for interacting with a hosted server

Convenience scripts (in [`github.com/kmabeeTT/scripts`](https://github.com/kmabeeTT/scripts)) for driving a running server without hand-writing curl/JSON. They auto-discover the model via `/v1/models`, wait for readiness, and default to API key `your-secret-key` / port 8000 (override with `PORT`/`SERVER`/`API_KEY`):

- **[test_all_llm_servers.sh](https://github.com/kmabeeTT/scripts/blob/main/test_all_llm_servers.sh)** — discovers every running LLM server (docker `tt-inference-server-*` containers + local uvicorn/vllm processes) and benchmarks / health-checks each. Supports `--health`, `--concurrent N[,N...]`, exact `--isl`/`--osl` token lengths (`ignore_eos`), concurrency×ISL×OSL sweeps with a comparison table, and explicit `--rep-penalty`/`--temperature`/`--seed`. The go-to for production-config benchmarking and comparing servers/configs side by side.
- **[client_demo.sh](https://github.com/kmabeeTT/scripts/blob/main/client_demo.sh)** — single-stream interactive demo against one server; auto-detects chat vs completions, streams the output, prints prompt/completion token counts. `MAX_TOKENS` as arg 1; `PORT`/`TEMPERATURE`/`REPETITION_PENALTY` via env. Quickest "is it generating sane text?" check.
- **[client_demo_concurrent.sh](https://github.com/kmabeeTT/scripts/blob/main/client_demo_concurrent.sh)** — fires N concurrent streams at one server and redraws them live in place (args: `N`, `MAX_TOKENS`, `STAGGER`). Good for eyeballing batched / concurrent decode behavior at a glance.

---

## Appendix — quick reference & gotchas

- **Which venv is live?** Add a one-line marker print in the model-loading path (tt-xla `vllm_tt/model_runner.py` `load_model()`), use a raw `print(..., file=sys.stderr, flush=True)` so it bypasses log-level suppression, and grep the server log. Confirms the server runs the source/branch you think it does.
- **`vllm_tt.*` loggers are at WARNING** in tt-media-server (INFO suppressed). For probe output use `logger.warning` or a raw stderr `print`. EngineCore prints show as `(EngineCore pid=...)`.
- **Benchmark hygiene**: prefix caching is on — use distinct per-run prompt tokens or TTFT is bogus. Always send `temperature`/`repetition_penalty` explicitly. Run each shape twice, trust the 2nd.
- **DRAM OOM at trace capture** is the #1 bring-up failure. It is a *fit* problem (GPU mem util too high, prefill chunk too big, or head_dim too large), not a KV-math problem. Lower gpu-memory-utilization, or `prefill_chunk_size` 2048 -> 1024, before assuming a deeper bug. Grep the docker/server log for `TT_FATAL` / `Out of Memory` first.
- **Env propagation**: custom env vars reach the EngineCore subprocess but not always the vllm_runner worker (only ones it reads explicitly). To force a plugin config for a test, set it in the launch script / `config/vllm_settings.py`, not via a new env var.
- **Default API key**: `your-secret-key` (`Authorization: Bearer your-secret-key`).
- **Single-P150 targeting on QB2 / multi-chip hosts.** If the box has multiple chips and you want just one P150, **standalone tt-xla** runs (Stage 1 benchmarks, or a bring-your-own server) need the chip pinned plus the board's mesh descriptor: `TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<tt-xla>/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`. **tt-media-server** needs the same mesh descriptor but pins the chip via `DEVICE_IDS` (e.g. `DEVICE_IDS='(0)'`) instead — it overwrites `TT_VISIBLE_DEVICES` with the worker's device id, so setting `TT_VISIBLE_DEVICES` for the server has no effect.
- **Stage ordering**: Stage 1 rules out the model/plugin in isolation; Stage 2 is the required integration wiring; Stage 3 (optional) catches serving/config issues locally before you burn CI cycles; Stage 4 is the CI gate. Doing the early isolation first keeps later failures unambiguous.