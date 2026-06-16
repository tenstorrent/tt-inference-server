# run.py against an already-running Forge server (`--server-url`) — smoke benchmark + eval

**Date:** 2026-06-12
**Model:** `Llama-3.1-8B-Instruct` (forge-vllm-plugin) on a single P150-class chip
**Branch:** `kmabee/forge_llm_chunked_prefill`
**Goal:** Drive `run.py` benchmark + eval workflows against an already-running
`tt-media-server` uvicorn process **without** spawning a new Docker container or
a second server.

---

## TL;DR

- To benchmark/eval a model on a server you already started, use **`--server-url`**
  (not `--docker-server`, not `--local-server`). It launches nothing, uses no
  Docker, and points the workflow client at the running server.
- `--server-url` is fully wired for **Forge** models. No code changes were needed.
- Three flags make it resolve the right spec:
  - `MODEL_SPECS_ENV=dev` — the forge single-chip P150 spec lives in `dev/cnn.yaml`
    (default is `prod`).
  - `--impl forge-vllm-plugin` — the forge block is `default_impl: false`, so it
    must be selected explicitly (otherwise it resolves to the `tt-transformers`/
    vLLM spec).
  - `--no-auth` — the dev-mode media server requires no auth.
- **Both smoke jobs passed** against the live server on port 8002.

---

## The running server under test

Launched earlier via `tt-media-server/launch_llama_8b.sh` (manual uvicorn, not
via run.py):

| field | value |
|---|---|
| process | `uvicorn main:app --port 8002` (pid 57869) |
| MODEL | `Llama-3.1-8B-Instruct` |
| DEVICE | `p150` |
| TT_VISIBLE_DEVICES | `1` (UMD chip ID 1 — second Blackhole p300c) |
| mesh | `(1,1)` single chip |
| auth | development mode, `/v1/models` returns 200 without a key |

Confirmed reachable:
```bash
curl -s http://127.0.0.1:8002/v1/models
# {"object":"list","data":[{"id":"meta-llama/Llama-3.1-8B-Instruct",...}]}
```

---

## Commands run

### Smoke benchmark

```bash
cd /home/kmabee/tt-inference-server
MODEL_SPECS_ENV=dev python run.py \
  --model Llama-3.1-8B-Instruct \
  --device p150 \
  --impl forge-vllm-plugin \
  --workflow benchmarks \
  --server-url http://127.0.0.1 \
  --service-port 8002 \
  --limit-samples-mode smoke-test \
  --no-auth
```

### Smoke eval

```bash
cd /home/kmabee/tt-inference-server
MODEL_SPECS_ENV=dev python run.py \
  --model Llama-3.1-8B-Instruct \
  --device p150 \
  --impl forge-vllm-plugin \
  --workflow evals \
  --server-url http://127.0.0.1 \
  --service-port 8002 \
  --limit-samples-mode smoke-test \
  --no-auth
```

Only the `--workflow` value differs between the two.

---

## Results

### Benchmark (`✅ Completed benchmarks` → `Completed run.py`)

vLLM `bench serve` against `/v1/chat/completions`, smoke params isl=128 / osl=128 /
concurrency=1 / 8 prompts:

| metric | value |
|---|---|
| Successful requests | 8 / 8 (0 failed) |
| Output token throughput | 11.0 tok/s (peak 14.0) |
| Total token throughput | 22.1 tok/s |
| Mean TTFT | 2420 ms (median 2341, P99 4484) |
| Mean TPOT | 72 ms |
| Mean E2EL | 11436 ms |

The `FAIL ⛔` cells in the perf table are the perf-target gate (functional target
wants TTFT ≤300 ms / 37 tok/s) — not a functional failure. A smoke run only
validates that it ran end-to-end and produced correct token counts. TTFT ~2.4 s
is consistent with the tt-xla validated ~2.09 s for this b32/64K stack.

### Eval (`✅ Completed evals` → `Completed run.py`)

`meta_ifeval`, smoke-test sample limit (2 samples):

| task | accuracy_check | score | ratio_to_reference | gpu_ref | published |
|---|---|---|---|---|---|
| meta_ifeval | **PASS ✅** | 100.00 | 1.23 | 81.38 | 80.40 |

All four ifeval metrics (`prompt_level_strict/loose_acc`, `inst_level_strict/loose_acc`)
returned `true`. Reports/acceptance stage also ran: `Acceptance criteria
enforcement: PASS (status=EXPERIMENTAL)`.

---

## Why this works (plumbing)

- `--server-url` is stored on `RuntimeConfig.server_url` (`workflows/runtime_config.py`).
- The benchmark/eval runners read it as the deploy URL:
  - `benchmarking/run_benchmarks.py:318` — `deploy_url = runtime_config.server_url or os.environ["DEPLOY_URL"]`
  - `evals/run_evals.py:915` — `env_config.deploy_url = runtime_config.server_url`
- The full target is `{server_url}:{service_port}` → `http://127.0.0.1:8002`.
- When neither `--docker-server` nor `--local-server` is set, `run.py` skips server
  launch (`run.py:699`) and goes straight to the workflows.
- `Llama-3.1-8B-Instruct` is **not** v2-routed (`workflows/v2_bridge.py`), so it uses
  the v1 benchmark/eval path.

### Gotchas

- **Spec env / impl:** Without `MODEL_SPECS_ENV=dev` and `--impl forge-vllm-plugin`,
  the model resolves to the `tt-transformers`/vLLM spec, which then trips an
  unrelated image-version gate: `⛔ Image v0.10.0 is not supported ... need v0.11.0+`.
  That gate (`validate_setup.py:_check_image_version_supported`) only applies to
  `inference_engine == VLLM`, so the forge spec skips it once correctly selected.
- The `vllm bench serve` client logs `Failed to load plugin tt` /
  `ModuleNotFoundError: No module named 'vllm.v1.attention.backends.registry'`.
  **Harmless** — the benchmark client is just an OpenAI HTTP client running on the
  CPU Platform; the real inference happens on the remote server at :8002.

---

## `--local-server` vs `--server-url` for Forge

- `--server-url` = target an **already-running** server. Correct tool here. No
  code change required; works for Forge today.
- `--local-server` = make `run.py` **start** the server itself (no Docker). This is
  **not implemented for Forge**: `workflows/run_local_server.py:generate_local_run_command`
  hard-raises `NotImplementedError` for non-vLLM specs, and even if unblocked it
  launches the **vLLM** entrypoint (`vllm-tt-metal/src/run_vllm_api_server.py`),
  not the `tt-media-server` uvicorn. It would also spawn a second server that
  collides with the running one. (The env-building half,
  `run_local_server.py:196`, already handles the FORGE/MEDIA `API_KEY`.)
- Implementing full `--local-server` Forge support (build a `uvicorn main:app`
  command for FORGE/MEDIA specs + lifecycle teardown) is a separate feature, not
  needed for the smoke runs above.
