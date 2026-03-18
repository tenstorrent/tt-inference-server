# tt-inference-server Agent Memory

Purpose: fast repo recall for future coding agents. Prefer this over re-reading broad user docs.

## Repo Mental Model
- This repo is not just an inference server. It is a model-readiness control plane plus multiple serving implementations plus release tooling.
- `run.py` + `workflows/`: host-side control plane. Resolves model/device/runtime, validates setup, exports runtime JSON, optionally launches Docker, runs client workflows, writes reports.
- `vllm-tt-metal/`: primary LLM serving path. Docker image and container entrypoint for Tenstorrent vLLM.
- `tt-media-server/`: non-LLM serving path, plus some LLM/embedding/OpenAI-style endpoints. Has both Python FastAPI and C++ Drogon implementations.
- `benchmarking/`, `evals/`, `stress_tests/`, `tests/`: certification/readiness workflows.
- `scripts/release/`: promotes CI outputs into updated model specs, docs, and release images.

## Primary Use Cases
- Operator: start a supported model quickly on Tenstorrent hardware.
- Validation engineer: run `benchmarks`, `evals`, `spec_tests`, `tests`, `stress_tests`, inspect reports, gate readiness.
- Model integrator: add or adjust supported models/devices/engines in `workflows/model_spec.py` and matching eval/benchmark/test coverage.
- Server developer: change vLLM container behavior, media-server routing/services/runners, or plugin packages.
- Release engineer: consume Models CI results, update model specs/docs, promote or build images, prepare release PRs.

## Most Important Files
- `run.py`: top-level CLI, runtime resolution, secret handling, startup flow, log/runtime JSON writing.
- `workflows/model_spec.py`: source of truth for supported models/devices/engines/default impls/runtime limits/env overrides/docker images.
- `workflows/run_workflows.py`: workflow dispatch. Code here decides what `release` actually runs.
- `workflows/runtime_config.py`: dynamic per-run state.
- `workflows/workflow_config.py`: maps workflow types to runner scripts.
- `workflows/run_docker_server.py`: generated Docker command and container lifecycle.
- `vllm-tt-metal/src/run_vllm_api_server.py`: direct container interface and model-spec resolution inside the image.
- `tt-media-server/open_ai_api/__init__.py`: active FastAPI route registration by `model_service`.
- `tt-media-server/config/settings.py`: env-driven media-server behavior and `MODEL`/`DEVICE` override logic.
- `tt-media-server/main.py`: FastAPI app lifecycle; note the docs/OpenAPI behavior caveat below.
- `scripts/release/README.md`: end-to-end release process.

## Execution Paths

### 1. Host workflow path
- User runs `python3 run.py --model ... --workflow ...`.
- `run.py` parses CLI, normalizes device/engine, exports `default_model_spec.json`, bootstraps `uv`, resolves `(RuntimeConfig, ModelSpec)`, handles secrets, writes runtime JSON, validates setup, optionally sets up host weights/caches, optionally starts Docker, then runs workflows.
- Runtime selection is centered on `get_runtime_model_spec()` in `workflows/model_spec.py`.
- `RuntimeConfig.runtime_model_spec` stores the serialized selected model spec for subprocesses.

### 2. Direct vLLM container path
- User bypasses `run.py` and runs `docker run <image> --model <hf_repo_or_short_name> --tt-device <device>`.
- `vllm-tt-metal/src/run_vllm_api_server.py` resolves a spec from bundled `default_model_spec.json`, unless `RUNTIME_MODEL_SPEC_JSON_PATH` points to a pre-resolved JSON from `run.py`.
- This path is the true container interface; `run.py` is optional host automation.

### 3. Media-server path
- Python path: `uvicorn main:app --lifespan on --port 8000` from `tt-media-server/`.
- C++ path: build under `tt-media-server/cpp_server/`, then run the compiled Drogon server.
- Active API surface depends on `settings.model_service`, which is usually inferred from `MODEL_RUNNER`.

## Workflow Map
- `server`: only starts the server. Requires `--docker-server` in normal usage. `run.py` skips `run_workflows()` for this workflow.
- `benchmarks`: waits for server health, may capture traces, runs benchmark tasks from `BENCHMARK_CONFIGS`, writes raw JSON to `workflow_logs/benchmarks_output/`, then auto-runs `reports`.
- `evals`: waits for server health, runs eval tasks from `EVAL_CONFIGS`, writes raw outputs to `workflow_logs/evals_output/`, then auto-runs `reports`.
- `reports`: summarizes previously written benchmark/eval/test outputs into markdown plus data files under `workflow_logs/reports_output/`.
- `spec_tests`: dynamic server integration tests driven by `tests/server_tests/server_tests_config.json`, then auto-runs `reports`.
- `tests`: pytest-based parameter tests, model-dependent via `tests/test_config.py`, then auto-runs `reports`.
- `stress_tests`: parameter sweeps and endurance/load testing, then auto-runs `reports`.
- `release`: wrapper workflow defined in code as `evals -> benchmarks -> spec_tests -> tests? -> reports`.

## Critical Workflow Invariants
- Non-`reports` workflows automatically run `reports` afterward.
- `release` is defined in `workflows/run_workflows.py`, not in docs prose.
- `tests` is included in `release` only if the model has an entry in `TEST_CONFIGS`.
- Trace capture is disabled after the first workflow step inside `release`.
- Client-side workflows can target an externally running compatible server; they do not require `run.py` to own the server lifecycle.

## Interfaces And Contracts

### `run.py` CLI
- Main human/operator interface.
- Key flags: `--model`, `--workflow`, `--tt-device`, `--impl`, `--engine`, `--docker-server`, `--service-port`, `--dev-mode`, `--workflow-args`.
- Device auto-detects if omitted.
- `--device` is a legacy alias of `--tt-device`.

### Workflow subprocess contract
- Workflow runners receive `--runtime-model-spec-json`, `--output-path`, and model/device context.
- The handoff JSON is the core machine-facing contract between `run.py` and workflow scripts.

### Runtime JSON contract
- `runtime_model_spec`: resolved static model/device/engine/impl config.
- `runtime_config`: dynamic run flags and workflow state.
- `run.py` writes these under `workflow_logs/runtime_model_specs/`.

### Model catalog contract
- `MODEL_SPECS` is built from `ModelSpecTemplate` expansion in `workflows/model_spec.py`.
- `ModelSpecTemplate` is the compact authoring format.
- `ModelSpec` is the resolved single model/device/impl/engine entry used at runtime.
- `default_model_spec.json` is exported from `MODEL_SPECS` and bundled into Docker images for direct container use.

### vLLM container interface
- Entrypoint: `vllm-tt-metal/src/run_vllm_api_server.py`.
- Accepts `--model`, `--tt-device`, optional `--engine`, `--impl`, `--no-auth`, `--disable-trace-capture`, `--service-port`, plus pass-through vLLM args.
- Requires TT Docker flags such as `/dev/tenstorrent`, hugepages, and `--ipc host`.

### Media-server HTTP interface
- Route registration lives in `tt-media-server/open_ai_api/__init__.py`.
- Enabled route families depend on `settings.model_service`.
- `llm`: `/v1/completions`
- `embedding`: `/v1/embeddings`
- `audio`: `/v1/audio/...`
- `image`: `/v1/images/...`
- `video`: `/v1/videos/...`
- `cnn`: `/v1/cnn/...`
- `training`: `/v1/fine_tuning/...`
- `tokenizer`: `/v1/tokenize`, `/v1/detokenize`
- maintenance: `/health`, `/tt-liveness`, `/tt-deep-reset`, `/tt-reset-device`, `/metrics`
- Legacy non-`/v1` aliases exist for several media routes via deprecation middleware.

### Plugin/package interfaces
- `tt-vllm-plugin/`: Tenstorrent vLLM plugin package.
- `tt-sglang-plugin/`: SGLang-based TT plugin/server path.
- These are separate from the main `run.py` workflow system but are part of the repo's serving surface.

## Directory-Level Mental Map
- `workflows/`: control plane and orchestration.
- `benchmarking/`: benchmark configs, runners, target references.
- `evals/`: eval configs, scoring, eval runners.
- `stress_tests/`: matrix generation, stress orchestration, summary reporting.
- `tests/`: root workflow tests plus server/spec test harness.
- `tt-media-server/`: API server, runners, schedulers, services, resolver, auth, telemetry, tests.
- `vllm-tt-metal/`: Docker image and LLM-serving entrypoint.
- `scripts/release/`: release automation and model-support doc generation.
- `utils/`: prompt clients, media clients, trace capture helpers.

## If The Task Is X, Start Here
- Add or change supported model/device/default impl: `workflows/model_spec.py`
- Change benchmark sweeps/targets: `benchmarking/benchmark_config.py` and `benchmarking/benchmark_targets/model_performance_reference.json`
- Change eval tasks/scoring: `evals/eval_config.py` and `evals/eval_utils.py`
- Change release sequence or auto-report behavior: `workflows/run_workflows.py`
- Change Docker launch behavior: `workflows/run_docker_server.py` and `workflows/setup_host.py`
- Change vLLM container behavior: `vllm-tt-metal/src/run_vllm_api_server.py`
- Change media HTTP routes: `tt-media-server/open_ai_api/`
- Change media env/config behavior: `tt-media-server/config/settings.py` and `tt-media-server/config/constants.py`
- Change media request/response schemas: `tt-media-server/domain/`
- Change media runner/service selection: `tt-media-server/resolver/`, `tt-media-server/model_services/`, `tt-media-server/tt_model_runners/`
- Add model-readiness support end to end: `workflows/model_spec.py`, `evals/eval_config.py`, benchmark targets, tests/spec-tests, then regenerate docs/release artifacts

## Model Support / Onboarding Notes
- For new support, start from `docs/add_support_for_new_model.md`, but treat code as final truth.
- add `ModelSpecTemplate` or related device entries in `workflows/model_spec.py`
- add eval coverage in `evals/eval_config.py`
- add performance targets in `benchmarking/benchmark_targets/model_performance_reference.json`
- add spec tests or parameter tests when relevant
- regenerate exported support docs / artifacts through release tooling if needed

## Useful Terminology
- `ModelSpecTemplate`: compact template expanded into many `ModelSpec` entries.
- `ModelSpec`: resolved model/device/impl/engine definition used at runtime.
- `DeviceModelSpec`: per-device limits and overrides like `max_context`, `max_concurrency`, `vllm_args`, TT config, env vars.
- `RuntimeConfig`: dynamic per-run flags from CLI and workflow state.
- `default impl`: chosen automatically if `--impl` is omitted.
- `runtime model spec JSON`: serialized runtime handoff written by `run.py`.
- `WorkflowConfig`: static mapping from workflow type to runner script/venv.

## Known Caveats / Mismatches
- Prefer code over docs when they conflict.
- `--local-server` is supported only for vLLM-backed model specs.
- `--local-server` always uses host filesystem persistence; if no host storage flags are passed it defaults to `REPO_ROOT/persistent_volume/` for logs, weights, and TT caches.
- `stress_tests/README.md` says stress tests are included in `release`, but `workflows/run_workflows.py` does not include them in the actual release sequence.
- Python media-server route registration does not automatically imply every OpenAI route exists in every mode; check the active router modules.
- The Python FastAPI media server exposes `/v1/completions`; do not assume `/v1/chat/completions` exists there just because some docs mention it. Re-check implementation before promising chat support.
- `tt-media-server/main.py` force-sets `env = "development"`, so docs/OpenAPI are effectively always enabled there.
- vLLM path uses `JWT_SECRET` and related API-key logic.
- Python media server uses `API_KEY`.
- C++ server documentation references `OPENAI_API_KEY`.
- `run.py` exports `default_model_spec.json` at startup; do not hand-edit that file as source of truth.

## Recommended Read Order For Unfamiliar Tasks
- `AGENTS.md`
- `docs/agent_memory.md`
- `run.py`
- `workflows/model_spec.py`
- `workflows/run_workflows.py`
- then the specific subsystem doc or entrypoint you are touching

## Bottom Line
- Think of this repo as a workflow orchestrator wrapped around multiple serving backends.
- When in doubt, trace the path `run.py -> model_spec/runtime_config -> docker/server -> workflow runner -> reports`.
- For support questions, `workflows/model_spec.py` is the center of gravity.
