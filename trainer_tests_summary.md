# Context: tt-blacksmith LoRA training test → tt-shield CI (`--workflow training_tests`)

## Goal
Run one tt-blacksmith LoRA fine-tune (Llama-3.1-8B + SST-2, single-chip **P150**) as a nightly CI correctness gate, via the **forge inference-server** path (not tt-metal, not vLLM). `tt-inference-server`'s `TrainerTrainingLoraRunner` wraps tt-blacksmith's `LoraLLMTrainer`; tt-blacksmith is pinned+baked into the forge image. `run.py --workflow training_tests` brings up the forge server, submits one LoRA job over HTTP, polls loss, grades the trajectory, and writes `report_data_*.json` for collect_data.

## Repos / branches (both have UNCOMMITTED changes)
- **tt-inference-server** → branch `mcupac/shield-CI-integration` (impl was committed in `c6ce10f16 "initial changes"`; my rename + registry fix sit on top, uncommitted).
- **tt-shield** → branch `mcupac/training-integration` (uncommitted working-tree changes).

## Naming decision
Workflow value is **`training_tests`** (plural, consistent with `spec_tests`/`stress_tests`), NOT bare `training`. Bare `training` is intentionally reserved for a possible future un-asserted training activity. `WorkflowType.from_string('training_tests')` → `TRAINING_TESTS`; bare `'training'` is now rejected.

## Files changed

### tt-inference-server (`mcupac/shield-CI-integration`)
Rename `training` → `training_tests`:
- `workflows/workflow_types.py` — enum member `TRAINING` → `TRAINING_TESTS`.
- `workflows/workflow_dispatch.py` — `_ENGINE_WORKFLOW_NAMES` maps `WorkflowType.TRAINING_TESTS: "training_tests"`; `_is_training_run` compares `TRAINING_TESTS` (still requires `model_spec.model_type == ModelType.TRAINING` — `ModelType.TRAINING` is unchanged); `_build_training_cmd` passes `"training_tests"` (launcher `--workflow` value + output-dir label + resolver label).
- `launchers/run_training_test.py` — `--workflow` guard (`!= "training_tests"`), guard message, docstring example, and report metadata `"workflow": "training_tests"`.
- `workflows/model_specs/dev/training.yaml` — comment updated.

Registry bug fix (found this session):
- `workflows/training/registry.py` — key was the HF path `("meta-llama/Llama-3.1-8B","p150")`, but runtime passes `model_spec.model_name` = **`Llama-3.1-8B`** (the weights basename). Changed key to `("Llama-3.1-8B","p150")`. Would otherwise `KeyError` at runtime.
- `tests/workflows/training/test_registry.py` — updated to `Llama-3.1-8B`.

Pre-existing impl files (unchanged this session, for reference):
- `launchers/run_training_test.py` — HTTP client: waits `/health`, `POST /v1/jobs` (retries on 405 = model-not-ready), polls `GET /v1/jobs/{id}` (terminal = completed/failed/cancelled), `GET /v1/jobs/{id}/metrics`, grades, writes report. Note: **model is NOT in the POST body** — `_build_request_body` only sets `device_type` + hyperparameters; the running forge server already *is* the model. `--model` only drives spec resolution, registry lookup, report labeling. JWT read from `--jwt-secret` or `$JWT_SECRET`.
- `workflows/training/loss_check.py` — pure grader (no http/torch), emits `spec_tests`-shaped records. Loose tolerances (`rtol=0.5, atol=0.1`, require-decreasing, final-loss ceiling).
- `workflows/training/expected/llama_3_1_8b_sst2_p150.yaml` — expected trajectory + `request` hyperparams (SST2, batch_size 8, max_steps 15, steps_freq 5, lora_r 4, etc.). **Values are placeholders** (`TODO(regenerate-on-hardware)`).
- Catalog: `trainer_training_lora` impl (impl_name `trainer-training-lora`), dev-only entry in `workflows/model_specs/dev/training.yaml`.
- Unit tests: `tests/workflows/training/{test_loss_check,test_registry}.py` — **18 pass** (`uv run pytest tests/workflows/training/`).

### tt-shield (`mcupac/training-integration`)
- `.github/workflows/dynamic-workflow.yml` — new `run-training` input (default true) + `test-training` job: `needs: [build-forge-media-inference-server]`, gated on `inputs.run-training` + forge build success; reuses forge image; calls `workflow_run-tests-with-inference-server.yml` with `workflow: training_tests`, hardcoded 1-entry matrix `[{"model":"meta-llama/Llama-3.1-8B","runner":{"label":"p150","type":"p150"},"impl":"trainer-training-lora"}]`, `timeout-minutes: 120`.
- `.github/workflows/on-nightly.yml` — `run-training` input (default true) passed through as `run-training: ${{ github.event_name == 'schedule' || inputs.run-training }}`.
- `CLAUDE.md` — integration notes (updated to `training_tests`).
- Left the toggle named `run-training` (human switch; not the `--workflow` string).
- NOTE: the tt-shield matrix still lists the model as `meta-llama/Llama-3.1-8B`. This flows to `--model` in the reusable workflow → **will fail argparse** (see below). Likely needs to become `Llama-3.1-8B` too, unless that workflow maps it. **Verify this before trusting CI.**

## Manual P150 run command (verified flags)
```bash
export HF_TOKEN=hf_...        # gated Llama-3.1-8B weights
export JWT_SECRET=test-secret # shared by server+launcher (or use --no-auth)
export MODEL_SPECS_ENV=dev    # loads dev catalog at import so trainer impl is a valid choice

MODEL_SPECS_ENV=dev python run.py \
  --model Llama-3.1-8B \
  --workflow training_tests \
  --device p150 \
  --impl trainer-training-lora \
  --dev-mode \
  --docker-server \
  --override-docker-image <forge-image-tag> \
  --service-port 8000
```
Key gotchas:
- **`--model Llama-3.1-8B`** (short basename), NOT `meta-llama/Llama-3.1-8B` — argparse `choices` are built from `model_name`.
- **Need both** `MODEL_SPECS_ENV=dev` (makes `--impl trainer-training-lora`/model valid argparse choices at import) **and** `--dev-mode` (resolves the dev spec + forwards into container).
- `--impl` takes the **hyphenated** impl_name `trainer-training-lora`.
- Do NOT pass `--expected-config` — `_build_training_cmd` auto-resolves it from the registry.
- `--device p150` is the hidden alias of `--tt-device p150`.
- Long runtime: launcher defaults `--health-timeout 3600`, `--job-timeout 5400`; 8B forge bringup takes many minutes — don't kill early.

## How to test (cheapest → real)
1. **No hardware, seconds** — routing check via `build_engine_commands`: assert argv has `--workflow training_tests`, ends in `run_training_test.py`, includes `--expected-config`. (Pattern: `tests/workflows/test_workflow_dispatch_routing.py`; no training case there yet.)
2. **No hardware** — launcher client path against a stub HTTP server (canned `/health`, `/v1/jobs`, `/metrics`); assert it writes `report_data_*.json` + exit code. (Not yet unit-tested.)
3. **Real P150** — the command above. **Verdict not trustworthy yet** (placeholder losses) — treat first green run as "runs end-to-end + emits metrics", then read `GET /v1/jobs/{id}/metrics` and paste real `train_loss`/`val_loss` into `llama_3_1_8b_sst2_p150.yaml` before relying on PASS/FAIL. Regenerate whenever the blacksmith or tt-forge pin bumps.
4. **CI** — manually `workflow_dispatch` `on-nightly.yml` with `run-training` true (flip other `run-*` off) to run only forge build + P150 training job.

## Open TODOs
- Regenerate expected losses from a real P150 run (they're seed placeholders); tune `final_train_loss_max`.
- Fix/verify the tt-shield matrix `model` value (`meta-llama/Llama-3.1-8B` vs `Llama-3.1-8B`) so CI's `--model` passes argparse.
- Commit both branches when ready (nothing committed for the rename/registry fix yet).
- actionlint not run locally (not installed) — rely on PR gate for the tt-shield workflow YAML.
- Optionally add the routing (#1) and stub-server (#2) tests; optionally promote nightly → `release.yml`.

