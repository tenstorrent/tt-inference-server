# Quetzal model intake checklist (repeatable)

Copy-pasteable procedure for onboarding a **new Quetzal-generated model** to
tt-inference-server. It codifies the 10-step new-model checklist and adds the
Quetzal-specific facts that a native-vLLM model does not have: `impl: quetzal`
selection, the **pre-built immutable artifact package** (nothing is compiled at
serve time), and the tt-shield / CODEOWNER gates on nightly enrollment.

> Scope note: this file is the *durable template*. For the concrete war-room
> models (Qwen3.6-27B, gemma-4-31B-it, gpt-oss-120b) the per-step artifacts are
> already drafted in PRs #5042 / #5068 / #5077 — see the status table in the
> intake report rather than re-authoring them.

---

## Ground facts you must know before starting

### F1 — Quetzal serving loads a PRE-BUILT package; it never emits code

The CI/serve path does **not** run the Quetzal emit (no `compile_hf`, no
`generate generated.py`) and does **not** fetch/verify/materialize the package.
It loads an already-installed, content-addressed (sha256-root) immutable bundle.

Evidence:
- `docs/quetzal_dev_integration.md:48-53` — "TTIS does **not** fetch, verify, or
  materialize a Quetzal model package into that content store … Install the
  content-addressed bundle into the persistent cache before server startup with
  Quetzal's `ttq-artifact-bundle install`."
- `run.py` `--quetzal-models-root` help (`run.py:388-395`, branch `nkapre/quetzal`):
  "Host directory containing an installed Quetzal artifact bundle. Required with
  `--impl quetzal` and mounted **read-only** in Docker." `--quetzal-runtime-attestation`
  and `--quetzal-auxiliary-root` are likewise read-only mounts checked against a
  catalog SHA-256.
- The pre-staged package layout (`docs/quetzal_dev_integration.md:35-42`) already
  contains `compiled/<artifact>/full/{prefill,decode}/generated.py` +
  `metadata.json`, `compiled_weights/<weights>/full/weights.pt`, and
  `qualification_manifest.yaml`.

**Consequence for intake:** producing the package is an out-of-band Quetzal step.
Before any on-dispatch/nightly job can serve the model, an administrator must
publish the immutable package and it must be installed into `cache_root`
(`/home/container_app_user/cache_root/quetzal/packages/<package-id>/`). The
intake files below only *reference* that package by sha256; they do not build it.

### F2 — `impl: quetzal` resolves only under `--dev-mode` (dev catalog)

- `workflows/model_spec.py:1276` — `_MODEL_SPECS_ENV = os.getenv("MODEL_SPECS_ENV", "prod")`
  (default catalog is **prod**).
- `run.py:9-14` — `--dev-mode` sets `MODEL_SPECS_ENV=dev` before import; without it,
  the **prod** catalog loads.
- Quetzal rows live in `workflows/model_specs/dev/llm.yaml`; `prod/llm.yaml` has
  **zero** `impl: quetzal` rows. So a model is reachable in CI only if the
  dispatch passes `--dev-mode`, **or** the row is promoted to prod
  (`scripts/release/promote_dev_spec_to_prod.py`, fail-closed on the digest-pinned
  package-capable image).

### F3 — Nightly enrollment depends on tt-shield (private) + CODEOWNERS

- The Model Readiness nightly matrix is built and dispatched by **tt-shield**
  (`tests/test_model_naming.py:6`; `.github/ISSUE_TEMPLATE/model-readiness.yml:77`;
  `.github/workflows/release-automation.yml` requires a `tt-shield-run-id`).
- The impl-identity preservation in the matrix generator is staged on the private
  tt-shield branch `nkapre/quetzal @ 4c89e5a` and is **not merged/upstream**
  (`docs/quetzal_dev_integration.md:133-136`). Until it merges, adding a second
  same-engine `impl: quetzal` row can silently select the native default.
- `.github/workflows/models-ci-config.json` is CODEOWNER-gated
  (`.github/CODEOWNERS:30-32`, owners `@acvejicTT @vcankovicTT @mjeremicTT
  @mdobrosavljevicTT @vmaksimovicTT`) — you cannot self-approve step 8.

---

## The 10 steps

### 1) Intake and scope
Create/track a Model Request in the `MODEL` Jira project (Summary, Reporter,
Request Type, Program, Model, Category, Target Summary, Due date, Priority,
Hardware Config, Requirements link). Align on business context, target hardware,
delivery date.

### 2) Add model support — `workflows/model_specs/dev/llm.yaml` (DEV ONLY)
Edit the **dev** catalog only; never hand-edit `prod/llm.yaml`.
Add a template entry. For a Quetzal row, set `impl: quetzal` (the ImplSpec enum
`quetzal` is defined in `workflows/model_spec.py`; `code_path:
serving/quetzal_vllm.py`, bound in the vLLM process by the tt-plugin registration
when `QUETZAL_VLLM=1`). Minimum fields: `weights`, `impl`, version/commit pins,
`inference_engine`, `device_model_specs`, `status`. Per device:
`device`, `max_concurrency`, `max_context`, optional `override_tt_config`/`env_vars`.

```yaml
- weights:
    - <hf-org>/<Model>
  impl: quetzal
  inference_engine: VLLM
  model_type: LLM
  status: EXPERIMENTAL           # see step 7
  device_model_specs:
    - device: P300X2             # advertise the GENERATED artifact's envelope, not the checkpoint max
      max_concurrency: 1
      max_context: 4096
      default_impl: false        # keep the native impl the default
```

The advertised `max_context`/`max_concurrency` must be the generated artifact's
qualified/candidate envelope (e.g. an S4096 monolithic serve pair), not the
checkpoint's native maximum.

### 3) Performance targets — `reference_config/benchmarking/benchmark_targets/model_performance_reference.json`
**Skip-eligible for EXPERIMENTAL** (benchmarks are required only for status
higher than EXPERIMENTAL — PDF step 3). If you want a later status bump to be
unblocked, add a `theoretical` skeleton block for the model/device pair and mark
it clearly as a placeholder (mirrored from a same-device same-class anchor, not a
measured ceiling). Status thresholds (`functional=0.10, complete=0.50,
target=1.0`) come from the `perf_targets_map` default in `workflows/model_spec.py`
and are NOT stored in the JSON; override per device via `perf_targets_map` in the
YAML entry if needed.

### 4) Evals — `reference_config/evals/eval_config.py`
> Path note: the PDF says `evals/eval_config.py`; the real path is
> `reference_config/evals/eval_config.py`.

Add/append the model's eval config with **≥2 `EvalTask`s** where possible. Include
the published/reference score when available (put the graded published/reference
fields on the scored task; a bounded collection lane can carry no score of its
own). Wire venv via `WorkflowVenvType`, set `limit_samples_map` for
`CI_NIGHTLY`/`SMOKE_TEST`.

### 5) Tests / spec tests — Scenario A (shared) or Scenario B (own class)
Spec tests are live-server integration tests (LLM/VLM API param conformance).
Files:
- `test_module/server_tests_config.json`
- `test_module/test_suites/llm.json`
- `test_module/llm_tests/vllm_param_conformance_test.py`
- `llm_module/test_<name>.py`

**Scenario A — share the LLM conformance suite (3 edits):**
1. `server_tests_config.json` → `model_configs`: add
   `"<key>": {"id_name": "...", "weights": ["<Model>"], "category": "LLM",
   "compatible_devices": ["p300x2", ...]}`
2. `server_tests_config.json` → `model_categories.LLM`: append `"<Model>"`
3. `test_suites/llm.json` → existing matrix `[0]`: append `<key>` to `models`.

**Scenario B — model gets its own test class (Scenario B = PR #5068 pattern):**
Steps 1–2 as above, then:
- a. New class in `test_module/llm_tests/vllm_param_conformance_test.py`
  subclassing `VLLMParamConformanceTest` (set `KIND`, `PYTEST_FILENAME`,
  `ENDPOINT_PATH`, `REPORT_TASK_NAME`).
- b. `server_tests_config.json` → `test_templates`: key = the **exact Python class
  name** (`_instantiate_spec_test` does `getattr(module, case["name"])`), value
  `{"module": "test_module.llm_tests.vllm_param_conformance_test", "markers":[...],
  "test_config": {...}}`.
- c. `test_suites/llm.json` → **new** matrix entry `{"models": ["<key>"],
  "devices": [...], "test_cases": [{"template": "<ClassName>", "enabled": true,
  "description": "..."}]}`.
- d. The pytest file `llm_module/test_<name>.py`.

**Prefer ADDITIVE** (keep the model in the shared matrix *and* add its own matrix,
like `qwen3_32b`). EXCLUSIVE (own matrix only) silently drops any device you omit
from the private matrix to `NO MATCH → Spec Tests: NA` with no failure.

### 6) Validate on hardware
Pass ≥1 on-dispatch job on the target device and ≥1 nightly run before the
integration cut-off. **For Quetzal this requires the immutable package to be
published and installed (F1) and the digest-pinned Quetzal dev image built**
(`scripts/build_quetzal_dev_image.sh`). Admission is fail-closed: an
image/package/tt-metal-commit mismatch fails before a device is opened.

### 7) Set readiness/status
Set `status` in the dev entry. Use **EXPERIMENTAL** for bring-up — it removes the
benchmark requirement (step 3). Higher statuses (functional/complete/top) require
benchmarks satisfying the perf targets.

### 8) Nightly/release CI coverage — `.github/workflows/models-ci-config.json`
Encode the model under `models.<Model>.implementations[]`, each implementation
carrying a distinct `impl` selector. The Quetzal implementation additionally
pins the pre-built package by sha256:

```jsonc
{ "impl": "quetzal",
  "ci": { "nightly": true, "release": false },
  // ... additional-args mounts the installed immutable package, read-only:
  "additional-args": "--quetzal-models-root /mnt/.../quetzal/.../packages/sha256-<compiled-tree>-<weights-tree>" }
```

Do **not** alter the native nightly row; add Quetzal as a **second** implementation
row on the same device. **Blockers (F3):** the tt-shield matrix generator must
preserve same-engine impl identity (staged, unmerged), the immutable package + dev
image must exist, and CODEOWNERS must approve. Until all three hold, enrollment
either selects the native default or produces an expected infrastructure failure —
so gemma/gpt-style rows are staged as `productization/*_models_ci_enrollment.blocked.json`
until unblocked.

### 9) Release/process follow-up
Once integrated and nightly-green, request staging in the release process. Prod
promotion of a Quetzal row is fail-closed unless the operator supplies the
digest-pinned package-capable image
(`scripts/release/promote_dev_spec_to_prod.py --quetzal-docker-image
ghcr.io/...@sha256:<64-hex>`). Cherry-pick fixes to main and stable during release.

### 10) Practical default file list
- `workflows/model_specs/dev/llm.yaml`  (+ `workflows/model_spec.py` for the enum, once)
- `reference_config/evals/eval_config.py`
- `reference_config/benchmarking/benchmark_targets/model_performance_reference.json`
- `test_module/server_tests_config.json`, `test_module/test_suites/llm.json`,
  `test_module/llm_tests/vllm_param_conformance_test.py`, `llm_module/test_<name>.py`
- `.github/workflows/models-ci-config.json`  (CODEOWNER-gated)

## Done when
Model runs via `--impl quetzal --dev-mode`; evals defined (≥2); benchmarks defined
(or EXPERIMENTAL); spec tests defined; target device passes ≥1 on-dispatch job
(needs published+installed package); status set in dev catalog; nightly coverage
exists (needs tt-shield generator + CODEOWNER approval).
