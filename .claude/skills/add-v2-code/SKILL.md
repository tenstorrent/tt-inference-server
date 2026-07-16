---
name: add-v2-code
description: Checklist for adding new code (models, workflows, runners, tests, CLI flags) to the v2 workflow engine (run_workflows.py plus the llm_module/report_module/test_module/workflow_module packages) and wiring it back into the v1 entry point run.py via workflows/workflow_dispatch.py. Use whenever code is being added or modified in those modules, when routing a model from v1 to v2, or when a v2 feature "works standalone but not through run.py".
---

# Adding code to the v2 workflow engine

## How the two halves connect

Repo-root `run.py` (v1) is the only user-facing entry point. It does NOT import v2
modules — it delegates through `workflows/workflow_dispatch.py`:

- `can_dispatch_to_engine(model_spec, runtime_config)` decides routing: the model name must
  be in `_V2_ROUTED_MODELS` **and** the v1 `WorkflowType` must have an entry in
  `_V2_WORKFLOW_NAMES`.
- `dispatch_workflows(...)` provisions the `WorkflowVenvType.V2_RUN_SCRIPT` venv
  (`workflows/workflow_venvs.py`), builds an explicit CLI command, and shells out to
  `run_workflows.py` as a subprocess.
- `run_workflows.py` re-imports v1's `workflows.model_spec.MODEL_SPECS` via a `sys.path`
  insert — the model catalog (`workflows/model_specs/prod/*.yaml`) is **shared**, not
  duplicated.

The recurring failure mode: code lands in v2, works when invoking
`run_workflows.py` directly, but is unreachable from the real entry point
because one of the bridge tables or the arg-forwarding list below was not updated.

## Read first

`docs/workflow_development.md` → section **"Adding things to v2"** is the
authoritative guide for the v2-internal steps (new workflow, media runner, spec test,
report kind). Follow it for the v2 side; this skill adds the v1-side wiring it only
mentions in passing.

## v1-side wiring checklist (the part people forget)

All in `workflows/workflow_dispatch.py` unless noted. After any v2 change, walk this list:

1. **New model on v2** → add its `model_name` to `_V2_ROUTED_MODELS`. Without this,
   v1 silently runs the legacy v1 path (or fails) — no error points at the bridge.
   The model must also exist in the shared catalog `workflows/model_specs/prod/<type>.yaml`
   (or `dev/` for `--dev-mode`); v2 `run.py` builds its `--model` choices from it.
2. **New v2 workflow** → register it in v2's `WORKFLOW_REGISTRY`
   (`workflow_module/workflows.py`) per the README, **and** map
   it in `_V2_WORKFLOW_NAMES` from a v1 `WorkflowType`. If no fitting `WorkflowType`
   exists, add one to the enum in `workflows/workflow_types.py` — otherwise the
   workflow can never be selected via `run.py --workflow`.
3. **New CLI flag** → forwarding is explicit, not pass-through. A flag added to v1
   `run.py` argparse reaches v2 only if `dispatch_workflows` appends it to `cmd`
   (see the existing `sdxl_num_prompts` → `--num-prompts` translation), **and** v2
   `run.py` has a matching `add_argument`. If a v1 flag is deliberately not supported
   on the v2 path, add it to `_warn_on_unsupported_args` so users get a warning
   instead of silent ignoring.
4. **Runner needs an extra venv on the host** (e.g. a special eval harness) → define a
   `WorkflowVenvType` + `VenvConfig` in `workflows/workflow_venvs.py` and map
   `model_type → venv` in `_V2_EVAL_VENV_BY_MODEL_TYPE` (pattern:
   `ModelType.AUDIO: WorkflowVenvType.EVALS_AUDIO`). `_ensure_v2_dependency_venvs`
   only provisions venvs listed there, and currently only for evals/release
   workflows (`_V2_EVAL_WORKFLOWS`).

## v2-side registries (quick map)

Details in the v2 README; the registries to touch are:

- **Workflows**: `WORKFLOW_REGISTRY` in `workflow_module/workflows.py`;
  optionally `ReleaseWorkflow.children` and `MediaTaskType` dispatch in `run_media_task`.
- **Media runners** (evals/benchmarks per model type): `EVAL_DISPATCH` /
  `BENCHMARK_DISPATCH` in `test_module/dispatch.py`, keyed by
  `model_spec.model_type.name` (e.g. `"IMAGE"`, `"AUDIO"`). The value is the runner's
  function *name* (string); `_resolve_runner` imports it lazily. Export the runner
  through `test_module/benchmark_tests/__init__.py` (or `eval_tests/`) so it resolves.
- **Spec tests**: data-driven via `test_module/test_suites/<category>.json`; guides in
  `test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md` and
  `TEST_MARKING_SYSTEM.md`.
- **Report kinds**: renderer in `report_module/renderers.py` (schema rules in
  `tests/report_module/SCHEMA_GUIDE.md`), acceptance in
  `report_module/acceptance_criteria.py` (`_check_<kind>` + category list in
  `acceptance_criteria_check`).

## Verify the wiring end-to-end

Always finish by exercising the v1 entry point, not v2 directly:

```bash
python run.py --model <model> --workflow <workflow> --device <device> ...
```

- The run log must contain `routes through v2 engine` (emitted by `run.py` when
  `can_dispatch_to_engine` returns true) followed by `Delegating image-model workflow ... to
  v2 engine.`. If those lines are missing, the bridge tables in step 1/2 are not wired.
- For flag changes, confirm the value survives the hop: check the v2 subprocess
  command logged by `run_command`, and `python run_workflows.py --help`
  for the receiving argparse entry.
