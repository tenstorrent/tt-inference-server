---
name: extend-workflow-engine
description: Checklist for adding or modifying code in the workflow engine (run_workflows.py plus the llm_module/report_module/test_module/workflow_module packages) — new models, workflows, media runners, spec tests, report kinds, or CLI flags — and wiring it back to the  entry point run.py through workflows/workflow_dispatch.py. Use whenever code lands in those modules, when onboarding a model type to the engine, or when an engine feature "works standalone via run_workflows.py but not through run.py".
---

# Extending the workflow engine

`docs/workflow_development.md` is the authoritative guide. Read its
**"Adding things to the workflow engine"** section first (new workflow, media
runner, spec test, report kind) — this skill covers the v1-side wiring in
`workflows/workflow_dispatch.py` that the doc only mentions in passing, plus the
end-to-end verification people skip.

## How the two halves connect

Repo-root `run.py` is the only user-facing entry point. It does NOT import the
engine modules — it bridges through `workflows/workflow_dispatch.py`:

- `can_dispatch_to_engine(model_spec, runtime_config)` decides routing (`run.py:961`).
  If it returns false for a supported workflow, `run.py` raises rather than falling
  back to a legacy path.
- `build_engine_commands(...)` builds `VenvCommand`s that run `run_workflows.py`
  inside the `WorkflowVenvType.WORKFLOW_RUN_SCRIPT` venv.
- `run_workflows.py` re-imports v1's `workflows.model_spec.MODEL_SPECS` via a
  `sys.path` insert — the model catalog (`workflows/model_specs/prod/*.yaml`) is
  **shared**, not duplicated.

Recurring failure mode: code works when invoking `run_workflows.py` directly but is
unreachable from `run.py` because a bridge table or a forwarding helper below was
not updated.

## v1-side wiring checklist (the part people forget)

All in `workflows/workflow_dispatch.py` unless noted.

1. **Onboarding a model** → routing is now by **model type**, not per-name.
   If the model's `ModelType` is already in `_ENGINE_ROUTED_MODEL_TYPES`
   (frozenset, checked by `is_engine_routed_model`), a new model of that type is
   picked up automatically — just add it to the shared catalog
   `workflows/model_specs/prod/<type>.yaml` (or `dev/` for `--dev-mode`).
   A brand-new model *type* must be added to `_ENGINE_ROUTED_MODEL_TYPES`.
   (LLM/VLM route via the workflow-specific `_is_*_run` predicates in
   `can_dispatch_to_engine`, not the frozenset.)
2. **New workflow** → register it in the engine's `WORKFLOW_REGISTRY`
   (`workflow_module/workflows.py`), **and** map its `WorkflowType` to the engine
   workflow name in `_ENGINE_WORKFLOW_NAMES`. If no fitting `WorkflowType` exists,
   add one to the enum in `workflows/workflow_types.py` — otherwise it can never be
   selected via `run.py --workflow`.
3. **New CLI flag** → forwarding is explicit, not pass-through. Add the flag to
   `run.py` argparse, then forward it in the relevant `_forward_*` /
   `_build_*_cmd` helper (use `_extend_if_set(cmd, "--flag", value)`), and give
   `run_workflows.py` a matching `add_argument`. If a v1 flag is deliberately
   unsupported on the engine path, add it to `_warn_on_unsupported_args` so users
   get a warning instead of silent ignoring.
4. **Runner needs an extra host venv** → define a `WorkflowVenvType` + `VenvConfig`
   in `workflows/workflow_venvs.py` and map `model_type → venv` in
   `_ENGINE_EVAL_VENV_BY_MODEL_TYPE`. `_engine_dependency_venv_types` only
   provisions venvs listed there, and only for `_ENGINE_EVAL_WORKFLOWS`.

## Engine-side registries (quick map)

Details in `docs/workflow_development.md`; the registries to touch:

- **Workflows**: `WORKFLOW_REGISTRY` in `workflow_module/workflows.py`; optionally
  `ReleaseWorkflow.children` and a new `MediaTaskType` wired in `run_media_task`.
- **Media runners**: `BENCHMARK_DISPATCH` / `EVAL_DISPATCH` in
  `test_module/dispatch.py`, keyed by `model_spec.model_type.name`. The value is
  the runner's function *name* (string); `_resolve_runner` imports it lazily.
  Export it from `test_module/benchmark_tests/__init__.py` (or `eval_tests/`).
- **Spec tests**: data-driven via `test_module/test_suites/<category>.json`; guides
  in `test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md` and
  `TEST_MARKING_SYSTEM.md`. New test classes subclass `BaseTest` from
  `_test_common/base_test.py`.
- **Report kinds**: renderer in `report_module/renderers.py` (schema rules in
  `tests/report_module/SCHEMA_GUIDE.md`); acceptance in
  `report_module/acceptance_criteria.py` (`_check_<kind>` + category list in
  `acceptance_criteria_check`).

## Verify the wiring end-to-end

Always finish by exercising `run.py`, not `run_workflows.py` directly:

```bash
python run.py --model <model> --workflow <workflow> --device <device> ...
```

- The run log must contain `Model <name> (model_type=<TYPE>) routes through the
  workflow engine.` (`run.py:967`) followed by `Delegating workflow '<name>' to
  workflow engine.` (`workflow_dispatch.py:297`). If those are missing, the routing
  in step 1/2 is not wired.
- For flag changes, confirm the value survives the hop: inspect the engine
  subprocess command logged during the run, and `python run_workflows.py --help`
  for the receiving argparse entry.
