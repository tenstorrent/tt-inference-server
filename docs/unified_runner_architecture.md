# Unified runner architecture

Status: **in progress** — Phases A, B1, and B2 have landed: `run.py` drives one
`WorkflowRunner` over `[ServerCommand, *build_engine_commands(...)]`, and
`workflow_dispatch.py` is a pure command builder. Phase C (fold in
`run_workflows.py`, drop the `v1`/`v2` naming) is planned.

## Why

After `tt-inference-server-v2/` was flattened into the repo root there is one
repo, but the code still carries the migration-era split it was born with:

- `run.py` is the full CLI. It does host setup, device inference, and **server
  bring-up** (`--local-server` / `--docker-server`), then for onboarded models
  **shells out** to `run_workflows.py` through the subprocess bridge in
  `workflows/workflow_dispatch.py`.
- `run_workflows.py` is a second, workflow-only CLI. Its own help text says it
  "drives a workflow against an **already-running** server… for full server
  bring-up, invoke through run.py instead." `WorkflowRunner.run()` just executes
  a `Sequence[Command]` built by `CommandFactory`.

So server bring-up and the workflow runner sit on opposite sides of a subprocess
wall, and the `v1`/`v2` labels describe *which side of that wall* code lives on
rather than what it does. The label is a versioning artifact, not a
description — there is no user-visible "version 2".

## Target architecture

One entry point, one runner, one command list. Server bring-up becomes just
another command in that list:

```
run.py                      ← the one entry point: parse args, host setup, device inference
  └─ CommandFactory.build(args, server_launch=…) → List[Command]
        [ ServerCommand,          # bring up docker/local server, wait healthy
          WorkflowCommand(…),     # run benchmarks / evals / spec_tests / …
          SummaryCommand ]        # aggregate reports
  └─ WorkflowRunner(commands).run()   ← executes the sequence in order
```

- **`ServerCommand`** wraps `workflows.run_docker_server` /
  `workflows.run_local_server`. Server bring-up stops being special-cased logic
  ahead of the runner and becomes the first command in the list — this is the
  seam that lets one runner "launch the server *and* run workflows."
- **`WorkflowRunner`** already takes `Sequence[Command]` and executes each in
  order, stopping on the first failure. It needs no conceptual change; it just
  receives a longer list.
- **`run_workflows.py`** stops being a second CLI. Its arg parsing folds into
  `run.py` (or a shared parser module); it can survive as a thin debug entry
  that builds the same command list.
- **`workflows/workflow_dispatch.py`** is reduced to a pure command **builder**
  (`build_engine_commands` + `can_dispatch_to_engine`); its separate
  subprocess-execution role is gone (run.py's single runner drives the commands).
  The old bridge helpers (`_ensure_v2_venv`, `_base_v2_cmd`,
  `_dispatch_via_engine_venv`, `run_llm_benchmark_workflow`, …) are deleted.

## The one hard constraint: venvs

The reason the runner runs as a subprocess today is that different workflows
need incompatible venvs (`V2_PREFIX_CACHE`, `V2_LLM_VLLM`, …). `run.py`'s
interpreter cannot be all of them at once, so `run_workflows.py` `os.execv`s
into the right one. Collapsing in-process must preserve that isolation.

The plan is to **push venv isolation down to the Command level**: a command that
needs a specific environment declares it and execs its script in that venv's
interpreter as a subprocess. Orchestration stays in one process (`run.py` →
`WorkflowRunner`) while individual heavy steps still run isolated. That is what
lets one runner drive server + N workflows without a single mega-venv or a
top-level subprocess bridge. `VenvCommand` (Phase B2) is that primitive.

## Migration (strangler, not big-bang)

### Phase A — introduce `ServerCommand` (done)

`ServerCommand` + `ServerLaunchSpec` added to `workflow_module/commands.py`;
`CommandFactory.build(args, server_launch=…)` prepends a `ServerCommand` when a
launch spec is supplied. Purely additive: with no launch spec the command list
is byte-for-byte what it was, so `run_workflows.py` and the existing bridge are
untouched. It establishes the abstraction and the wiring point independently of
any behavior change.

### Phase B1 — unify server bring-up under the runner (done)

Three changes, all verified by unit tests, no hardware needed:

- **`WorkflowRunner` moved into the package** (`workflow_module/runner.py`,
  exported from `workflow_module`). It was stranded in the `run_workflows.py`
  script; now any entry point can drive it. `run_workflows.py` imports it.
- **The command model is import-light.** `workflow_module/commands.py` imports
  `MediaContext` / `OrchestratorMetadata` / `WorkflowResult` only under
  `TYPE_CHECKING` (they were annotation-only, and `from __future__ import
  annotations` keeps them unevaluated). Importing `ServerCommand` /
  `WorkflowRunner` / `VenvCommand` no longer pulls the heavy
  `test_module.context` / `report_module` stack, so `run.py` can import them
  safely in its base venv.
- **`run.py` brings the server up through the runner.** The direct
  `run_docker_server` / `run_local_server` calls are replaced by a
  `ServerLaunchSpec` executed via `WorkflowRunner([ServerCommand(spec)])`. The
  runner now owns server bring-up. Behavior note: a bring-up failure is now
  caught and surfaced as a non-zero return code (with the traceback logged),
  matching every other command, instead of propagating as an exception.

This works because the workflow *core* (`report_module`, `workflow_module`) is
pure in-repo Python with no external deps, importable in any venv; the heavy
model stack stays behind `test_module`'s lazy facade and only loads when a
workflow actually runs. `ServerCommand` itself only imports the launchers
(`workflows.run_docker_server` / `run_local_server`) that `run.py` already used.

### Phase B2 — route workflow execution in-process

**Mechanism (done):** `VenvCommand` in `workflow_module/commands.py`. It takes a
`WorkflowVenvType`, an argv, an optional env, and a `model_spec`; on `execute()`
it provisions the venv (idempotent) and execs `[venv_python, *argv]` as a
subprocess, returning the exit code as a `CommandResult`. All `workflows.*`
imports are deferred to `execute()` so the module stays import-light. This is the
per-Command venv-exec primitive the doc's constraint called for, unit-tested in
`tests/workflow_module/test_commands.py`.

**Generic path (done):** the largest `workflow_dispatch.py` branch — image /
video / audio / tts / cnn / embedding plus LLM `evals` / `release` /
`spec_tests`, i.e. everything that ran `run_workflows.py` in the `V2_RUN_SCRIPT`
venv — now builds a `VenvCommand` and runs it through a `WorkflowRunner`
(`_dispatch_via_engine_venv` + the pure `_engine_run_argv` builder), replacing
the hand-rolled `run_command`. `_ensure_v2_venv` is gone (subsumed by the
command's own venv setup). Behavior-equivalent to the old path: same venv, same
argv, same `TT_V1_RUN_COMMAND` env. Verified by unit tests
(`tests/workflows/test_workflow_dispatch_routing.py` now assert on the built
`VenvCommand.argv`) and on real hardware (the command provisions `V2_RUN_SCRIPT`
and execs `run_workflows.py`).

**Launcher branches (done):** `agentic`, `prefix_cache`, `spec_decode`, the
LLM-benchmark launcher, and `stress_tests` also run as `VenvCommand`s. Two shapes
cover every branch:

- `venv_type=<WorkflowVenvType>` — runs the script in that venv's interpreter,
  provisioning it (plus any `dependency_venvs`) first (generic → `V2_RUN_SCRIPT`,
  stress → `STRESS_TESTS_RUN_SCRIPT`).
- `venv_type=None` — runs in the *current* interpreter (`sys.executable`); used
  for the self-re-execing launchers (`run_agentic.py` / `run_prefix_cache.py` /
  `run_spec_decode.py` / `run_llm_bench.py`), which switch into their own venv
  internally.

`_base_v2_cmd` → `_base_engine_argv` and `_build_stress_cmd` → `_stress_argv`
return argv without the leading interpreter (the command prepends it).

**run.py drives one runner (done):** `run.py` now builds a single command list —
`ServerCommand` (if bringing up a server) followed by
`build_engine_commands(...)` — and executes it with one `WorkflowRunner`. The
runner stops at the first failure, so a failed server bring-up aborts before the
workflow runs. `workflow_dispatch.py` is reduced to the pure builder
`build_engine_commands` (returns the `VenvCommand`(s), no execution, no
provisioning — that moved into `VenvCommand.execute` via `dependency_venvs`, so a
failed server never triggers workflow-venv provisioning). A thin
`dispatch_workflows` wrapper (build + run) remains for standalone callers. The
`run_command`/`os` imports and `_ensure_v2_venv` / `_ensure_v2_dependency_venvs`
/ `_base_v2_cmd` / `_dispatch_via_engine_venv` / `run_llm_benchmark_workflow`
helpers are gone. Verified by unit tests and hardware (the generic path runs
through run.py's single runner on bh-qbge-08; the `ServerCommand` success path was
validated separately in B1).

The self-re-exec launcher branches weren't run end-to-end on hardware (each needs
minutes of tool-venv provisioning for a path that reuses the already-validated
primitive) — worth a real run before relying on them.

**What "deleting the subprocess bridge" means here:** the *separate* execution
layer (run.py bringing up the server, then a second engine that shells out) is
gone — one runner owns the whole run. The per-`VenvCommand` subprocesses remain:
that's the venv isolation the hard constraint requires, now expressed as
first-class commands rather than a hand-rolled bridge.

### Phase C — collapse & rename

Fold `run_workflows.py` into a shared parser; drop the `v1`/`v2` language. The
venv enum (`V2_RUN_SCRIPT` → `RUN_SCRIPT` / `WORKFLOW_RUN_SCRIPT`) and
`requirements/v2-*.txt` get their descriptive names here — the real artifacts
survive the refactor, the bridge symbols simply disappear.

## Note on naming cleanup

Do **not** do a standalone `v2 → workflow` rename ahead of this work. The
remaining bridge-internal `v2` identifiers (`_V2_WORKFLOW_NAMES`,
`_V2_ROUTED_MODEL_TYPES`, `_ensure_v2_dependency_venvs`, …) live in the routing
layer that Phase B2 collapses, so renaming them is wasted churn. (The B2 work so
far already retired `_ensure_v2_venv` and `_base_v2_cmd` as a side effect.) The
only `v2` names that outlive the refactor are the venv enum (`V2_RUN_SCRIPT`, …)
and the `requirements/v2-*.txt` files, renamed for free in Phase C. Let the
naming fall out of the architecture work.
