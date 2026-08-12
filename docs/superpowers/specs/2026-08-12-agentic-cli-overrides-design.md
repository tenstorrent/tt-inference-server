# Design: CLI overrides for agentic eval concurrency and task count

**Date:** 2026-08-12
**Status:** Approved
**Author:** brainstormed with Claude

## Problem

For agentic evals (`--workflow agentic`, and the agentic portion of `--workflow
release`), concurrency and dataset size are hardcoded in `evals/eval_config.py`:

- `TerminalBenchEvalConfig.n_concurrent_trials` / `SWEbenchEvalConfig.n_concurrent_trials`
  are fixed per model/task (e.g. 64 for Kimi-K2.7-Code `terminal_bench_2_1`).
- The task count comes from `n_tasks` (e.g. 89), or in CI mode
  (`--ci-mode` → `EvalLimitMode.CI_NIGHTLY`) from a pinned explicit task-name
  list in `task_names_map` (5 tasks for Kimi-K2.7-Code).

There is no way to run, say, 10 tasks at concurrency 5 without editing
`eval_config.py`. `--eval-samples` exists but is text/lm-eval only.

## Decision summary

Two new optional CLI flags on `run.py`, threaded through `RuntimeConfig` and
applied in the v2 agentic drivers (Approach A: RuntimeConfig passthrough,
matching the existing `eval_samples` / `limit_samples_mode` pattern).

- `--agentic-n-concurrent N` — overrides `n_concurrent_trials`.
- `--agentic-n-tasks N` — overrides the number of dataset tasks.

Semantics decided with the user:

1. **Explicit flag always wins.** `--agentic-n-tasks 10` combined with
   `--ci-mode` discards the CI pinned task-name list and takes 10 tasks from
   the full dataset. CLI override beats mode defaults.
2. **Naming** mirrors the underlying fields and Harbor's own CLI
   (`--n-concurrent`, `--n-tasks`), with an `agentic-` prefix for scope.

## Design

### 1. CLI arguments (`run.py`)

Two optional `type=int` flags, default `None` (no override → behavior is
unchanged):

```
--agentic-n-concurrent N   Override n_concurrent_trials for agentic evals
                           (terminal-bench / swe-bench). Takes precedence over
                           eval_config.py.
--agentic-n-tasks N        Override the number of dataset tasks for agentic
                           evals. Takes precedence over eval_config.py n_tasks
                           and over CI/limit-mode pinned task lists.
```

Validation: values must be >= 1, enforced with `parser.error`. No
mutual-exclusivity rules — the flags compose with `--ci-mode` /
`--limit-samples-mode` (the flag wins). Both values are added to the
runtime-config log dump in `run.py` so runs are self-documenting.

### 2. Plumbing (`workflows/runtime_config.py`)

Two new optional fields on `RuntimeConfig`:

```python
agentic_n_concurrent: Optional[int] = None
agentic_n_tasks: Optional[int] = None
```

Populated from parsed args in the same place `eval_samples` is. No v2-bridge
changes needed: `ctx.runtime_config` already reaches the agentic drivers.

### 3. Override application (`tt-inference-server-v2/llm_module/drivers/agentic.py`)

- `resolve_n_tasks(task, runtime_config)`: if
  `runtime_config.agentic_n_tasks` is set, return it immediately — it beats
  both `limit_samples_map[limit_mode]` and the config's `n_tasks`.
- `resolve_task_names(task, runtime_config)` and
  `resolve_instance_ids(task, runtime_config)`: when `agentic_n_tasks` is set,
  skip the limit-mode pinned list (`task_names_map` / `instance_ids_map`) so
  e.g. CI's 5 pinned Terminal-Bench names do not constrain the run. The base
  `task_names` list is kept — tau3-style configs use it as a dataset filter
  (wildcard), not as a limit.
- `build_terminal_bench_config()` and `build_swebench_config()`:
  `n_concurrent_trials` = `runtime_config.agentic_n_concurrent` when set, else
  the eval-config value. All `runtime_config` reads use `getattr(..., None)`
  defensive access, consistent with `_get_limit_mode`.

Both harness types (Terminal-Bench/Harbor and SWE-bench) and both entry paths
(standalone `agentic` workflow, `release` workflow) get the overrides
automatically because they share these builder/resolver functions.

### 4. Error handling

- `run.py` rejects 0 or negative values at argparse time.
- The flags are silent no-ops for non-agentic workflows, consistent with other
  workflow-specific flags.
- `resolve_n_tasks` keeps its existing `n_tasks == 0` skip path; the argparse
  validation means the override can never trigger it.

### 5. Testing

Unit tests alongside the existing v2 driver tests
(`tt-inference-server-v2/tests/`):

- No override set → all resolvers return current values (regression guard).
- Override + CI limit mode → `resolve_n_tasks` returns the flag value and
  `resolve_task_names` drops the pinned CI list (keeps base `task_names`).
- Override without limit mode → applied over the config `n_tasks`.
- `--agentic-n-concurrent` reaches `n_concurrent_trials` in both
  `build_terminal_bench_config` and `build_swebench_config` outputs.

## Out of scope

- Overriding lm-eval `max_concurrent` for text evals (names deliberately
  scoped `agentic-` so a future `--eval-concurrency` remains possible).
- Per-mode concurrency values in `eval_config.py`.
- Fixing the pre-existing `run.py` bug where `"--limit-samples-mode" not in
  args` always evaluates true (noted separately; not part of this change).

## Example

```
run.py --model Kimi-K2.7-Code --workflow agentic --device super_cluster \
  --server-url https://pd-k27-q9-ngrok.n.cloud.tenstorrent.com:443 \
  --skip-system-sw-validation --ci-mode \
  --agentic-n-concurrent 5 --agentic-n-tasks 10
```

Runs 10 Terminal-Bench tasks from the full dataset at concurrency 5, even
though CI mode would otherwise pin 5 named tasks at configured concurrency 64.
