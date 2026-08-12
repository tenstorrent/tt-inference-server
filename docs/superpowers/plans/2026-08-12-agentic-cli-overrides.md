# Agentic CLI Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--agentic-n-concurrent` and `--agentic-n-tasks` flags to `run.py` that override agentic-eval concurrency and dataset task count, beating `eval_config.py` values and CI-mode pinned task lists.

**Architecture:** Two optional int flags parsed in v1 `run.py`, stored as new `RuntimeConfig` fields. `RuntimeConfig` crosses the v1→v2 subprocess boundary via the runtime-model-spec JSON (`to_dict()`/`from_dict()`, unknown keys ignored — verified in `workflows/v2_bridge.py` `_build_agentic_cmd` → v2 `command_factory._load_runtime_config`), so **no bridge or v2-argparse changes are needed**. The overrides are applied in `tt-inference-server-v2/llm_module/drivers/agentic.py` where the Harbor/SWE-bench run configs are built.

**Tech Stack:** Python 3, argparse, dataclasses, pytest.

**Spec:** `docs/superpowers/specs/2026-08-12-agentic-cli-overrides-design.md`

## Global Constraints

- Flag values must be >= 1; reject with `parser.error` (exit code 2).
- Explicit flag always wins: `--agentic-n-tasks` beats `limit_samples_map` AND drops limit-mode pinned lists (`task_names_map` / `instance_ids_map`), but base `task_names` is kept (dataset filter, e.g. tau3 wildcards).
- Default `None` = zero behavior change (regression guard tests required).
- All `runtime_config` reads in v2 drivers use `getattr(..., None)` defensive access (matches existing `_get_limit_mode` style).
- Pre-commit runs ruff + pytest on commit; commits must pass it.
- SPDX headers already exist in all touched files; do not remove them.

---

### Task 1: v1 CLI flags + RuntimeConfig fields

**Files:**
- Modify: `run.py` (arg defs near line 254-262; validation near line 600; log dump near line 743-744)
- Modify: `workflows/runtime_config.py` (fields near line 70; `from_args` near line 159)
- Test: `tests/test_run_arguments.py` (new class `TestAgenticOverrideArgs`)

**Interfaces:**
- Produces: `args.agentic_n_concurrent: Optional[int]`, `args.agentic_n_tasks: Optional[int]` on the parsed namespace; `RuntimeConfig.agentic_n_concurrent: Optional[int]`, `RuntimeConfig.agentic_n_tasks: Optional[int]` fields that survive `to_dict()`/`from_dict()` (this is what Task 2's drivers read).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_arguments.py` (after class `TestArgumentParsing`, matching its patterns):

```python
class TestAgenticOverrideArgs:
    """--agentic-n-concurrent / --agentic-n-tasks parsing and plumbing."""

    def test_defaults_are_none(self, base_args):
        with patch("sys.argv", ["run.py"] + base_args):
            args = parse_arguments()
        assert args.agentic_n_concurrent is None
        assert args.agentic_n_tasks is None

    def test_values_parsed_as_ints(self, base_args):
        full_args = base_args + [
            "--agentic-n-concurrent",
            "5",
            "--agentic-n-tasks",
            "10",
        ]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()
        assert args.agentic_n_concurrent == 5
        assert args.agentic_n_tasks == 10

    @pytest.mark.parametrize("flag", ["--agentic-n-concurrent", "--agentic-n-tasks"])
    @pytest.mark.parametrize("bad_value", ["0", "-3"])
    def test_rejects_non_positive_values(self, base_args, flag, bad_value, capsys):
        full_args = base_args + [flag, bad_value]
        with patch("sys.argv", ["run.py"] + full_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert flag in captured.err
        assert "must be >= 1" in captured.err

    def test_runtime_config_from_args_and_round_trip(self, base_args):
        full_args = base_args + [
            "--agentic-n-concurrent",
            "5",
            "--agentic-n-tasks",
            "10",
        ]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()
        runtime_config = RuntimeConfig.from_args(args)
        assert runtime_config.agentic_n_concurrent == 5
        assert runtime_config.agentic_n_tasks == 10
        # Same mechanism the v2 bridge uses: to_dict -> JSON doc -> from_dict.
        restored = RuntimeConfig.from_dict(runtime_config.to_dict())
        assert restored.agentic_n_concurrent == 5
        assert restored.agentic_n_tasks == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/acvejic/tt/tt-inference-server && python -m pytest tests/test_run_arguments.py::TestAgenticOverrideArgs -v`
Expected: FAIL — `AttributeError: ... 'agentic_n_concurrent'` (unrecognized argument / missing attribute).

- [ ] **Step 3: Implement run.py flags + validation + log dump**

In `run.py`, after the `--eval-samples` `add_argument` block (ends line 262):

```python
    parser.add_argument(
        "--agentic-n-concurrent",
        type=int,
        default=None,
        help="Override n_concurrent_trials for agentic evals "
        "(terminal-bench / swe-bench). Takes precedence over eval_config.py.",
    )
    parser.add_argument(
        "--agentic-n-tasks",
        type=int,
        default=None,
        help="Override the number of dataset tasks for agentic evals. Takes "
        "precedence over eval_config.py n_tasks and over CI/limit-mode pinned "
        "task lists.",
    )
```

After the `--eval-samples`/`--limit-samples-mode` mutual-exclusion check (line ~601):

```python
    for flag, value in (
        ("--agentic-n-concurrent", args.agentic_n_concurrent),
        ("--agentic-n-tasks", args.agentic_n_tasks),
    ):
        if value is not None and value < 1:
            parser.error(f"{flag} must be >= 1 (got {value})")
```

In the CLI-args summary (after the `eval_samples` line, ~744):

```python
        f"  agentic_n_concurrent:       {runtime_config.agentic_n_concurrent}",
        f"  agentic_n_tasks:            {runtime_config.agentic_n_tasks}",
```

- [ ] **Step 4: Implement RuntimeConfig fields**

In `workflows/runtime_config.py`, after `eval_samples: Optional[str] = None` (line 70):

```python
    agentic_n_concurrent: Optional[int] = None
    agentic_n_tasks: Optional[int] = None
```

In `from_args`, after `eval_samples=args.eval_samples,` (line 159):

```python
            agentic_n_concurrent=getattr(args, "agentic_n_concurrent", None),
            agentic_n_tasks=getattr(args, "agentic_n_tasks", None),
```

(`getattr` because other callers build partial namespaces — same pattern as `goodput`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_arguments.py::TestAgenticOverrideArgs -v`
Expected: PASS (all 8).

- [ ] **Step 6: Run the whole file as regression guard**

Run: `python -m pytest tests/test_run_arguments.py -v`
Expected: PASS (no existing test broken).

- [ ] **Step 7: Commit**

```bash
git add run.py workflows/runtime_config.py tests/test_run_arguments.py
git commit -m "feat: add --agentic-n-concurrent / --agentic-n-tasks CLI flags

Parsed in run.py, validated >= 1, stored on RuntimeConfig so they reach
the v2 agentic drivers via the runtime-model-spec JSON document."
```

---

### Task 2: Apply overrides in v2 agentic drivers

**Files:**
- Modify: `tt-inference-server-v2/llm_module/drivers/agentic.py` (resolvers lines 249-292, builders lines 189, 229, `__all__` line 312)
- Test: `tt-inference-server-v2/tests/test_module/llm_tests/test_agentic_eval_tests.py` (extend `_runtime` helper line 95-96; new class `TestAgenticCliOverrides`)

**Interfaces:**
- Consumes: `RuntimeConfig.agentic_n_concurrent` / `RuntimeConfig.agentic_n_tasks` from Task 1 (read via `getattr`, so tests may use `SimpleNamespace`).
- Produces: `resolve_n_concurrent(cfg, runtime_config) -> int` (new, exported); changed behavior of `resolve_n_tasks`, `resolve_task_names`, `resolve_instance_ids` when the override is set. `build_terminal_bench_config` / `build_swebench_config` signatures unchanged.

- [ ] **Step 1: Extend the `_runtime` test helper**

In `test_agentic_eval_tests.py` replace lines 95-96:

```python
def _runtime(
    limit_samples_mode: Optional[str] = None,
    agentic_n_concurrent: Optional[int] = None,
    agentic_n_tasks: Optional[int] = None,
):
    return SimpleNamespace(
        limit_samples_mode=limit_samples_mode,
        agentic_n_concurrent=agentic_n_concurrent,
        agentic_n_tasks=agentic_n_tasks,
    )
```

- [ ] **Step 2: Write the failing tests**

Add to `test_agentic_eval_tests.py` (near `TestAgenticDriverConfigMapping`; note `FakeTerminalBenchConfig` defaults: `n_concurrent_trials=5`, `n_tasks=89`). Import `resolve_n_concurrent` in the existing `llm_module.drivers.agentic` import block (line 16-22):

```python
class TestAgenticCliOverrides:
    """--agentic-n-tasks / --agentic-n-concurrent override semantics."""

    def test_n_tasks_override_beats_limit_mode_and_config(self):
        task = _terminal_task()
        task.agentic_eval_config.task_names_map = {
            EvalLimitMode.CI_NIGHTLY: ["terminal-bench/pinned-a"]
        }
        rt = _runtime("ci-nightly", agentic_n_tasks=10)
        assert resolve_n_tasks(task, rt) == 10
        # Pinned CI list dropped; base task_names (empty) kept.
        assert resolve_task_names(task, rt) == []

    def test_n_tasks_override_keeps_base_task_names_filter(self):
        task = _terminal_task()
        task.agentic_eval_config.task_names = ["dataset/wildcard-*"]
        task.agentic_eval_config.task_names_map = {
            EvalLimitMode.CI_NIGHTLY: ["dataset/pinned"]
        }
        rt = _runtime("ci-nightly", agentic_n_tasks=10)
        assert resolve_task_names(task, rt) == ["dataset/wildcard-*"]

    def test_no_override_preserves_limit_mode_behavior(self):
        task = _terminal_task()
        task.agentic_eval_config.task_names_map = {
            EvalLimitMode.CI_NIGHTLY: ["dataset/pinned"]
        }
        rt = _runtime("ci-nightly")
        assert resolve_task_names(task, rt) == ["dataset/pinned"]
        assert resolve_n_tasks(task, rt) == 89

    def test_swebench_instance_ids_dropped_on_override(self):
        task = _swebench_task()
        task.swebench_eval_config.instance_ids_map = {
            EvalLimitMode.CI_NIGHTLY: ["django__django-11299"]
        }
        rt = _runtime("ci-nightly", agentic_n_tasks=10)
        assert resolve_n_tasks(task, rt) == 10
        assert resolve_instance_ids(task, rt) == []

    def test_swebench_no_override_preserves_instance_ids(self):
        task = _swebench_task()
        task.swebench_eval_config.instance_ids_map = {
            EvalLimitMode.CI_NIGHTLY: ["django__django-11299"]
        }
        rt = _runtime("ci-nightly")
        assert resolve_instance_ids(task, rt) == ["django__django-11299"]

    def test_concurrency_override_reaches_terminal_bench_config(self):
        task = _terminal_task()
        cfg = build_terminal_bench_config(
            task,
            _server(),
            _driver_context(),
            runtime_config=_runtime(agentic_n_concurrent=3),
        )
        assert cfg.n_concurrent_trials == 3

    def test_concurrency_default_without_override(self):
        task = _terminal_task()
        cfg = build_terminal_bench_config(
            task,
            _server(),
            _driver_context(),
            runtime_config=_runtime(),
        )
        assert cfg.n_concurrent_trials == 5

    def test_concurrency_override_reaches_swebench_config(self):
        task = _swebench_task()
        cfg = build_swebench_config(
            task,
            _server(),
            _driver_context(),
            runtime_config=_runtime(agentic_n_concurrent=3),
        )
        assert cfg.n_concurrent_trials == 3

    def test_none_runtime_config_unchanged(self):
        task = _terminal_task()
        assert resolve_n_tasks(task, None) == 89
        assert resolve_task_names(task, None) == []
        cfg = build_terminal_bench_config(task, _server(), _driver_context())
        assert cfg.n_concurrent_trials == 5
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/acvejic/tt/tt-inference-server/tt-inference-server-v2 && python -m pytest tests/test_module/llm_tests/test_agentic_eval_tests.py::TestAgenticCliOverrides -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_n_concurrent'`.

- [ ] **Step 4: Implement driver overrides**

In `tt-inference-server-v2/llm_module/drivers/agentic.py`:

Add below `_get_limit_mode` (line 287-292):

```python
def _get_agentic_override(runtime_config: Any, attr: str) -> Optional[int]:
    """CLI override (--agentic-n-concurrent / --agentic-n-tasks) or None."""
    value = getattr(runtime_config, attr, None) if runtime_config is not None else None
    return int(value) if value is not None else None


def resolve_n_concurrent(cfg: Any, runtime_config: Any = None) -> int:
    override = _get_agentic_override(runtime_config, "agentic_n_concurrent")
    if override is not None:
        return override
    return cfg.n_concurrent_trials
```

Rewrite `resolve_n_tasks` (line 269) — only the two override lines at the top are new; the rest is the current body verbatim:

```python
def resolve_n_tasks(task: Any, runtime_config: Any = None) -> Optional[int]:
    override = _get_agentic_override(runtime_config, "agentic_n_tasks")
    if override is not None:
        return override
    agentic_config = task.agentic_eval_config or task.swebench_eval_config
    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode is None:
        return agentic_config.n_tasks if agentic_config else None

    limit_arg = task.limit_samples_map.get(limit_mode)
    if limit_arg is None:
        return agentic_config.n_tasks if agentic_config else None
    if isinstance(limit_arg, float) and limit_arg < 1:
        logger.warning(
            "Agentic eval limits are task counts, not fractions; using one task for %s",
            task.task_name,
        )
        return 1
    return int(limit_arg)
```

Rewrite `resolve_task_names` (line 249) — the override drops the limit-mode
pinned list but keeps base `task_names` (dataset filter, e.g. tau3 wildcards):

```python
def resolve_task_names(task: Any, runtime_config: Any = None) -> List[str]:
    agentic_config = task.agentic_eval_config
    if agentic_config is None:
        return []
    # --agentic-n-tasks replaces the limit-mode pinned list: the run takes
    # n_tasks from the full dataset instead of the pinned subset.
    if _get_agentic_override(runtime_config, "agentic_n_tasks") is None:
        limit_mode = _get_limit_mode(runtime_config)
        if limit_mode is not None and limit_mode in agentic_config.task_names_map:
            return agentic_config.task_names_map[limit_mode]
    return agentic_config.task_names
```

Rewrite `resolve_instance_ids` (line 259) the same way:

```python
def resolve_instance_ids(task: Any, runtime_config: Any = None) -> List[str]:
    swebench_config = task.swebench_eval_config
    if swebench_config is None:
        return []
    if _get_agentic_override(runtime_config, "agentic_n_tasks") is None:
        limit_mode = _get_limit_mode(runtime_config)
        if limit_mode is not None and limit_mode in swebench_config.instance_ids_map:
            return swebench_config.instance_ids_map[limit_mode]
    return []
```

In `build_swebench_config` replace `n_concurrent_trials=cfg.n_concurrent_trials,` (line 189) with `n_concurrent_trials=resolve_n_concurrent(cfg, runtime_config),`.
In `build_terminal_bench_config` replace `n_concurrent_trials=cfg.n_concurrent_trials,` (line 229) the same way.

Add `"resolve_n_concurrent",` to `__all__` (line 312, keep alphabetical order).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_module/llm_tests/test_agentic_eval_tests.py::TestAgenticCliOverrides -v` (from `tt-inference-server-v2/`)
Expected: PASS (all 9).

- [ ] **Step 6: Run the whole file + workflow test as regression guard**

Run: `python -m pytest tests/test_module/llm_tests/test_agentic_eval_tests.py tests/workflow_module/test_agentic_workflow.py -v` (from `tt-inference-server-v2/`)
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tt-inference-server-v2/llm_module/drivers/agentic.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_agentic_eval_tests.py
git commit -m "feat: apply agentic CLI overrides in v2 agentic drivers

--agentic-n-tasks beats limit_samples_map and drops CI pinned task
lists (base task_names dataset filters are kept); --agentic-n-concurrent
overrides n_concurrent_trials for terminal-bench and swe-bench."
```

---

### Task 3: End-to-end wiring verification

**Files:** none created/modified (verification only; fix-forward if anything fails).

**Interfaces:**
- Consumes: everything from Tasks 1-2.

- [ ] **Step 1: Verify flags visible from the real entry point**

Run: `cd /Users/acvejic/tt/tt-inference-server && python run.py --help 2>&1 | grep -A2 "agentic-n"`
Expected: both flags with their help text.

- [ ] **Step 2: Verify the v1→v2 JSON hop end-to-end in one command**

Run:

```bash
python - <<'EOF'
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
from run import parse_arguments
from workflows.runtime_config import RuntimeConfig

argv = [
    "run.py", "--model", "Mistral-7B-Instruct-v0.3", "--workflow", "agentic",
    "--tt-device", "n150", "--agentic-n-concurrent", "5", "--agentic-n-tasks", "10",
]
with patch("sys.argv", argv):
    args = parse_arguments()
rc = RuntimeConfig.from_args(args)
restored = RuntimeConfig.from_dict(rc.to_dict())
assert restored.agentic_n_concurrent == 5, restored.agentic_n_concurrent
assert restored.agentic_n_tasks == 10, restored.agentic_n_tasks
print("v1->v2 JSON hop OK:", restored.agentic_n_concurrent, restored.agentic_n_tasks)
EOF
```

Expected: `v1->v2 JSON hop OK: 5 10`.

- [ ] **Step 3: Full test sweep of touched areas**

Run: `python -m pytest tests/test_run_arguments.py -q && (cd tt-inference-server-v2 && python -m pytest tests/test_module/llm_tests/test_agentic_eval_tests.py -q)`
Expected: all PASS.

- [ ] **Step 4: Cross-check plan against spec**

Re-read `docs/superpowers/specs/2026-08-12-agentic-cli-overrides-design.md`; confirm every requirement (flags, validation, log dump, override semantics, both harnesses, tests) landed. Fix any gap before finishing.
