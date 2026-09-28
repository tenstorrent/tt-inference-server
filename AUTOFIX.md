# AutoFix: select the Gemma 4 autoport bringup benchmark profile

## Contract

The Gemma 4 autoport readiness run must report the repository-standard
4K-input/128-output vLLM serving points at concurrency 1 and 32. It must not
change the benchmark coverage of the canonical Gemma implementation or any
other model, and it must remain compatible with the already-built serving
image.

## Diagnosis

The first remote run was healthy but selected the generic 12-shape benchmark
table. With the autoport's 262144-token capacity, that table expands to 23
sequential cases. The run was canceled while actively processing the
8192-input/1024-output high-concurrency case; this was an operator error, not a
device hang. `AUTODEBUG.md` contains the full causal trace and alternatives.

The decisive local checks reproduced:

- ordinary mode: 23 autoport cases;
- `ci-nightly` mode: the same 23 cases;
- smoke mode: one unrelated 16-input/4-output case;
- target-only mode without reference rows: zero cases;
- the desired 4K profile: `(4096, 128, 1, 4)` and
  `(4096, 128, 32, 128)`.

## Change

`reference_config/benchmarking/benchmark_config.py` now contains an
implementation-scoped profile map keyed by weights repository, implementation
ID, and device. The only entry selects `(4096, 128)` for
`google/gemma-4-26B-A4B-it`, `gemma4_autoport`, and `P300X2`. Existing sweep
expansion supplies concurrency 1 and the allowed maximum of 32 with the normal
prompt counts.

This changes only host-side benchmark selection. It does not add a runtime
model-spec field, alter serving limits, or change code inside the serving
container, so the previously built image remains compatible and reusable.

## Validation

- `47 passed`:
  `tests/llm_module/test_benchmark_configs.py` and
  `tests/test_benchmark_config.py`.
- `1 passed, 63 deselected`:
  the Gemma autoport serving-contract test in
  `tests/test_run_vllm_api_server.py`.
- Direct construction produced exactly:
  `[(4096, 128, 1, 4), (4096, 128, 32, 128)]`.
- The sibling `tt_transformers` Gemma leaf retained 21 cases.
- Both changed Python files compile and pass `ruff format --check`.
- `git diff --check` passes.

A standalone Ruff 0.16.9 lint run reports pre-existing whole-file findings
(typing modernization and import ordering) in these files; no broad lint
cleanup was included in this focused fix.

## Remaining hardware validation

Dispatch the QB2 benchmark workflow from its existing `main` branch with the
updated tt-inference-server commit, the existing tt-metal/vLLM commits, and the
exact previously built image. Confirm that the report contains only the two 4K
rows and that both produce plausible latency/throughput results. Then run evals
and agentic evals separately against the same immutable image and revisions.
