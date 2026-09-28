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

## Follow-up: prevent trace warmup from contaminating measurements

The first two-point rerun proved the reduced sweep was selected, but its
performance was not valid: the Docker server returned HTTP 200 from `/health`
while the launcher's background trace process was still submitting requests.
The server log showed the benchmark overlapping trace captures through the
130944-token case; `/tmp/ready` was created only when the C32 measurement
finished. The resulting C1 mean TPOT was 82.81 ms (about 12.1 decode tok/s),
far below the 19.75 ms / 50.6 tok/s local baseline and the canceled run's
23.09 ms / 43.3 tok/s 4K result.

`workflow_module/commands.py` now treats the existing in-container
`/tmp/ready` file as part of readiness for local Docker VLLM launches that use
background trace capture. HTTP health still proves the API process is alive;
the workflow starts only after `docker exec <container> test -f /tmp/ready`
succeeds. Built-in-warmup models and `--disable-trace-capture` launches retain
their prior readiness behavior. This is a host-side orchestration change, so
the previously built image remains reusable.

Validation after this change:

- `135 passed`: server-command, benchmark-config, and API-server contract
  suites.
- The new regression tests cover marker selection, bypass conditions, the
  exact Docker probe, and the two-phase health-then-trace readiness sequence.
- Changed files compile, pass formatting, focused lint (`E`, `F`, `I`), and
  `git diff --check`.

## Remaining hardware validation

Dispatch the QB2 benchmark workflow from its existing `main` branch with the
updated tt-inference-server commit, the existing tt-metal/vLLM commits, and the
exact previously built image. Confirm that the report contains only the two 4K
rows, begins only after the server log reports background trace completion, and
recovers plausible latency/throughput results. Then run evals and agentic evals
separately against the same immutable image and revisions.
