# AutoFix: make Gemma 4 benchmark measurements valid and reproducible

## Contract

The Gemma 4 autoport readiness run must cover the same repository-standard
ISL/OSL matrix used by Qwen 3.8, including the 4K-input/128-output points at
concurrency 1 and 32 used for the bring-up performance comparison. It must
remain compatible with the already-built serving image.

## Diagnosis

The first remote run was healthy but selected the generic 12-shape benchmark
table. With the autoport's 262144-token capacity, that table expands to 23
sequential cases. The run was canceled while actively processing the
8192-input/1024-output high-concurrency case; this was an operator error, not a
device hang. `AUTODEBUG.md` contains the full causal trace and alternatives.

The decisive local checks reproduced the original selection behavior:

- ordinary mode: 23 autoport cases;
- `ci-nightly` mode: the same 23 cases;
- smoke mode: one unrelated 16-input/4-output case;
- target-only mode without reference rows: zero cases;
- the 4K diagnostic subset: `(4096, 128, 1, 4)` and
  `(4096, 128, 32, 128)`.

## Change

A temporary implementation-scoped profile narrowed the first diagnostic rerun
to the two 4K cases. After confirming the Qwen 3.8 contract, that narrowing was
removed: Gemma now uses `BENCHMARK_ISL_OSL_PAIRS`, the same 12 ISL/OSL shapes
as Qwen 3.8. Gemma's declared token capacity expands them to 23 cases, with
concurrency capped at long contexts; Qwen's larger explicit all-user token
budget expands the same shapes to 24 cases at C1/C16.

This remains a host-side benchmark-selection choice. It does not alter serving
limits or code inside the serving container, so the previously built image is
compatible and reusable.

## Validation

- `47 passed`:
  `tests/llm_module/test_benchmark_configs.py` and
  `tests/test_benchmark_config.py`.
- `1 passed, 63 deselected`:
  the Gemma autoport serving-contract test in
  `tests/test_run_vllm_api_server.py`.
- Direct construction produces 23 cases covering exactly the 12 standard
  ISL/OSL pairs, including 4K/C1 and 4K/C32.
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
exact previously built image. First confirm with the pinned two-point diagnostic
run that measurement begins only after the server log reports background trace
completion and that 4K performance recovers. Then dispatch the updated
inference-server revision and confirm the full 23-row Qwen-style matrix. Run
evals and agentic evals separately against the same immutable image and
revisions.

## QB2-main agentic dispatch compatibility

The inference-server engine and Gemma eval config already support standalone
`agentic` execution, but the QB2 `main` workflow and its downstream tt-shield
dispatcher both constrain the `workflow` input to `release`, `benchmarks`,
`evals`, or `spec_tests`. GitHub rejects `agentic` with HTTP 422 before a run is
created. Using the older Qwen-specific QB2 branch was explicitly excluded.

For one pinned compatibility revision, Gemma's model metadata declares
`ci_workflow_overrides: {evals: agentic}`. `EvalsWorkflow` honors this generic
metadata mapping by delegating to the existing `AgenticWorkflow` with the same
context, accumulator, and orchestration metadata. The agentic driver explicitly
resolves the `EVALS_AGENTIC` Harbor venv, so this remains a host-side dispatch
change and reuses the immutable serving image. The normal standard-eval run is
pinned to the preceding revision; the shim run is dispatched separately and
its exact revision is recorded in the remote evidence log. The metadata flag is
removed again in the following commit so the branch head retains normal
`evals` behavior; the generic, tested override mechanism remains available for
thin CI callers.

Focused validation covers the metadata routing, model-catalog load, API-server
contract, Docker readiness, benchmark configuration, formatting, compilation,
and import/lint checks (`181 passed` across the executed subsets).

The first hardware attempt reached the override only after the server was fully
ready, then failed before issuing any model request: the outer `evals` transport
had provisioned `EVALS_COMMON`, while `AgenticWorkflow` correctly looked for
`EVALS_AGENTIC/bin/harbor`. Dependency selection now resolves the same metadata
override before launching `run_workflows.py`, provisioning only
`EVALS_AGENTIC` for an `evals -> agentic` transport. A routing regression test
asserts this effective-workflow behavior; the workflow-dispatch suite passes
(`120 passed`).

The provisioning retry reached Harbor and launched both five-task CI subsets,
but its nominal CI success was not a valid model result. All Terminal-Bench
containers failed before agent startup because the Gemma task requested 32
CPUs on a 16-CPU QB2 runner. All SWE-Bench containers used
`127.0.0.1:8000` from inside their own Docker namespace, so none of their
requests reached the host vLLM server. The server log confirms zero requests
during the agent phase.

The follow-up ports the already-proven Qwen 3.8 branch behavior: mini-swe-agent
Docker environments receive a `host.docker.internal:host-gateway` compose
overlay and translated OpenAI endpoint, while Harbor itself retains the host
loopback endpoint. Gemma 4 26B's Terminal-Bench CPU override is capped to the
runner's 16 CPUs. The Qwen-derived gateway regressions and related agentic
suites pass (`87 passed`).

## Follow-up: complete the GPQA subset without timeout sentinels

The first standard-eval job was nominally green because EXPERIMENTAL-model
accuracy misses are informational, but lm-eval returned nonzero: 6 of 40
GPQA-Diamond prompts exceeded its 30-minute default request timeout. The saved
samples contain six `__INFERENCE_ERROR__` sentinels. The aggregate was 23/40
(57.5%); among the 34 completed prompts it was 23/34 (67.65%), so the incomplete
aggregate is not accepted as the final score.

Gemma 4's 26B GPQA config now follows the existing sibling long-thinking
policy: `max_concurrent=1` and a 7200-second client timeout. This avoids C32
queueing/KV contention and gives each potentially 32K-token response the full
decode rate. A model-config regression pins both values.

## Follow-up: make Gemma tool calls work in the reused image

The first corrected agentic run produced a valid Terminal-Bench result (2/5,
40%; 6,836,813 input and 90,518 output tokens), but SWE-Bench returned HTTP 400
for every trial because `tool_choice=auto` requires automatic tool choice and
a named parser. The Gemma autoport model spec now enables automatic tool choice
and selects the `gemma4` parser; the API-launch regression test asserts both
arguments.

Activating the parser exposed a second compatibility issue in the immutable
image: its Gemma parser accepted only `tokenizer`, while the installed vLLM
factory passed `tokenizer, model_config`. All five trials again produced zero
tokens. `run_vllm_api_server.py` now installs a narrowly scoped compatibility
wrapper only when the selected parser is `gemma4` and the installed constructor
cannot bind the newer two-argument call. The wrapper accepts the newer call and
delegates to the legacy tokenizer-only implementation. This file is bind-mounted
by dev-mode serving, so the exact existing image remains reusable. A regression
test recreates the legacy signature and proves construction with both factory
arguments.

The final same-image retry completed the full serving and scoring path:
SWE-Bench Verified resolved 1/5 tasks (20%). All five agents reached their
two-hour timeout, so the result is recorded as valid but timeout-limited rather
than a clean completion. One timed-out trial still produced a scorable patch
with reward 1; the other four scored zero. The durable parser constructor fix
also exists in the paired TT-Metal branch so a future rebuilt image will not
need the runtime bridge.

## Final serialized SWE retry

Review of run `36518073007` found that all five SWE trials reached their agent
timeout. The final retry sets Gemma SWE concurrency to one, emits compact
per-trial summaries under `workflow_logs/agentic_summaries`, and calculates
mean task duration from child timestamps when they are available instead of
dividing concurrent suite wall time by task count.

Validation passed: 88 focused tests and 373 expanded relevant tests, plus
format, import, compile, and diff checks. Run
https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36530661132
was dispatched with temporary inference revision
`7babaac11b8a7c9bf17dabf287dac2f4a9264be5`; the immutable image was reused
and all image builds were skipped. Branch head
`bfc2ee9bd5bda291f7b58f2f8d978e151b70b3a4` restores normal eval routing.
The run is intentionally left running, and no further eval repair is planned.
