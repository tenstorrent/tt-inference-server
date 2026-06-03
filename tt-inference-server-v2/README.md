# tt-inference-server-v2

This document is the onboarding doc for people adding new workflows, runners,
or test categories to v2. It assumes you already know what v1 does.

## TLDR

```bash
python tt-inference-server-v2/run.py \
    --model stable-diffusion-xl-base-1.0 \
    --workflow release \
    --device n150 \
    --service-port 8000
```

This launches the `release` workflow (evals + benchmarks + spec_tests) against
the inference server on `localhost:8000`, accumulates per-test `Block`s into a
single `ReportSchema`, applies acceptance criteria, and writes a markdown +
JSON report into `output/<model>_<device>_<workflow>/`.

## Repeated benchmark runs (`--repeat`)

To characterise run-to-run variance, repeat a workflow N times and emit one
aggregated summary report on top of the per-run reports:

```bash
python tt-inference-server-v2/run.py \
    --model stable-diffusion-xl-base-1.0 \
    --workflow benchmarks \
    --device n150 \
    --repeat 10
```

Each run keeps its own report; the summary is written alongside them:

```
workflow_logs/reports_output/<workflow>/<model>_<device>_<workflow>/
├── run_01/report_<id>.{md,json}
├── run_02/report_<id>.{md,json}
├── ...
├── run_10/report_<id>.{md,json}
└── summary/report_summary_<id>.{md,json}
```

The summary aggregates every per-run `benchmarks` block into per-metric
statistics (n, mean, median, stdev, min, max, p50/p90/p99, and coefficient of
variation) and re-runs the acceptance criteria on the aggregated means.
`--repeat 1` (the default) is unchanged: a single report, no `run_NN/`
subfolder, no summary.

When `--repeat > 1`, a failed run (e.g. a transient HTTP error mid-benchmark)
is logged and skipped rather than aborting the whole sweep — the summary is
still produced over the runs that succeeded. If *no* run produces a report the
summary step fails (`rc=1`).

Implementation: the pure stats core lives in `report_module/summary.py`; the
disk-driven aggregation + `SummaryCommand` live in
`workflow_module/summary_report.py`. The summary is built by reading each run's
`report_<id>.json` back off disk (runs are independent processes, so the
in-memory accumulator can't be shared across them).

## Architecture at a glance

```
                 run.py  (CLI entry point)
                    │
                    ▼
       ┌────────── workflow_module ──────────┐
       │  WorkflowExecution.run():           │
       │    prepare → run_tasks              │
       │      → format_results               │
       │      → apply_acceptance_criteria    │
       │      → inject_metadata              │
       │      → generate_report              │
       └──────┬──────────────────┬───────────┘
              │                  │
              ▼                  ▼
        test_module        report_module
        (runs tests,       (Block, ReportSchema,
         returns Block)     ReportGenerator,
              │             acceptance_criteria)
              ▼
        llm_module
        (server-control + driver/parser
         abstraction for LLM perf tools:
         GuideLLM, AIPerf, InferenceMax,
         GenAIPerf, vllm-bench)
```

Data flow for a single workflow run:

```
MediaContext ──► test_module.run_media_task ──► Block ─┐
                                                       │ accept_blocks(...)
                                                       ▼
                                               BlockAccumulator
                                                       │
                                                       │ build_schema()
                                                       ▼
                                                ReportSchema
                                                       │
                              acceptance_criteria_check│
                                                       ▼
                                               ReportGenerator
                                                       │
                                               ┌───────┴───────┐
                                               ▼               ▼
                                       report_*.md       report_*.json
```

## Modules

### `workflow_module/`

`WorkflowExecution` is a template-method base class. Subclasses set `name` and
`task_types`; the default `run()` invokes each task, builds a schema from the
process-global accumulator, runs acceptance, and generates the report.

`WORKFLOW_REGISTRY` is just a name → class dict:

```python
WORKFLOW_REGISTRY = {
    "evals":      EvalsWorkflow,       # task_types = (EVALUATION,)
    "benchmarks": BenchmarksWorkflow,  # task_types = (BENCHMARK,)
    "spec_tests": SpecTestsWorkflow,   # task_types = (SPEC_TESTS,)
    "release":    ReleaseWorkflow,     # composes the three above
}
```

`ReleaseWorkflow` is the composition primitive: it runs each child by name and
flattens their task lists. Adding a new leaf workflow to release is one line.

`blocks_sink.py` owns the process-global `BlockAccumulator`. Runners do not
hand a Block back to the workflow directly — they call `accept_blocks([block],
envelope=sweep_envelope(ctx))` and the accumulator collects them in insertion
order. The first non-empty envelope wins, so `model_name` / `device` /
`generated_at` are recorded once and don't need to live on every per-block
`targets` dict.

### `test_module/`

Dispatches a `MediaContext` to the right runner.

```
test_module/
├── context.py                       # MediaContext, get_health, count_tokens
├── dispatch.py                      # MediaTaskType + run_media_task
├── _test_common/                    # BaseTest, Block helpers, targets, target_check
├── benchmark_tests/                 # one *_benchmark_tests.py per media
├── eval_tests/                      # one *_eval_tests.py per media
├── llm_tests/                       # LLM performance tests
├── health_tests/                    # DeviceLiveness, MediaServerLiveness
├── stability_tests/                 # long-running stability checks
├── stress_tests/                    # stress regimen + its own runner
├── integration_tests/               # cross-component smoke tests
├── unit_tests/                      # fast, isolated checks
├── load_param_tests/                # load/param test classes
├── test_suites/<category>.json      # declarative test matrices (one per media)
└── test_categorization_system/      # TestFilter, suite loader, marker docs
```

`MediaTaskType` has three values: `EVALUATION`, `BENCHMARK`, `SPEC_TESTS`.
`run_media_task(ctx, task_type)`:

- For `EVALUATION` / `BENCHMARK`: looks up the runner by
  `ctx.model_spec.model_type.name` in `EVAL_DISPATCH` / `BENCHMARK_DISPATCH`,
  invokes it, hands the resulting Block to the accumulator, and returns
  `(exit_code, block)`.
- For `SPEC_TESTS`: resolves matching cases from `test_suites/*.json` via
  `TestFilter`, instantiates each test class, calls `BaseTest.run_tests()`, and
  hands every resulting Block to the accumulator.

`BaseTest` (in `_test_common/base_test.py`) wraps the per-test retry loop,
timeout handling, log capture, and Block construction. Every spec-test class
inherits from it and only implements `_run_specific_test_async()`.

### `report_module/`

Renderer-agnostic schema and a registry-based renderer.

- `schema.py` defines `Block` and `ReportSchema`. Each `Block` carries `kind`
  (`benchmarks` / `evals` / `spec_tests` / `health` / `stress` / …), optional
  `title` / `task_type` / `id`, a `targets` dict (per-block thresholds), and a
  free-form `data` payload.
- `generator.py` consumes a `ReportSchema`, dispatches each Block to the
  renderer registered for its `kind`, collapses consecutive same-heading blocks,
  injects the spec-test summary, and writes `report_<report_id>.md` + the
  matching JSON.
- `renderers.py` is the kind → renderer registry. Today every registered kind
  (`benchmarks`, `evals`, `spec_tests`, `stress_tests`) and every unregistered
  kind both go through the same `render_generic_table`, which follows the
  rules in [`tests/report_module/SCHEMA_GUIDE.md`](tests/report_module/SCHEMA_GUIDE.md).
  Per-kind renderers can be added via the `@register(kind)` decorator when a
  shape genuinely doesn't fit the generic table; we haven't needed one yet.
- `acceptance_criteria.py` walks the schema and produces a categorized
  pass/fail summary. Routing is by `Block.kind` alone — no substring matching.
  Per-kind rules:
    - `benchmarks`: at least one tier (`functional` / `complete` / `target`)
      in `targets.target_checks` must have every `*_check` field pass.
    - `evals`: `data.accuracy_check` must not be FAIL (3) and `data.success`
      must not be False.
    - `spec_tests`: `data.success` must not be False. Infra task types
      (`health`, `infra`, `unit`, `stability`, `integration`) are excluded
      from acceptance.

### `llm_module/`

Work in progress — do not use yet, except for the **prefix-caching benchmark**
which is end-to-end on v2 today (see "Prefix-caching benchmark" below). The
directory also holds early scaffolding for a driver/parser abstraction over the
other LLM perf tools (GuideLLM, GenAIPerf, InferenceMax, vllm-bench); those
aren't wired up yet and LLM benchmarking still happens through v1 for
everything except prefix caching.

## Prefix-caching benchmark

Run the AIPerf prefix-cache sweep directly against an already-up vLLM-compatible
server (no v1 entry point involved). The workflow is `benchmarks`; the
prefix-cache flag swaps the default media-task dispatch for the scenario sweep
defined in [`llm_module/prefix_cache/manifest.json`](llm_module/prefix_cache/manifest.json).

`run.py` itself has no import-time side effects, so it must run inside the
dedicated `V2_PREFIX_CACHE` venv. Use the thin launcher `run_prefix_cache.py`,
which selects/creates that venv and re-execs `run.py` inside it — no manual venv
setup required:

```bash
python tt-inference-server-v2/run_prefix_cache.py \
    --model Llama-3.1-8B-Instruct \
    --workflow benchmarks \
    --device gpu \
    --service-port 8000 \
    --prefix-cache \
    --prefix-cache-preset ci \
    --jwt-secret "$JWT_SECRET"
```

`run_prefix_cache.py` calls
`VENV_CONFIGS[WorkflowVenvType.V2_PREFIX_CACHE].setup(...)` (declared in
[`workflows/workflow_venvs.py`](../workflows/workflow_venvs.py), requirements in
[`requirements/v2-prefix-cache.txt`](../requirements/v2-prefix-cache.txt)), then
`os.execv`s into `.workflow_venvs/.venv_v2_prefix_cache/bin/python`, forwarding
every CLI argument to `run.py`. Setup is idempotent, so subsequent runs reuse
the existing venv. This mirrors how [`workflows/v2_bridge.py`](../workflows/v2_bridge.py)
selects the per-workflow venv externally for image-model runs, keeping venv
selection out of `run.py`.

Scenarios (`shared_system`, `prefix_pool`, `multi_turn`, `baseline`,
`mooncake_trace`) and per-preset grids are JSON-defined and overridable with
`--prefix-cache-scenarios-json`. Override the mooncake trace input with
`--prefix-cache-trace`; the in-tree fixture at
[`llm_module/prefix_cache/sample_traces/ci_mooncake.jsonl`](llm_module/prefix_cache/sample_traces/ci_mooncake.jsonl)
ships with the repo for reproducible CI runs.

Each AIPerf run emits a `Block(kind="aiperf_prefix_cache")`, which the report
generator collapses into three Markdown tables (Synthetic, Trace-Driven, Uplift
vs zero-prefix baseline) via the renderer registered in
[`report_module/prefix_cache_renderer.py`](report_module/prefix_cache_renderer.py).
vLLM prefix-cache hit-rate is derived from the Prometheus counters AIPerf
scrapes into `server_metrics_export.jsonl`; on Tenstorrent hardware the
`tt-vllm-plugin` currently disables prefix caching, so the hit-rate column
renders as `null` until that's lifted (validation work was done against a
reference GPU vLLM).

## Agentic evals

Run agentic accuracy evals (Terminal-Bench and SWE-bench) directly against an
already-up OpenAI-compatible LLM server. The workflow is `agentic`; it bypasses
the generic media-task dispatcher and emits `Block(kind="evals")` results through
the same report/acceptance path as other evals.

Agentic harnesses require the dedicated `EVALS_AGENTIC` venv (Harbor,
mini-swe-agent, SWE-bench, and related tools). Use the thin launcher
`run_agentic.py`, which selects/creates that venv and re-execs `run.py` inside
it:

```bash
MODEL_SPECS_ENV=dev python tt-inference-server-v2/run_agentic.py \
    --model Qwen3.6-27B \
    --workflow agentic \
    --device gpu \
    --service-port 8000 \
    --runtime-model-spec-json /tmp/qwen36_agentic_nightly.json
```

`MODEL_SPECS_ENV=dev` is only needed when the target model spec lives in
`workflows/model_specs/dev/`; omit it for models present in the default `prod`
catalog. `run_agentic.py` calls
`VENV_CONFIGS[WorkflowVenvType.EVALS_AGENTIC].setup(...)` (declared in
[`workflows/workflow_venvs.py`](../workflows/workflow_venvs.py), requirements in
[`requirements/evals-agentic.txt`](../requirements/evals-agentic.txt)), then
`os.execv`s into `.workflow_venvs/.venv_evals_agentic/bin/python`, forwarding
every CLI argument to `run.py`.

Agentic task selection still comes from [`evals/eval_config.py`](../evals/eval_config.py).
The runtime config JSON is optional, but it is how limit modes are forwarded to
the agentic drivers. For a nightly-limited run, include:

```json
{
  "runtime_config": {
    "model": "Qwen3.6-27B",
    "workflow": "agentic",
    "device": "gpu",
    "service_port": "8000",
    "limit_samples_mode": "ci-nightly"
  }
}
```

The workflow checks the server via `/v1/models`, sets OpenAI-compatible
environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_BASE`),
then runs each configured agentic task through the v2 LLM driver/parser
adapters.

## How v1 routes to v2

While the migration is in progress, v1 stays the entry point for everything
that hasn't been ported yet. The routing rules live in
[`workflows/v2_bridge.py`](../workflows/v2_bridge.py):

- `_V2_ROUTED_MODELS` lists the model names that are validated end-to-end on
  v2. Today: SDXL base + img2img + inpainting.
- `_V2_WORKFLOW_NAMES` maps v1 `WorkflowType` → v2 workflow name
  (`BENCHMARKS` / `EVALS` / `SPEC_TESTS` / `RELEASE`).
- `can_route_to_v2(model_spec, runtime_config)` is the predicate v1's runner
  checks before delegating.
- `run_v2_workflows(...)` materializes the v2 venv (`WorkflowVenvType.V2_RUN_SCRIPT`,
  defined in `workflows/workflow_venvs.py`), shells out to `run.py`, and
  forwards stdout/stderr.

When you're ready to move a model from v1 to v2, add it to
`_V2_ROUTED_MODELS` and make sure v2 has the runners and suites needed.

## Adding things to v2

### Add a new workflow

1. Subclass `WorkflowExecution` in `workflow_module/workflows.py`:
    ```python
    class StructuredOutputsWorkflow(WorkflowExecution):
        name = "structured_outputs"
        task_types = (MediaTaskType.BENCHMARK,)  # or a new MediaTaskType
    ```
2. Register it in `WORKFLOW_REGISTRY`.
3. If the workflow needs a new task type (e.g. you want it dispatched separately
   from regular benchmarks), add a value to `MediaTaskType` and wire it in
   `run_media_task`.
4. (Optional) Add it to `ReleaseWorkflow.children` if release should run it.

### Add a new media runner (e.g. structured outputs, agentic accuracy)

1. Add a runner under `test_module/benchmark_tests/<media>_benchmark_tests.py`
   (or `eval_tests/<media>_eval_tests.py`). It must:
    - Take a `MediaContext`.
    - Run the test against `ctx.base_url`.
    - Return a `Block` with the appropriate `kind` (`benchmarks` / `evals`).
2. Register the runner in `BENCHMARK_DISPATCH` / `EVAL_DISPATCH` in
   `test_module/dispatch.py`, keyed by `ctx.model_spec.model_type.name`.
3. Export it from `test_module/benchmark_tests/__init__.py` (or eval equivalent)
   so the top-level `test_module/__init__.py` re-exports it.
4. The dispatcher calls `accept_blocks([block], envelope=sweep_envelope(ctx))`
   for you — runners do not call it directly.

### Add a new spec test

Spec tests are pure data, driven by `test_suites/<category>.json` and the
filter system. See:

- [`test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md`](test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md)
  for adding/removing test cases, matrices, model+device timing overrides.
- [`test_module/test_categorization_system/TEST_MARKING_SYSTEM.md`](test_module/test_categorization_system/TEST_MARKING_SYSTEM.md)
  for markers, prerequisites, and the `TestFilter` API.

To author a new test class:

1. Subclass `BaseTest` from `_test_common/base_test.py`. Set `KIND` and
   `TASK_TYPE` class attributes (`KIND="spec_tests"` for the default routing).
2. Implement `_run_specific_test_async(self) -> dict`. Return a dict with at
   least `{"success": bool, ...}`; the parent class merges retry/log/timing
   metadata and wraps it in a `Block` automatically.
3. Add a template referencing your class in the relevant suite JSON
   (`test_suites/<category>.json`) and reference it from `test_cases` /
   `test_matrices`.

### Add a new report kind

1. Pick a new `kind` string for the Block.
2. Register a renderer in `report_module/renderers.py`. If your data shape fits
   the generic renderer's recipe, you can skip this and rely on the fallback.
   See [`tests/report_module/SCHEMA_GUIDE.md`](tests/report_module/SCHEMA_GUIDE.md)
   for the schema-authoring rules the generic renderer follows (one record per
   `(kind, model, device)`, nested dicts become sub-tables, etc.).
3. If the new kind has acceptance criteria, extend `acceptance_criteria.py`
   with a `_check_<kind>(schema)` and add it to the category list in
   `acceptance_criteria_check`.

## Status & migration roadmap

| Area | Status |
|---|---|
| SDXL base / img2img / inpainting (eval, benchmark, release) | Routed to v2 today via `v2_bridge.py` |
| Other image models (Flux, Motif, SD3.5) | Runners exist in v2; not yet routed |
| LLM benchmarking via `llm_module` | Work in progress — LLMs still run through v1, except prefix-caching which is end-to-end on v2 |
| Prefix-caching benchmark | Implemented on v2 (`--workflow benchmarks --prefix-cache`); validated against reference GPU vLLM |
| Agentic evals | Implemented on v2 (`--workflow agentic` via `run_agentic.py`); runs Terminal-Bench and SWE-bench against an external OpenAI-compatible server |
| CNN / audio / TTS / video / embedding runners | Scaffolded; correctness gaps tracked as bugs |
| Spec tests | Ported from v1's `server_tests/`; renamed consistently to `spec_tests` |
| New workflows on the horizon | Spec-decode bench, structured-outputs bench |

Migration policy (current consensus): start using v2 right away for SDXL and
treat anything missing as a bug. New benchmarks should be authored as v2
modules from the start rather than bolted onto v1.

## Layout reference

```
tt-inference-server-v2/
├── run.py                          # CLI entry point (no import-time side effects)
├── run_prefix_cache.py             # thin launcher: ensures V2_PREFIX_CACHE venv, execs run.py
├── run_agentic.py                  # thin launcher: ensures EVALS_AGENTIC venv, execs run.py
├── workflow_module/                # Workflow scaffolding + block accumulator
│   ├── workflows.py                # Concrete workflows + WORKFLOW_REGISTRY
│   ├── execution.py                # WorkflowExecution template + WorkflowResult
│   └── blocks_sink.py              # BlockAccumulator + accept_blocks
├── test_module/                    # Per-media runners + spec-test dispatch
│   ├── context.py                  # MediaContext + health helpers
│   ├── dispatch.py                 # run_media_task + EVAL/BENCHMARK_DISPATCH
│   ├── _test_common/               # BaseTest, blockify, targets, target_check
│   ├── benchmark_tests/            # cnn/image/audio/video/tts/embedding/llm
│   ├── eval_tests/                 # cnn/image/audio/video/tts/embedding
│   ├── llm_tests/                  # LLM performance, prefix-cache, and agentic tests
│   ├── health_tests/               # DeviceLiveness, MediaServerLiveness
│   ├── stability_tests/            # device stability checks
│   ├── stress_tests/               # stress regimen (has its own runner)
│   ├── integration_tests/          # cross-component smoke tests
│   ├── unit_tests/                 # fast, isolated checks
│   ├── load_param_tests/           # load/param test classes
│   ├── test_suites/                # declarative test matrices (per category)
│   └── test_categorization_system/ # TestFilter + suite loader + marker docs
├── report_module/                  # Schema, generator, renderers, acceptance
│   ├── schema.py                   # Block, ReportSchema
│   ├── generator.py                # ReportGenerator + spec-test summary
│   ├── renderers.py                # kind → renderer registry
│   ├── acceptance_criteria.py      # per-category pass/fail rules
│   ├── markdown_table.py           # table helpers
│   ├── formatting.py               # value coercion
│   ├── display.py                  # CLI display helpers
│   └── report_file_saver.py        # md/json writer
├── llm_module/                     # LLM perf tool drivers + parsers
│   ├── runner.py                   # LLMPerformanceRunner
│   ├── server_control.py           # ServerController (warmup, traces, health)
│   ├── config.py                   # LLMRunConfig, ServerConnection, DriverContext
│   ├── benchmark_configs.py        # get_llm_configs(model_spec, device)
│   ├── drivers/                    # base, agentic, aiperf, aiperf_prefix_cache, genai_perf, guidellm, inferencex, vllm
│   ├── parsers/                    # mirror of drivers/
│   ├── agentic/                    # Terminal-Bench/SWE-bench harness wrappers
│   └── prefix_cache/               # Scenario manifest + expander + CI mooncake trace
├── tests/                          # pytest tests for the modules above
└── output/                         # generated reports land here
```

## In-tree references

- [`test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md`](test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md)
  — how to add models, devices, and test cases in suite JSON.
- [`test_module/test_categorization_system/TEST_MARKING_SYSTEM.md`](test_module/test_categorization_system/TEST_MARKING_SYSTEM.md)
  — markers, prerequisites, and the `TestFilter` CLI/API.
- [`tests/report_module/SCHEMA_GUIDE.md`](tests/report_module/SCHEMA_GUIDE.md)
  — schema-authoring rules the generic renderer follows.
- [`test_module/stress_tests/README.md`](test_module/stress_tests/README.md)
  — stress-test specifics.
- [`workflows/v2_bridge.py`](../workflows/v2_bridge.py) — v1 → v2 delegation.
