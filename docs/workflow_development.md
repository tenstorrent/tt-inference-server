# Workflow development guide

This document is the onboarding doc for people adding new workflows, runners, or
test categories to the **workflow engine** — the `run_workflows.py` +
`workflow_module` / `test_module` / `report_module` / `llm_module` stack. `run.py`
(the user-facing CLI) brings up the inference server and drives the engine
through a single `WorkflowRunner`; you can also invoke `run_workflows.py`
directly against an already-running server, which is what this guide's examples
do.

## TLDR

```bash
python run_workflows.py \
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
python run_workflows.py \
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

- For `EVALUATION` / `BENCHMARK`: looks up the runner *name* by
  `ctx.model_spec.model_type.name` in `EVAL_DISPATCH` / `BENCHMARK_DISPATCH`,
  lazily imports it via `_resolve_runner` (so importing `dispatch` doesn't pull
  in every runner's optional deps), invokes it, hands the resulting Block to the
  accumulator, and returns `(exit_code, block)`.
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

Server-control + driver/parser abstraction for the LLM perf tools. The
**prefix-caching** and **speculative-decoding** benchmarks are end-to-end (see
the matching sections below); the standard LLM perf benchmark runs through the
`run_llm_bench.py` launcher (dispatched from `build_engine_commands`). The
directory also holds earlier scaffolding for the other tools (GuideLLM,
GenAIPerf, InferenceMax, vllm-bench) at varying maturity.

## Prefix-caching benchmark

Run the AIPerf prefix-cache sweep directly against an already-up vLLM-compatible
server. The workflow is `benchmarks`; the prefix-cache flag swaps the default
media-task dispatch for the scenario sweep defined in
[`llm_module/prefix_cache/manifest.json`](../llm_module/prefix_cache/manifest.json).

`run_workflows.py` has no import-time side effects, so it must run inside the
dedicated `PREFIX_CACHE` venv. Use the thin launcher `run_prefix_cache.py`,
which selects/creates that venv and re-execs `run_workflows.py` inside it — no
manual venv setup required:

```bash
python launchers/run_prefix_cache.py \
    --model Llama-3.1-8B-Instruct \
    --workflow benchmarks \
    --device gpu \
    --service-port 8000 \
    --prefix-cache \
    --prefix-cache-preset ci \
    --jwt-secret "$JWT_SECRET"
```

`run_prefix_cache.py` calls
`VENV_CONFIGS[WorkflowVenvType.PREFIX_CACHE].setup(...)` (declared in
[`workflows/workflow_venvs.py`](../workflows/workflow_venvs.py), requirements in
[`requirements/prefix-cache.txt`](../requirements/prefix-cache.txt)), then
`os.execv`s into `.workflow_venvs/.venv_prefix_cache/bin/python`, forwarding
every CLI argument to `run_workflows.py`. Setup is idempotent, so subsequent runs
reuse the existing venv. This mirrors how the `VenvCommand`s built by
[`workflows/workflow_dispatch.py`](../workflows/workflow_dispatch.py) provision
each workflow's venv at execute time.

Scenarios (`shared_system`, `prefix_pool`, `multi_turn`, `baseline`,
`mooncake_trace`) and per-preset grids (`ci`, `full`, `highcache_50k`) are
JSON-defined and overridable with `--prefix-cache-scenarios-json`. Override the
mooncake trace input with `--prefix-cache-trace`; the in-tree fixture at
[`llm_module/prefix_cache/sample_traces/ci_mooncake.jsonl`](../llm_module/prefix_cache/sample_traces/ci_mooncake.jsonl)
ships with the repo for reproducible CI runs.

### `highcache_50k` preset (trillion-scale customer shape)

`--prefix-cache-preset highcache_50k` encodes a high-reuse, large-context
serving shape: a **50K shared (cacheable) system prefix + 5K new ISL + 500 OSL
at concurrency 32** (one SC16 decode unit). Once warm the per-session KV cache
hit-rate is `50000 / (50000 + 5000) = ~90.9%` — meeting the ≥ 90% target — and
total input is ~55K tokens/request. The shape is modeled **two ways** under one
preset, plus a control:

- **`shared_system`** (synthetic): `shared_system_prompt_length=50000` is sent as
  an identical system message across every session (100% prefix reuse). Exact and
  deterministic.
- **`mooncake_trace`** (trace-driven, AIPerf prefix-synthesis Use Case 3/4):
  replays the in-tree
  [`customer_mooncake.jsonl`](../llm_module/prefix_cache/sample_traces/customer_mooncake.jsonl)
  whose 98-block (~50K) root is shared across all sessions, exercising a realistic
  radix-tree reuse pattern. Scalable via the `--synthesis-*` multipliers. Override
  the trace with `--prefix-cache-trace`; regenerate the fixture with
  [`generate_customer_mooncake.py`](../llm_module/prefix_cache/sample_traces/generate_customer_mooncake.py).
- **`baseline`** (control): a matched zero-prefix run (same 5K ISL / 500 OSL / c32)
  so the report's *Uplift vs baseline* table isolates the TTFT P50/P90/P99
  improvement attributable to prefix caching.

#### Goodput SLO enforcement (`--prefix-cache-goodput`)

The preset ships a default AIPerf [`--goodput`](https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking#use-case-4-goodput-analysis---measuring-sla-compliance)
SLO that turns the customer KPIs into a per-request "good" bar:

```
time_to_first_token:4000 output_token_throughput_per_user:45
```

i.e. a request is *good* when its TTFT ≤ 4000 ms (the P50 target used as the
per-request bar) **and** its output speed ≥ 45 tokens/s/user. AIPerf reports the
fraction of good requests as the **Goodput (req/s)** column. Override the bar
with `--prefix-cache-goodput "<KEY:VALUE …>"` (valid tags: `time_to_first_token`,
`request_latency`, `inter_token_latency` in ms; `output_token_throughput_per_user`
in tokens/s).

Because goodput is a single-threshold metric it can't express percentiles, so the
report also emits an **SLA Compliance vs Customer Targets** sub-table that grades
each run PASS/FAIL against the full KPI set — TTFT P50 < 4s, P90 < 10s, P99 < 35s;
output speed ≥ 45 t/s/u; hit-rate ≥ 90% — with an **Overall** verdict (PASS only
when every target is met, `N/A` when a metric wasn't captured, e.g. hit-rate when
the worker `/metrics` endpoint is unreachable).

`request_count=256` (8 waves of 32) gives usable TTFT percentiles including a
rough P99; bump it in
[`llm_module/prefix_cache/manifest.json`](../llm_module/prefix_cache/manifest.json)
for a tighter P99. Pair it with `--prefix-cache-metrics-url` (below) so the
worker `tt_prefix_cache_*` counters populate the hit-rate column:

```bash
python launchers/run_prefix_cache.py \
    --model <trillion-class-model> --workflow benchmarks --device tt \
    --service-port 8000 --prefix-cache --prefix-cache-preset highcache_50k \
    --prefix-cache-metrics-url <cpp_server-worker-host:port> \
    --jwt-secret "$JWT_SECRET"
```

By default AIPerf auto-derives the `/metrics` scrape from the load target
(`--service-port`). In a Dynamo deployment that target is the prefix-unaware
frontend, which does not aggregate the worker prefix-cache counters, so the
hit-rate column would render `null`. Point the scrape at the cpp_server
worker(s) with `--prefix-cache-metrics-url` (forwarded to AIPerf's
`--server-metrics`), keeping load on the frontend. It accepts a full URL,
`host:port`, or `host:port/metrics`, and is repeatable for multi-worker
(KV-routed) deployments — the parser sums hit/query deltas across the
`endpoint_url`-tagged series:

```bash
python launchers/run_prefix_cache.py \
    --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
    --service-port 8000 --prefix-cache --prefix-cache-preset ci \
    --prefix-cache-metrics-url bh-glx-120-a03u08.exabox.tenstorrent.com:9000 \
    --jwt-secret "$JWT_SECRET"
```

Each AIPerf run emits a `Block(kind="aiperf_prefix_cache")`, which the report
generator collapses into Markdown tables (Synthetic, Trace-Driven, *SLA Compliance
vs Customer Targets*, and Uplift vs zero-prefix baseline) via the renderer
registered in
[`report_module/prefix_cache_renderer.py`](../report_module/prefix_cache_renderer.py).
The synthetic/trace tables include `TTFT P90`, `Output Tok/s/User`, and
`Goodput (req/s)` columns alongside the existing percentiles.
Prefix-cache hit-rate is derived from the worker Prometheus counters
(`tt_prefix_cache_*` on cpp_server, or `vllm:prefix_cache_*` on vLLM) AIPerf
scrapes into `server_metrics_export.jsonl`; on Tenstorrent hardware the
`tt-vllm-plugin` currently disables prefix caching, so the hit-rate column
renders as `null` until that's lifted (validation work was done against a
reference GPU vLLM).

## Speculative-decoding benchmark

Run the AIPerf SPEED-Bench spec-decode sweep directly against an already-up
vLLM-compatible server. The workflow is
`benchmarks`; the spec-decode flag swaps the default media-task dispatch for
the sweep defined in [`llm_module/spec_decode/runs.py`](../llm_module/spec_decode/runs.py):
all 11 SPEED-Bench qualitative categories at concurrency 1 plus a 1k–32k ISL
throughput sweep at concurrency 1/16/64.

Server-side speculative config is out of scope — it belongs to whoever
launched the server, before the benchmark starts. Each run scrapes the vLLM
`vllm:spec_decode_*` Prometheus counters before/after every AIPerf invocation,
so acceptance rate and mean accepted length are per-run deltas. Note the TT
backend (`tt-vllm-plugin`) does not support speculative decoding yet, so a
spec-enabled target currently requires a reference GPU vLLM.

`run_workflows.py` must run inside the dedicated `SPEC_DECODE` venv (aiperf >= 0.8
for the SPEED-Bench dataset plugins — its pillow requirement conflicts with the
shared `constraints.txt` pin, hence the separate venv). Use the thin launcher
`run_spec_decode.py`, which selects/creates that venv (requirements in
[`requirements/spec-decode.txt`](../requirements/spec-decode.txt)) and
re-execs `run_workflows.py` inside it:

```bash
python launchers/run_spec_decode.py \
    --model Llama-3.1-8B-Instruct \
    --runtime-model-spec-json [spec_decode_runtime_spec.json] \
    --workflow benchmarks \
    --device gpu \
    --service-port 8000 \
    --spec-decode \
    --spec-decode-preset [ci | full]
```

Each AIPerf run emits a `Block(kind="aiperf_spec_decode")`, which the report
generator collapses into a per-run Markdown table (latency percentiles,
throughput, acceptance metrics) via the renderer registered in
[`report_module/spec_decode_renderer.py`](../report_module/spec_decode_renderer.py).
An acceptance rate of `0.000` means the target server is not actually running
with speculative decoding enabled.

## Agentic evals

Run agentic accuracy evals (Terminal-Bench and SWE-bench) directly against an
already-up OpenAI-compatible LLM server. The workflow is `agentic`; it bypasses
the generic media-task dispatcher and emits `Block(kind="evals")` results through
the same report/acceptance path as other evals.

Agentic harnesses require the dedicated `EVALS_AGENTIC` venv (Harbor,
mini-swe-agent, SWE-bench, and related tools). Use the thin launcher
`run_agentic.py`, which selects/creates that venv and re-execs `run_workflows.py`
inside it:

```bash
MODEL_SPECS_ENV=dev python launchers/run_agentic.py \
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
every CLI argument to `run_workflows.py`.

Agentic task selection still comes from [`reference_config/evals/eval_config.py`](../reference_config/evals/eval_config.py).
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
then runs each configured agentic task through the LLM driver/parser
adapters.

## How run.py drives the workflow engine

`run.py` builds one command list and drives it with a single `WorkflowRunner`:
a `ServerCommand` (when it's bringing up a docker/local server) followed by the
engine command(s). The runner runs them in order and stops at the first failure,
so a failed server bring-up aborts before any workflow runs. The routing +
command building lives in
[`workflows/workflow_dispatch.py`](../workflows/workflow_dispatch.py):

- `can_dispatch_to_engine(model_spec, runtime_config)` is the predicate `run.py`
  checks; everything supported routes to the engine.
- `_ENGINE_ROUTED_MODEL_TYPES` lists the model *types* that route purely by
  `model_type` — no per-name allowlist, so new models are picked up
  automatically. Today: image, video, audio, text-to-speech, CNN, embedding.
- `_LLM_LIKE_TYPES` (LLM + VLM) route per-workflow rather than by
  `_ENGINE_ROUTED_MODEL_TYPES`.
- `_ENGINE_WORKFLOW_NAMES` maps `WorkflowType` → engine workflow name
  (`BENCHMARKS` / `EVALS` / `SPEC_TESTS` / `RELEASE` / …).
- `build_engine_commands(model_spec, runtime_config, json_fpath)` is the **pure
  builder**: it returns the `VenvCommand`(s) for the requested workflow — no
  subprocess, no provisioning. `run.py` prepends the `ServerCommand` and runs
  the whole list.

Each `VenvCommand` handles venv isolation itself: on `execute()` it provisions
the venv it declares (`WORKFLOW_RUN_SCRIPT` for the generic engine path, plus any
`dependency_venvs`) and execs its script (`run_workflows.py`, or a launcher) in
that interpreter — so provisioning only happens once the server is up. Launcher
branches (agentic / prefix-cache / spec-decode / llm-bench) run in the current
interpreter and re-exec into their own venv themselves.

To onboard a new model type, add it to `_ENGINE_ROUTED_MODEL_TYPES` and make sure
the engine has the runners and suites it needs.

## Adding things to the workflow engine

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
   `test_module/dispatch.py`, keyed by `ctx.model_spec.model_type.name`. The
   value is the runner's *function name* as a string; `_resolve_runner` imports
   it lazily through the package `__getattr__`.
3. Export it from `test_module/benchmark_tests/__init__.py` (or eval equivalent)
   so both `_resolve_runner` and the top-level `test_module/__init__.py` can
   resolve it by name.
4. The dispatcher calls `accept_blocks([block], envelope=sweep_envelope(ctx))`
   for you — runners do not call it directly.

### Add a new spec test

Spec tests are pure data, driven by `test_suites/<category>.json` and the
filter system. See:

- [`test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md`](../test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md)
  for adding/removing test cases, matrices, model+device timing overrides.
- [`test_module/test_categorization_system/TEST_MARKING_SYSTEM.md`](../test_module/test_categorization_system/TEST_MARKING_SYSTEM.md)
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

## Status roadmap

| Area | Status |
|---|---|
| Image (SDXL base / img2img / inpainting, Flux, Motif, SD3.5) | Routed by `model_type`; SDXL fully validated |
| LLM / VLM benchmarks, evals, spec_tests, release | Routed through the engine (`build_engine_commands`) |
| Prefix-caching benchmark | `--workflow benchmarks --prefix-cache`; validated against reference GPU vLLM |
| Speculative-decoding benchmark | `--workflow benchmarks --spec-decode`; needs a spec-enabled (reference GPU) vLLM |
| Agentic evals | `--workflow agentic` via `run_agentic.py`; Terminal-Bench + SWE-bench against an external OpenAI-compatible server |
| CNN / audio / TTS / video / embedding runners | Routed by `model_type`; any correctness gaps tracked as bugs |
| Spec tests | Consolidated under `spec_tests` |
| New workflows on the horizon | Structured-outputs bench |

Policy: new benchmarks and runners should be authored as engine modules
(`workflow_module` / `test_module` / `llm_module`) from the start.

## Layout reference

```
<repo root>/
├── run_workflows.py                # CLI entry point (no import-time side effects)
├── launchers/
│   ├── run_prefix_cache.py         # thin launcher: ensures PREFIX_CACHE venv, execs run_workflows.py
│   └── run_agentic.py              # thin launcher: ensures EVALS_AGENTIC venv, execs run_workflows.py
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

- [`test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md`](../test_module/test_categorization_system/TEST_SUITE_CONFIG_GUIDE.md)
  — how to add models, devices, and test cases in suite JSON.
- [`test_module/test_categorization_system/TEST_MARKING_SYSTEM.md`](../test_module/test_categorization_system/TEST_MARKING_SYSTEM.md)
  — markers, prerequisites, and the `TestFilter` CLI/API.
- [`tests/report_module/SCHEMA_GUIDE.md`](tests/report_module/SCHEMA_GUIDE.md)
  — schema-authoring rules the generic renderer follows.
- [`test_module/stress_tests/README.md`](../test_module/stress_tests/README.md)
  — stress-test specifics.
- [`workflows/workflow_dispatch.py`](../workflows/workflow_dispatch.py) — `build_engine_commands` (engine command builder) + routing predicates.
