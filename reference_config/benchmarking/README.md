# Performance Benchmarks

The `benchmarks` workflow measures the latency and throughput of a running OpenAI-compatible inference server and grades the results against per-model targets. This page is the reference for what the workflow runs, how it is graded, and how to read and reproduce the results. For the differences between the four client tools (vLLM, AIPerf, GenAI-Perf, GuideLLM) and how each one defines its metrics, see the [Benchmarking Tools Guide](../../docs/benchmarking_tools.md).

- [Quick start](#quick-start)
- [How a run works](#how-a-run-works)
- [Warmup](#warmup)
- [The default sweep](#the-default-sweep)
- [Targets and grading](#targets-and-grading)
- [Goodput and requirements documents](#goodput-and-requirements-documents)
- [Fixed-workload protocol](#fixed-workload-protocol)
- [Repeated runs](#repeated-runs)
- [Outputs and reading the results](#outputs-and-reading-the-results)
- [Recommended procedure for acceptance runs](#recommended-procedure-for-acceptance-runs)
- [Other benchmarks](#other-benchmarks)
- [Code map](#code-map)

## Quick start

Start the server once and leave it running; client workflows reuse it:

```bash
python3 run.py --model Qwen/Qwen3-32B --tt-device t3k --workflow server --docker-server
```

Wait until the container log shows `Background trace capture completed successfully` (see [Warmup](#warmup)), then run the sweep:

```bash
python3 run.py --model Qwen/Qwen3-32B --tt-device t3k --workflow benchmarks
```

`--tools aiperf` (or `genai`, `guidellm`) selects a different client; `vllm` is the default. Adding `--docker-server` to the `benchmarks` command starts the server and benchmarks in one step, but then the sweep can begin before device warmup completes.

| Option | Effect |
|---|---|
| `--limit-samples-mode smoke-test` | One tiny point (ISL 16 / OSL 4) instead of the sweep; ignores `--concurrency-sweeps` |
| `--concurrency-sweeps` | Adds every power-of-two concurrency up to the maximum to each ISL/OSL pair |
| `ONLY_BENCHMARK_TARGETS=1` | Runs only the graded target points, not the sweep |
| `OVERRIDE_BENCHMARK_TARGETS=<path>` | Replaces `benchmark_targets/model_performance_reference.json` with your own file (same format) |
| `--goodput "<SLOs>"` | AIPerf goodput SLOs, e.g. `"time_to_first_token:2000 inter_token_latency:50"` (AIPerf only; see [Goodput](#goodput-and-requirements-documents)) |
| `--server-url`, `--service-port` | Benchmark an already-running server on another host or port |
| `--workflow release` | Runs accuracy evals, benchmarks and spec tests together, as used for releases |

## How a run works

1. `run.py` resolves the model spec (`workflows/model_specs/{prod,dev}/*.yaml`), sets up the client's virtual environment under `.workflow_venvs/` on first use, and hands off to the workflow engine (`run_workflows.py`).
2. The engine waits for the server's `/health` endpoint.
3. `test_module/llm_tests/llm_benchmark_tests.py` builds the list of sweep points from this directory's [`benchmark_config.py`](benchmark_config.py) and the model's targets ([`llm_module/benchmark_configs.py`](../../llm_module/benchmark_configs.py)).
4. `llm_module/runner.py` runs each point through the selected driver in [`llm_module/drivers/`](../../llm_module/drivers/) (`vllm.py`, `aiperf.py`, `genai_perf.py`, `guidellm.py`), parses the output with [`llm_module/parsers/`](../../llm_module/parsers/), and grades it with [`llm_module/target_checks.py`](../../llm_module/target_checks.py).
5. `report_module` writes a Markdown and JSON report at the end of the workflow; there is no separate reports step.

The client measures the server over HTTP, so any OpenAI-compatible server can be benchmarked, not only one started by `run.py`.

## Warmup

Warmup happens at three levels, and none of it should be measured.

1. **First boot.** The first start of a model on a host builds the device weight cache and compiles kernels; this can take up to an hour on large models. Later starts reuse the cache in the mounted volume. Never benchmark a first boot.
2. **Device trace capture, after every server start.** The vLLM container sends one request at each padded input length (128, 256, 512, … up to the model's maximum context) with 4 output tokens, then logs `Background trace capture completed successfully` and writes `/tmp/ready` (`vllm-tt-metal/src/run_vllm_api_server.py`, `utils/prompt_client.py`). `/health` returns 200 *before* this finishes, so a client that starts on `/health` alone can measure capture traffic and uncompiled shapes. Models with `has_builtin_warmup: true` in their spec warm up inside the model runner instead. `--disable-trace-capture` skips this step.
3. **Client warmup, per sweep point.** The default sweep does none: `vllm bench serve` sends one unmeasured readiness request per point and runs with `--num-warmups 0`, and the AIPerf, GenAI-Perf and GuideLLM drivers pass no warmup option. With small request counts, one cold request can dominate the mean and the P99. The [fixed-workload protocol](#fixed-workload-protocol) adds one complete, discarded pass of each graded point.

## The default sweep

Defined in [`benchmark_config.py`](benchmark_config.py) and built per model from its spec's `max_context`, `max_concurrency` (the batch size) and `max_tokens_all_users` (the KV-cache token budget).

**ISL/OSL pairs** (`BENCHMARK_ISL_OSL_PAIRS`): 128/128, 128/1024, 1024/128, 2048/128, 4096/128, 8192/128, 8192/1024, 10000/1024, 16384/128, 32768/128, 65536/128, 131072/128. A pair is kept only if ISL + OSL ≤ the model's `max_context`. `SUPER_CLUSTER` endpoints add 196608/128 and 255872/128.

**Concurrency.** Each pair runs at concurrency 1 and at the largest concurrency the KV budget allows, `min(max_concurrency, max_tokens_all_users // (ISL + OSL))`. `--concurrency-sweeps` adds every power of two in between.

**Requests per point** (`get_num_prompts`): 8× concurrency for 128-token sequences, 4× up to ISL 4096 or OSL 1024, 2× up to ISL 16384, and 1× beyond. These counts track regressions; they are too small for stable tail percentiles (see [acceptance runs](#recommended-procedure-for-acceptance-runs)).

For example, Qwen3-32B on T3K (batch 32, 131072-token budget) runs:

| ISL / OSL | Concurrency (requests) |
|---|---|
| 128 / 128 | 1 (8), 32 (256) |
| 128 / 1024 | 1 (4), 32 (128) |
| 1024 / 128 | 1 (4), 32 (128) |
| 2048 / 128 | 1 (4), 32 (128) |
| 4096 / 128 | 1 (4), 31 (124) |
| 8192 / 128 | 1 (2), 15 (30) |
| 8192 / 1024 | 1 (2), 14 (28) |
| 10000 / 1024 | 1 (2), 11 (22) |
| 16384 / 128 | 1 (2), 7 (14) |
| 32768 / 128 | 1 (1), 3 (3) |
| 65536 / 128 | 1 (1) |

**Other point types.** Vision-language models add image points (`ISL_OSL_IMAGE_RESOLUTION_PAIRS`: 128/128 at 512×512, 1024×1024, 1024×512 and 512×1024). LLMs and VLMs add structured-output points (`STRUCTURED_OUTPUT_PAIRS`, run with vLLM's `benchmark_serving_structured_output.py`). Media models (CNN, image, audio, TTS) have their own task types in the same file.

**Workload shape.** Prompts are random tokens (`random` dataset in vLLM, synthetic inputs in AIPerf) with fixed ISL and OSL and no shared prefix, sent through `/v1/chat/completions` with streaming. On a local server the vLLM client also sends `truncate_prompt_tokens: <ISL>`, so the chat template cannot push a prompt past ISL.

## Targets and grading

Targets live in [`benchmark_targets/model_performance_reference.json`](benchmark_targets/model_performance_reference.json), keyed by Hugging Face repo id, then device:

```json
"Qwen/Qwen3-32B": {
  "t3k": [
    {
      "isl": 128, "osl": 128, "max_concurrency": 1, "num_prompts": 8,
      "targets": {
        "theoretical": {"ttft_ms": 62.0, "tput_user": 41.0, "tput": 41.0}
      }
    }
  ]
}
```

Each entry is one graded point; it runs in addition to the sweep. Optional keys: `impl` scopes an entry to one implementation (`--impl`), `data_parallel` declares the entry already describes the whole data-parallel system (otherwise throughput targets of a data-parallel spec are scaled), and `task_type` / `image_*` describe VLM points.

**Graded metrics** (`llm_module/target_checks.py`):

| Field | Metric | Better |
|---|---|---|
| `ttft_ms` | Mean time to first token, ms | lower |
| `tput_user` | Per-user decode throughput, tokens/s/user (`1000 / mean TPOT` for vLLM) | higher |
| `tput` | Aggregate output throughput, tokens/s | higher |

Requirements documents can also grade TPOT, E2E latency, total throughput and goodput.

**Tiers.** `theoretical` is an analytical ceiling for the model at its deployed precision and parallelism, not a measured number. Three tiers derive from it through the spec's `perf_targets_map` (default below; a spec or device may override it):

| Tier | Default | Meaning |
|---|---|---|
| `functional` | 10% of theoretical | Throughput × 0.10, TTFT ÷ 0.10 |
| `complete` | 50% of theoretical | Throughput × 0.50, TTFT ÷ 0.50 |
| `target` | 100% of theoretical, or the `measured` block if present | Pass/fail verdict for the point |

A `measured` block (`{"ttft_ms": …, "tput_user": …, "tput": …, "tolerance": 0.05}`) replaces the theoretical ceiling for the `target` tier; it records what a given implementation reaches and gates regressions within its tolerance (default 5%). For Qwen3-32B on T3K above, `functional` needs TTFT ≤ 620 ms and ≥ 4.1 tokens/s/user, and `complete` needs ≤ 124 ms and ≥ 20.5.

A point with no targets is reported as `NA` (ungraded), never as a pass.

## Goodput and requirements documents

Goodput is the share of requests that meet every per-request SLO (TTFT, TPOT/ITL, E2E latency). Two ways to get it:

- `--tools aiperf --goodput "time_to_first_token:2000 inter_token_latency:50"` passes the SLOs to AIPerf for every sweep point.
- `--requirements-json <document>` drives the run from an LLM-serving requirements document (`schemaVersion` 2.x, loader in `workflow_module/requirements_schema.py`, example in `tests/fixtures/requirements/acme-llm-serving.json`): its scenarios define the sweep points, their SLOs (passed to `vllm bench serve --goodput` or AIPerf), and scalar targets graded with `must`/`should` priority. `--model` and `--tt-device` default from the document.

## Fixed-workload protocol

An opt-in protocol for qualification runs, enabled per model in the spec's `metadata` with `benchmark_protocol: fixed_workload` (the older `benchmark_token_timing: true` is an alias). It applies only to `--tools vllm`. For each point that has a measured target it:

- runs one complete, discarded warmup pass, then `benchmark_repetitions` measured repetitions (default 3), each graded and reported separately; points without targets run once, ungraded;
- pins the client (vLLM 0.13.0, Transformers 4.57.6, Python 3.11) and the model's exact tokenizer revision, verifies the tokenizer file hashes, and writes `tokenizer_identity.json` beside the results;
- uses fixed seeds, greedy decoding, ignore-EOS with exact output lengths, zero length variation and an unlimited request rate;
- measures TTFT to the first non-empty content event and decode speed over the first-to-last content interval;
- matches targets on request count as well as shape, and fails a point on partial requests, wrong token counts or inconsistent timing, even if the client exits 0. `benchmark_require_complete_metrics: true` also fails a point when a declared metric is missing.

The timing adapter (`llm_module/vllm_token_timing.py`) uses private vLLM 0.13.0 helpers; revalidate payloads, tokenization and streaming timing before upgrading either client pin. Server versions are independent of these pins. Worked example: [Llama-3.1-8B-Instruct on QuietBox 2](../../docs/models/llama31_8b_qb2.md).

## Repeated runs

To measure run-to-run variance, run the workflow engine directly with `--repeat N` against a running server:

```bash
python3 run_workflows.py --model Qwen/Qwen3-32B --device t3k --workflow benchmarks --repeat 3
```

Each run keeps its own report under `run_NN/`, and `summary/` aggregates every metric (mean, median, stdev, min, max, P50/P90/P99, coefficient of variation) and re-runs acceptance on the means. See [Repeated benchmark runs](../../docs/workflow_development.md#repeated-benchmark-runs---repeat).

## Outputs and reading the results

| What | Where |
|---|---|
| Report (Markdown + JSON) | `workflow_logs/reports_output/benchmarks/<org>__<model>_<device>_benchmarks/` |
| Raw client output | `llm/` under the same directory: vLLM `benchmark_*.json` with per-request detail, AIPerf `aiperf_artifacts/bench_<isl>_<osl>_<conc>_n<N>/` (`profile_export_aiperf.json`, per-request `profile_export.jsonl`), fixed-workload points under `point-<i>/{warmup,rep1,…}/` |
| Server log | `workflow_logs/docker_server/` |
| Run log | `workflow_logs/run_logs/` |

**Per-point table.** ISL, OSL, concurrency, requests, TTFT (mean, and P50/P99 where the tool reports them), TPOT, `tput_user`, input/output/total token throughput, request throughput, errors, and goodput when SLOs were given.

**Target checks.** Each graded point shows the `functional`, `complete` and `target` tiers with the target value, the measured/target ratio and a PASS/FAIL/NA check per metric.

**Consistency checks.**

- At concurrency 1, `tput` equals `tput_user`: there is one user.
- `tput_user × concurrency` approximates `tput` only while every slot is decoding; a large shortfall means queueing or prefill interrupting decode.
- TTFT at high concurrency includes queueing behind other requests' prefills, so it grows with concurrency even when prefill speed is unchanged.
- Long-input, short-output points (8192/128, 16384/128, …) characterize prefill; TTFT should scale with ISL. 128/1024 characterizes decode. Prefill and decode share the devices, so under concurrency new prefills pause running decodes; that shows up as ITL and P99 TPOT spikes.
- Numbers from different client tools are not directly comparable; see [Understanding metric differences](../../docs/benchmarking_tools.md#understanding-metric-differences).

## Recommended procedure for acceptance runs

The default sweep is built for regression tracking. For acceptance or customer comparison, keep the same shapes and add:

1. A warm server: not a first boot, and trace capture finished.
2. One discarded pass of each (ISL, OSL, concurrency) point, then at least three measured repetitions; report per-metric medians. The fixed-workload protocol does this for graded points.
3. Enough requests for the percentiles you report: at least 32 at concurrency 1 and at least 4× concurrency (minimum 64) at high concurrency. A P99 from fewer than about 100 requests is effectively the maximum.
4. Exact output lengths: vLLM 0.13.0 forces ignore-EOS on its `random` dataset; AIPerf needs `--extra-inputs ignore_eos:true`.
5. Reported metrics beyond the three graded ones: P50 and P99 TTFT, P50 and P99 TPOT and ITL, mean E2E latency, request throughput, failed requests (must be zero), and goodput against your SLOs.
6. Your production ISL/OSL mix as extra points (`OVERRIDE_BENCHMARK_TARGETS` or a requirements document).
7. A record of the environment: the release image tag (tt-metal and vLLM commits), the client tool and version, the exact command, and a `tt-smi -s` snapshot (firmware, KMD, board type).

The [Benchmarking Tools Guide](../../docs/benchmarking_tools.md#running-a-tool-outside-the-harness) gives stand-alone `vllm bench serve` and `aiperf profile` commands with these options.

## Other benchmarks

| Benchmark | Run with | Documentation |
|---|---|---|
| Prefix caching | `launchers/run_prefix_cache.py … --workflow benchmarks --prefix-cache` | [Prefix-caching benchmark](../../docs/workflow_development.md#prefix-caching-benchmark) |
| Speculative decoding | `launchers/run_spec_decode.py` | [Speculative-decoding benchmark](../../docs/workflow_development.md#speculative-decoding-benchmark) |
| Agentic trace replay (SemiAnalysis AgentX) | `run.py --workflow agentic_traces` | [Agentic-traces benchmark](../../docs/workflow_development.md#agentic-traces-benchmark) |
| Soak and stress | `run.py --workflow stress_tests` | [Stress tests](../../test_module/stress_tests/README.md) |
| Multi-turn, custom-dataset and omni-modal scenarios | `--tools guidellm` | [Benchmarking Tools Guide](../../docs/benchmarking_tools.md#guidellm) |

## Code map

| Path | Role |
|---|---|
| [`benchmark_config.py`](benchmark_config.py) | Sweep definition: ISL/OSL pairs, concurrency and request-count rules, image and structured-output points |
| [`benchmark_targets/model_performance_reference.json`](benchmark_targets/model_performance_reference.json) | Graded points and targets per model and device |
| `workflows/model_spec.py` | Reads the targets file and derives the tiers (`get_perf_reference_map`, `perf_targets_map`) |
| `llm_module/benchmark_configs.py`, `benchmark_protocol.py` | Builds the run configs; resolves the fixed-workload protocol |
| `llm_module/runner.py` | Runs each point, including warmup and repetitions |
| `llm_module/drivers/`, `llm_module/parsers/` | One driver and parser per client tool |
| `llm_module/target_checks.py` | Tiered grading and verdict |
| `test_module/llm_tests/llm_benchmark_tests.py`, `llm_performance_tests.py` | Workflow entry point into `llm_module` |
