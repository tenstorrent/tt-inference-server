# Benchmarking Tools for Tenstorrent Hardware

Comprehensive guide to performance benchmarking tools integrated with tt-inference-server.

## Overview

tt-inference-server supports four benchmarking tools for measuring LLM inference performance on Tenstorrent accelerators:

| Tool | Source | Installation | Best For |
|------|--------|--------------|----------|
| **vLLM** | vLLM project | Docker | Baseline reference |
| **GenAI-Perf** | NVIDIA Triton SDK | Docker (~10GB) | Official Triton validation |
| **AIPerf** | NVIDIA ai-dynamo | `pip install` | Detailed percentiles, VLM support |
| **GuideLLM** | vLLM project | `pip install` | Multi-turn, custom datasets, omni-modal scenarios |

## Quick Start

All tools use the same CLI pattern:

```bash
# vLLM (default)
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --docker-server

# GenAI-Perf
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --docker-server --tools genai

# AIPerf
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --docker-server --tools aiperf

# GuideLLM
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --docker-server --tools guidellm
```

## Tool Comparison

### vLLM benchmark_serving.py

**What it is:**  
Server-side benchmarking using vLLM's built-in `benchmark_serving.py` script.

**Key features:**
- Server-side TTFT measurement (measures first SSE chunk)
- Minimal overhead
- Direct HTTP/JSON interface
- Supports text and image (VLM) benchmarks

**Metrics provided:**
- TTFT (mean only)
- TPOT (mean only)
- Throughput (decode, prefill, user-level)
- E2EL (mean only)
- Request throughput

**Output:**
```json
{
  "mean_ttft_ms": 73.2,
  "mean_tpot_ms": 38.2,
  "tps_decode_throughput": 26.1,
  "mean_e2el_ms": 4930.4
}
```

**When to use:**
- Quick baseline reference
- Server-side measurements
- Standard vLLM validation

---

### GenAI-Perf

**What it is:**  
NVIDIA's official benchmarking tool for Triton Inference Server, also supports OpenAI-compatible endpoints.

**Key features:**
- Runs inside Triton Docker container
- Uses Triton's client library
- Docker-based execution
- Built-in warm-up support (`--warmup-request-count`)

**Metrics provided:**
- TTFT (mean)
- TPOT (mean)
- Throughput (decode, prefill, user-level)
- E2EL (mean)
- Request throughput

**Output:**
```json
{
  "mean_ttft_ms": 73.7,
  "mean_tpot_ms": 33.8,
  "tps_decode_throughput": 29.6,
  "mean_e2el_ms": 29782.8
}
```

**When to use:**
- Official NVIDIA validation
- Triton Inference Server compatibility testing
- Standard industry benchmark

**Note:**  
Current implementation has warm-up disabled (`--warmup-request-count "0"` in `benchmarking/genai_benchmark.py`). This causes high TTFT on the first benchmark due to cold start. NVIDIA's standard practice is 10 warm-up requests.

---

### AIPerf

**What it is:**  
NVIDIA's detailed performance benchmarking tool from the ai-dynamo project, designed for OpenAI-compatible APIs.

**Key features:**
- Client-side measurement (includes connection overhead)
- **Detailed percentiles** (mean, P50, P99, std) for all metrics
- Per-request metrics in JSONL format
- Automatic warm-up requests (implemented in our integration)
- VLM/multimodal support

**Metrics provided:**
- TTFT (mean, median, P99, std)
- TPOT (mean, median, P99, std)
- E2EL (mean, median, P99, std)
- Throughput (output tokens, requests)

**Output:**
```json
{
  "mean_ttft_ms": 93.5,
  "median_ttft_ms": 91.2,
  "p99_ttft_ms": 112.8,
  "std_ttft_ms": 8.4,
  "mean_tpot_ms": 40.2,
  "median_tpot_ms": 39.8,
  "p99_tpot_ms": 48.7,
  "std_tpot_ms": 3.1,
  "mean_e2el_ms": 5180.3,
  "output_token_throughput": 25.1,
  "request_throughput": 0.194
}
```

**When to use:**
- Detailed performance analysis
- Understanding latency distribution (not just averages)
- VLM/multimodal benchmarking
- Quick validation (supports `--limit-samples-mode smoke-test`)

---

## Understanding TTFT Differences

TTFT values differ across tools **by design** - they measure different points in the streaming response.

### vLLM's Streaming Pattern

When vLLM streams a response:

```
Chunk 1: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
         ↑ Empty role announcement (~73ms)

Chunk 2: {"choices": [{"delta": {"content": "The"}}]}
         ↑ First actual token (~93ms)

Chunk 3+: {"choices": [{"delta": {"content": " quick"}}]} ...
```

### Measurement Points

| Tool | Stops Timer At | What It Measures |
|------|----------------|------------------|
| **vLLM** | First SSE chunk (role) | Server response time |
| **AIPerf** | First content-bearing chunk | Time to actual text |
| **GenAI-Perf** | First token via Triton client | Full client roundtrip |

**Visual Timeline:**
```
HTTP POST → Role chunk → First token → Last token
0ms         73ms        93ms          5000ms
            ↑           ↑             ↑
            vLLM        AIPerf        E2EL (all tools)
```

**Real-world example (ISL=128, OSL=128, Con=1):**

| Tool | TTFT | TPOT | Analysis |
|------|------|------|----------|
| vLLM | 73ms | 38ms | Server-side baseline |
| AIPerf | 93ms | 40ms | +20ms (skips empty role chunk) |
| GenAI-Perf | 73ms* | 34ms | *2484ms on first run (cold start) |

**Key insight:** TPOT variance is only ~5%, confirming consistent decode performance. TTFT differences are measurement artifacts, not performance differences.

---

## Benchmark Configurations

All tools run the same benchmark sweep defined in `benchmark_config.py`:

### Text-to-Text Benchmarks

| Scenario | ISL Range | OSL | Concurrency | Requests |
|----------|-----------|-----|-------------|----------|
| Low concurrency | 128 - 32000 | 64-128 | 1 | 2-8 |
| High concurrency | 128 - 16000 | 64-2048 | 8-32 | 16-256 |

### Image (VLM) Benchmarks

| Scenario | ISL | OSL | Concurrency | Image Sizes | Requests |
|----------|-----|-----|-------------|-------------|----------|
| Low concurrency | 128 | 128 | 1 | 512x512 to 1024x1024 | 8 |
| High concurrency | 128 | 128 | 32 | 512x512 to 1024x1024 | 256 |

---

## Output Files

### Unified Output Directory

All three tools save results to `workflow_logs/benchmarks_output/`:

```
workflow_logs/benchmarks_output/
├── benchmark_<model>_<timestamp>_<params>.json              # vLLM results
├── genai_benchmark_<model>_<timestamp>_<params>.json        # GenAI-Perf results
└── aiperf_benchmark_<model>_<timestamp>_<params>.json       # AIPerf results
```

### AIPerf Raw Artifacts

AIPerf also generates detailed raw output in `.workflow_venvs/.venv_benchmarks_aiperf/artifacts/`:

```
.workflow_venvs/.venv_benchmarks_aiperf/artifacts/<model_id>/
└── bench_<isl>_<osl>_<concurrency>/
    ├── profile_export_aiperf.json    # Aggregated metrics with percentiles
    ├── profile_export.jsonl          # Per-request detailed metrics
    └── profile_export_aiperf.csv     # CSV format for spreadsheet analysis
```

**JSONL per-request format:**
```json
{"request_id": 0, "ttft_ms": 91.2, "tpot_ms": 39.8, "e2el_ms": 5102.4, "output_tokens": 128}
{"request_id": 1, "ttft_ms": 93.5, "tpot_ms": 40.1, "e2el_ms": 5145.2, "output_tokens": 128}
```

---

## Generated Reports

Run the reports workflow to generate unified summary tables:

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow reports
```

### Report Structure

**1. Combined Benchmark Table** (`workflow_logs/reports_output/benchmarks/`)

Stacks all three tools for direct comparison:

| Source | ISL | OSL | Concur | TTFT (ms) | TPOT (ms) | Tput Decode (TPS) |
|--------|-----|-----|--------|-----------|-----------|-------------------|
| vLLM | 128 | 128 | 1 | 73.2 | 38.2 | 26.1 |
| aiperf | 128 | 128 | 1 | 93.5 | 40.2 | 25.1 |
| genai-perf | 128 | 128 | 1 | 2484.6* | 320.8 | 3.1 |

*Cold start - needs warm-up fix

**2. AIPerf Detailed Percentile Tables** (`workflow_logs/reports_output/benchmarks_aiperf/`)

Separate tables with AIPerf's unique percentile metrics:

| ISL | OSL | Concur | TTFT Avg | TTFT P50 | TTFT P99 | TPOT Avg | TPOT P50 | TPOT P99 |
|-----|-----|--------|----------|----------|----------|----------|----------|----------|
| 128 | 128 | 1 | 93.5 | 91.2 | 112.8 | 40.2 | 39.8 | 48.7 |
| 128 | 128 | 32 | 2396.5 | 2580.1 | 2729.4 | 44.7 | 43.3 | 62.8 |

---

## Prefix-Caching Benchmarks (AIPerf)

Use the `--prefix-cache` flag on `--tools aiperf` to switch the sweep from the
default ISL/OSL grid to an opinionated prefix-caching scenario set that exercises
KV-cache reuse with controlled prefix-sharing patterns and variable arrival rates.

```bash
# Full validation sweep (~3 dozen runs across synthetic + trace-driven)
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --tools aiperf --prefix-cache

# CI smoke (5 scenarios x 2 concurrencies; ~12 runs)
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --tools aiperf --prefix-cache --prefix-cache-preset ci

# Subset of scenarios, overridden arrival pattern, fixed request rate
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --tools aiperf --prefix-cache \
  --prefix-cache-scenarios prefix_pool,baseline \
  --prefix-cache-arrival gamma --prefix-cache-request-rate 5.0

# Trace-driven: replay a production mooncake JSONL trace through AIPerf's
# prefix-synthesis pipeline (synthesis multipliers from the manifest still apply)
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --tools aiperf --prefix-cache \
  --prefix-cache-scenarios mooncake_trace \
  --prefix-cache-trace /path/to/production.jsonl
```

### Scenarios

| Scenario | Reuse model | AIPerf flags used | What it answers |
|----------|-------------|-------------------|-----------------|
| `shared_system` | 100% shared system prompt | `--shared-system-prompt-length` | Best-case prefix-cache uplift |
| `prefix_pool` | Pool of N prefixes (reuse ratio = `request_count / N`) | `--num-prefix-prompts`, `--prefix-prompt-length` | Realistic chat-style reuse at tunable rates |
| `multi_turn` | Organic reuse via re-sent chat history | `--conversation-num`, `--conversation-turn-mean`, `--conversation-turn-delay-mean` | Multi-turn chatting scenario |
| `mooncake_trace` | Replay of a mooncake JSONL trace + AIPerf synthesis multipliers | `--custom-dataset-type mooncake_trace`, `--input-file`, `--synthesis-*` | Production-realistic patterns, statistically scaled |
| `baseline` | Zero shared prefix (control) | _none_ | Reference for measuring uplift |

Each scenario expands across:

- Concurrencies: `[1, 8]` (CI) or `[1, 8, 32]` (full)
- Arrival patterns: `constant`, `poisson`, `gamma` (smoothness <1.0 = bursty) — per scenario
- ISL profiles: `short` (mean 256, stddev 64) and (full only) `long` (mean 4096, stddev 256)
- Reuse ratios for `prefix_pool`: `[4]` (CI) or `[2, 4, 16, 64]` (full)
- `mooncake_trace.synthesis_grid` entries (apply `--synthesis-speedup-ratio`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-max-isl`, `--synthesis-max-osl`)

Scenarios are defined in `benchmarking/benchmark_targets/prefix_cache_scenarios.json`
and can be overridden via `--prefix-cache-scenarios-json /path/to/custom.json`.

#### Trace-driven mode (`mooncake_trace`)

The `mooncake_trace` scenario uses AIPerf's
[prefix-synthesis pipeline](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md):

1. **`aiperf analyze-trace`** runs once per unique trace before the benchmark
   sweep. The resulting report (cache hit rate, prefix reuse ratio, ISL/OSL
   distributions) is bundled into every run's JSON under `trace_analysis`.
2. **`aiperf profile --custom-dataset-type mooncake_trace --input-file <trace>`**
   replays the trace. The `--synthesis-*` multipliers from the manifest's
   `synthesis_grid` allow scaling:
   - `speedup` (`--synthesis-speedup-ratio`): faster/slower request timing
   - `prefix_len` (`--synthesis-prefix-len-multiplier`): scale shared-prefix length
   - `prefix_root` (`--synthesis-prefix-root-multiplier`): split traces across N independent radix trees (lowers achievable cache hit rate)
   - `prompt_len` (`--synthesis-prompt-len-multiplier`): scale unique-prompt length
   - `max_isl` / `max_osl`: filter/cap sequence lengths
3. The runner pairs the trace's **theoretical** hit rate (from `analyze-trace`) with the **measured** hit rate (from `vllm:prefix_cache_*_total` Prometheus counters).

A small reproducible sample trace lives at
`benchmarking/benchmark_targets/sample_traces/ci_mooncake.jsonl` (regeneratable
via `python benchmarking/benchmark_targets/sample_traces/generate_ci_mooncake.py`).
Override the trace for a real run with `--prefix-cache-trace /path/to/prod.jsonl`.

### Cache hit-rate metric

AIPerf's `--server-metrics` flag (on by default) scrapes the vLLM Prometheus
endpoint during every run. The runner diffs the cumulative counters
`vllm:prefix_cache_hits_total` and `vllm:prefix_cache_queries_total` to compute
the per-run cache hit rate:

```
prefix_cache_hit_rate = (hits_final - hits_initial) / (queries_final - queries_initial)
```

If the server does not expose Prometheus metrics, the runner still emits the
TTFT/TPOT/ITL/E2EL percentiles and logs a warning that the hit rate is
unavailable.

### Output files

Each run writes a vLLM-compatible JSON to
`workflow_logs/benchmarks_output/`:

```
aiperf_prefix_cache_<model_id>_<timestamp>_<scenario>_<label>.json
```

with all the percentile fields plus the prefix-cache-specific fields:

```json
{
  "backend": "aiperf",
  "task_type": "prefix_cache",
  "scenario": "prefix_pool",
  "label": "pool_reuse4_plen512_short_c8_poisson",
  "arrival_pattern": "poisson",
  "concurrency": 8,
  "isl_mean": 256, "isl_stddev": 64,
  "num_prefix_prompts": 8, "prefix_prompt_length": 512,
  "prefix_cache_hit_rate": 0.74,
  "prefix_cache_hits_delta": 1832,
  "prefix_cache_queries_delta": 2470,
  "mean_ttft_ms": ..., "p50_ttft_ms": ..., "p95_ttft_ms": ..., "p99_ttft_ms": ...,
  "mean_tpot_ms": ..., "p95_tpot_ms": ..., "p99_tpot_ms": ...,
  "mean_itl_ms":  ..., "p95_itl_ms":  ..., "p99_itl_ms":  ...,
  "mean_e2el_ms": ..., "p95_e2el_ms": ..., "p99_e2el_ms": ...,
  "output_token_throughput": ...,
  "request_throughput": ...
}
```

For `mooncake_trace` runs the JSON additionally carries the trace + synthesis
provenance and the pre-run trace analysis:

```json
{
  "scenario": "mooncake_trace",
  "label": "trace_ci_sample_more_reuse_c8_poisson",
  "trace_input_file": "/.../sample_traces/ci_mooncake.jsonl",
  "custom_dataset_type": "mooncake_trace",
  "fixed_schedule": false,
  "block_size": 512,
  "synthesis_prefix_len_multiplier": 1.5,
  "synthesis_prompt_len_multiplier": 0.7,
  "trace_analysis": {
    "total_requests": 64,
    "unique_prefixes": 71,
    "cache_hit_rate": 0.54,
    "prefix_reuse_ratio": 0.4,
    "isl_stats": { "mean": 670.5, ... }
  },
  "prefix_cache_hit_rate": 0.49,
  ...
}
```

Raw AIPerf artifacts (per-request JSONL, server-metrics JSONL) land under
`.workflow_venvs/.venv_benchmarks_aiperf/artifacts/<model_id>/prefix_cache/<scenario>/<label>/`.

### Report

`--workflow reports` produces a dedicated section per model under
`workflow_logs/reports_output/benchmarks_prefix_cache/`:

- `aiperf_prefix_cache_display_<report_id>.md` - three tables:
  1. **Synthetic Scenarios — Per-run Percentiles** (TTFT/TPOT/ITL/E2EL P50/P95/P99 + measured cache hit %).
  2. **Trace-Driven (`mooncake_trace`) — Per-run Percentiles** with the trace name, synthesis variant, multipliers and a **theoretical vs measured** cache hit-rate column.
  3. **Uplift vs Zero-Prefix Baseline** that pairs each synthetic reuse scenario with the matching `baseline` run, with Δ% on mean TTFT/TPOT/E2EL.
- `data/aiperf_prefix_cache_stats_<report_id>.csv` - full schema for auditing.

### Requirements coverage

| Requirement | Where it lives |
|-------------|----------------|
| Configurable concurrency | `concurrencies` list per preset |
| Tunable prefix re-use ratio | `prefix_pool.reuse_ratios` → AIPerf `--num-prefix-prompts` plus `mooncake_trace` synthesis multipliers |
| Content / sequence diversity | `isl_profiles` with mean+stddev → AIPerf synthetic ISL distribution; `mooncake_trace` replays trace-native ISL/OSL distributions |
| CI vs full sweeps | `--prefix-cache-preset ci\|full` |
| Variable arrival rates | `arrival_patterns` per scenario / `--prefix-cache-arrival` (`constant`, `poisson`, `gamma` w/ smoothness <1 = bursty) |
| Mixed short/long context | `short` + `long` ISL profiles inside same sweep; `mooncake_trace.synthesis_grid` with `max_isl`/`max_osl` for context filtering |
| Cache hit rate | `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` delta from `server_metrics_export.jsonl` |
| Production-trace replay | `mooncake_trace` scenario + `--prefix-cache-trace /path/to/prod.jsonl` |
| Theoretical (achievable) hit rate | `aiperf analyze-trace` runs pre-sweep; report shows theoretical-vs-measured side by side |
| TPOT / ITL / Output Tput / E2EL with P50/P95/P99 | `parse_aiperf_output()` (extracts mean/p50/p95/p99/std) |
| Open-source, reproducible, auditable | AIPerf Apache 2.0 + JSON scenario manifest + in-tree sample trace checked into the repo |

---

## Smoke Testing

Limit benchmark runs to 2 configurations for quick validation:

```bash
# Works for vLLM (default) and AIPerf
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --docker-server --tools aiperf --limit-samples-mode smoke-test

# Works for GenAI-Perf
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --docker-server --tools genai --limit-samples-mode smoke-test
```

---

## Metrics Reference

| Metric | Description | Unit |
|--------|-------------|------|
| **TTFT** | Time To First Token - latency until first token/chunk | ms |
| **TPOT** | Time Per Output Token - average inter-token latency | ms |
| **ITL** | Inter-Token Latency - same as TPOT | ms |
| **E2EL** | End-to-End Latency - total request duration | ms |
| **Tput User** | User-level throughput (single request) | tokens/sec |
| **Tput Decode** | Decode throughput (all concurrent requests) | tokens/sec |
| **Tput Prefill** | Prefill/prompt processing throughput | tokens/sec |
| **Req Tput** | Request throughput | requests/sec |

### Percentile Statistics (AIPerf only)

- **Avg (mean)**: Average across all requests
- **P50 (median)**: 50th percentile - typical latency
- **P99**: 99th percentile - worst-case latency (tail latency)
- **std**: Standard deviation - variance from mean

---

## Architecture

### vLLM Benchmarks

```
run.py --tools vllm
  ↓
benchmarking/run_benchmarks.py
  ↓
.workflow_venvs/.venv_benchmarks_vllm/bin/serve
  ↓
Results → workflow_logs/benchmarks_output/benchmark_*.json
```

### GenAI-Perf Benchmarks

```
run.py --tools genai
  ↓
benchmarking/run_benchmarks.py
  ↓
benchmarking/run_genai_benchmarks.py (Docker orchestration)
  ↓
Docker container runs: benchmarking/genai_benchmark.py
  ↓
Results → workflow_logs/benchmarks_output/genai_benchmark_*.json
```

### AIPerf Benchmarks

```
run.py --tools aiperf
  ↓
benchmarking/run_benchmarks_aiperf.py
  ↓
.workflow_venvs/.venv_benchmarks_aiperf/bin/aiperf profile ...
  ↓
Results → workflow_logs/benchmarks_output/aiperf_benchmark_*.json
Raw data → .workflow_venvs/.venv_benchmarks_aiperf/artifacts/
```

---

## VLM (Vision-Language Model) Support

Both vLLM and AIPerf support image benchmarks.

### Backend Labels

| Tool | Text Benchmarks | Image Benchmarks |
|------|-----------------|------------------|
| vLLM | `"backend": "vllm"` | `"backend": "openai-chat"` |
| AIPerf | `"backend": "aiperf"` | `"backend": "aiperf"` |
| GenAI-Perf | `"backend": "genai-perf"` | Not yet supported |

vLLM automatically switches to `openai-chat` for image requests because it uses OpenAI Chat Completions API format:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "data:image/..."}}
    ]
  }]
}
```

---

## Example: Full Benchmark Run

### Step 1: Run vLLM Benchmarks (baseline)

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --docker-server
```

**Output:**
- 18 text benchmark JSON files
- 10 image benchmark JSON files (if model supports VLM)
- Saved to `workflow_logs/benchmarks_output/benchmark_*.json`

### Step 2: Run AIPerf Benchmarks

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --tools aiperf
```

**Note:** Server already running from Step 1, so omit `--docker-server`.

**Output:**
- 18 text benchmark JSON files with percentiles
- 10 image benchmark JSON files with percentiles
- Saved to `workflow_logs/benchmarks_output/aiperf_benchmark_*.json`
- Raw per-request data in `.workflow_venvs/.venv_benchmarks_aiperf/artifacts/`

### Step 3: Run GenAI-Perf Benchmarks

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --tools genai
```

**Output:**
- 18 text benchmark JSON files
- Saved to `workflow_logs/benchmarks_output/genai_benchmark_*.json`

### Step 4: Generate Unified Report

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow reports
```

**Output:**
- Combined table with vLLM → AIPerf → GenAI-Perf per configuration
- Separate AIPerf table with detailed percentiles
- Saved to `workflow_logs/reports_output/benchmarks/` and `benchmarks_aiperf/`

---

## Sample Results: gemma-3-4b-it on N300

### Text Benchmark Comparison (ISL=128, OSL=128, Concurrency=1)

| Source | TTFT (ms) | TPOT (ms) | Tput Decode (TPS) | E2EL (ms) | Req Tput (RPS) |
|--------|-----------|-----------|-------------------|-----------|----------------|
| vLLM | 73.2 | 38.2 | 26.1 | 4930.4 | 0.203 |
| aiperf | 93.5 | 40.2 | 25.1 | 5180.3 | 0.194 |
| genai-perf | 2484.6* | 320.8 | 3.1 | 51770.0 | 0.019 |

*Cold start artifact - needs warm-up configuration

### AIPerf Detailed Percentiles (ISL=128, OSL=128, Concurrency=32)

| Metric | Avg | P50 (Median) | P99 | Std Dev |
|--------|-----|--------------|-----|---------|
| TTFT | 2396.5ms | 2580.1ms | 2729.4ms | 614.0ms |
| TPOT | 44.7ms | 43.3ms | 62.8ms | 4.8ms |
| E2EL | 8026.7ms | 8090.2ms | 8235.0ms | 312.1ms |

**Insights:**
- P99 TTFT is 333ms higher than median (13% variance)
- P99 TPOT is 19ms higher than median (45% variance)
- High TPOT variance suggests occasional slowdowns under load

---

## Configuration Controls

### Limit to Target Configs Only

Skip sweep and only run configurations with defined targets:

```bash
export ONLY_BENCHMARK_TARGETS=1
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks
```

### Override Benchmark Parameters

Use custom benchmark configurations:

```bash
export OVERRIDE_BENCHMARK_TARGETS=/path/to/custom_benchmarks.json
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks
```

Example `custom_benchmarks.json`:
```json
{
  "gemma-3-4b-it": {
    "n300": [
      {
        "isl": 128,
        "osl": 128,
        "max_concurrency": 1,
        "num_prompts": 8
      }
    ]
  }
}
```

---

## Troubleshooting

### vLLM Benchmarks

| Issue | Solution |
|-------|----------|
| Server not starting | Check Docker logs: `docker logs <container>` |
| Connection refused | Verify server health: `curl http://localhost:8000/health` |
| Slow benchmarks | Ensure server is warm (run 1-2 requests first) |

### GenAI-Perf

| Issue | Solution |
|-------|----------|
| Docker pull errors | Normal informational messages, not actual errors |
| High first TTFT | Expected - warm-up disabled (needs fix in `genai_benchmark.py`) |
| Container conflicts | Clean up: `docker ps -a \| grep genai-tritonserver` |

### AIPerf

| Issue | Solution |
|-------|----------|
| All metrics are 0 | Old JSON files - delete and re-run |
| Missing dependencies | Delete `.workflow_venvs/.venv_benchmarks_aiperf/` and re-run |
| Authentication errors | Check `JWT_SECRET` is set |
| High cold-start TTFT | Automatic warm-up runs before benchmarks (already implemented) |

---

## Advanced Usage

### Running Without Docker Server

If vLLM server is already running and healthy:

```bash
# Skip --docker-server to run benchmarks only
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --tools aiperf
```

### Quick Smoke Test

Run only 2 benchmarks for rapid validation:

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks \
  --docker-server --tools aiperf --limit-samples-mode smoke-test
```

---

## File Structure

```
tt-inference-server/
├── benchmarking/
│   ├── run_benchmarks.py              # Main vLLM/GenAI-Perf runner
│   ├── run_benchmarks_aiperf.py       # AIPerf runner
│   ├── run_genai_benchmarks.py        # GenAI-Perf Docker orchestration
│   ├── genai_benchmark.py             # GenAI-Perf in-container script
│   ├── summary_report.py              # Report generation (all tools)
│   └── benchmark_config.py            # Benchmark configurations
├── workflow_logs/
│   ├── benchmarks_output/             # All benchmark results (unified)
│   └── reports_output/
│       ├── benchmarks/                # Combined reports
│       └── benchmarks_aiperf/         # AIPerf detailed reports
└── .workflow_venvs/
    ├── .venv_benchmarks_vllm/                  # vLLM environment
    ├── .venv_benchmarks_aiperf/                # AIPerf environment
    └── .venv_benchmarks_aiperf/artifacts/      # AIPerf raw output
```

---

## References

- **vLLM benchmarking:** [vllm/benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- **GenAI-Perf:** [NVIDIA Triton GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html)
- **AIPerf:** [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)
- **Workflow guide:** [Model Readiness Workflows User Guide](../docs/workflows_user_guide.md#performance-benchmarks)
