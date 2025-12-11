# [AIPerf](https://github.com/ai-dynamo/aiperf) Benchmarking for Tenstorrent Hardware

AIPerf benchmarks measure LLM inference performance on Tenstorrent accelerators, providing detailed latency percentiles (mean, P50, P99) and throughput metrics.

## Usage

Run AIPerf benchmarks through tt-inference-server:

```bash
python run.py --model <model> --device <device> --workflow benchmarks --docker-server --tools aiperf
```

### Example: gemma-3-4b-it on N300

```bash
python run.py --model gemma-3-4b-it --device n300 --workflow benchmarks --docker-server --tools aiperf
```

### Example: gemma-3-27b-it on T3K

```bash
python run.py --model gemma-3-27b-it --device t3k --workflow benchmarks --docker-server --tools aiperf
```

## What the Workflow Does

1. **Starts vLLM server** in Docker container on Tenstorrent hardware
2. **Waits for server health** check to pass
3. **Runs benchmark sweep** across multiple ISL/OSL/concurrency configurations
4. **Parses AIPerf output** and saves standardized JSON results
5. **Generates markdown report** with detailed performance tables

## Benchmark Configurations

The benchmark sweep runs multiple configurations defined in `benchmark_config.py`:

| Batch | ISL Range | OSL | Concurrency | Requests |
|-------|-----------|-----|-------------|----------|
| Low concurrency | 128 - 32000 | 128 | 1 | 2-8 |
| High concurrency | 128 - 16000 | 64-2048 | 8-32 | 16-256 |

## Output Files

### Individual Benchmark Results

Results are saved to `workflow_logs/benchmarks_aiperf_output/`:

```
aiperf_benchmark_<model_id>_<timestamp>_isl-<isl>_osl-<osl>_maxcon-<concurrency>_n-<requests>.json
```

### Sample Result: gemma-3-4b-it on N300 (ISL=128, OSL=128, Concurrency=32)

```json
{
  "date": "20251210-160152",
  "backend": "aiperf",
  "model_id": "google/gemma-3-4b-it",
  "num_prompts": 256,
  "max_concurrency": 32,
  "mean_ttft_ms": 2396.53,
  "median_ttft_ms": 2580.12,
  "p99_ttft_ms": 2729.42,
  "std_ttft_ms": 614.01,
  "mean_tpot_ms": 44.69,
  "median_tpot_ms": 43.35,
  "p99_tpot_ms": 62.83,
  "std_tpot_ms": 4.82,
  "mean_e2el_ms": 8026.66,
  "median_e2el_ms": 8090.18,
  "p99_e2el_ms": 8234.98,
  "output_token_throughput": 502.62,
  "request_throughput": 3.96,
  "completed": 256,
  "total_input_tokens": 32768,
  "total_output_tokens": 32486
}
```

### Raw AIPerf Artifacts

AIPerf also generates raw output in `.workflow_venvs/.venv_benchmarks_aiperf/artifacts/`:

| File | Description |
|------|-------------|
| `profile_export_aiperf.json` | Aggregated metrics with percentiles |
| `profile_export.jsonl` | Per-request detailed metrics |
| `profile_export_aiperf.csv` | CSV format for analysis |

## Generated Report

The workflow generates a markdown report in `workflow_logs/reports_output/benchmarks_aiperf/`:

| ISL | OSL | Concur | N | TTFT Avg (ms) | TTFT P50 (ms) | TTFT P99 (ms) | TPOT Avg (ms) | TPOT P50 (ms) | TPOT P99 (ms) | Tok/s |
|-----|-----|--------|---|---------------|---------------|---------------|---------------|---------------|---------------|-------|
| 128 | 128 | 1 | 8 | 8905.3 | 95.6 | 46478.1 | 337.9 | 273.2 | 1093.1 | 2.47 |
| 128 | 128 | 32 | 256 | 2396.5 | 2580.1 | 2729.4 | 44.7 | 43.3 | 62.8 | 502.62 |
| 2048 | 128 | 32 | 128 | 27276.2 | 28142.1 | 31245.8 | 285.3 | 278.6 | 312.4 | 3.51 |

## Metrics Reference

| Metric | Description | Unit |
|--------|-------------|------|
| **TTFT** | Time To First Token - latency until first token generated | ms |
| **TPOT** | Time Per Output Token - average inter-token latency | ms |
| **ITL** | Inter-Token Latency - same as TPOT | ms |
| **E2EL** | End-to-End Latency - total request duration | ms |
| **Tok/s** | Output token throughput | tokens/sec |
| **Req/s** | Request throughput | requests/sec |

### Percentile Statistics

Each latency metric includes:
- **Avg (mean)**: Average across all requests
- **P50 (median)**: 50th percentile - typical latency
- **P99**: 99th percentile - worst-case latency
- **std**: Standard deviation

## Comparison: AIPerf vs vLLM Benchmarks

| Feature | `--tools vllm` | `--tools aiperf` |
|---------|----------------|------------------|
| Metrics | Mean only | Mean, P50, P99, std |
| Output format | Basic JSON | Detailed JSON + JSONL (see [Raw AIPerf Artifacts](#raw-aiperf-artifacts)) |
| Report columns | 6 metrics | 12+ metrics |
| Tool source | vLLM repo | NVIDIA ai-dynamo |

> **Note:** The JSONL file (`profile_export.jsonl`) contains per-request metrics and is saved in `.workflow_venvs/.venv_benchmarks_aiperf/artifacts/<model_id>/bench_<isl>_<osl>_<concurrency>/`.

Use `--tools vllm` (default) for quick benchmarks, `--tools aiperf` for detailed performance analysis.

## File Structure

```
workflow_logs/
├── benchmarks_aiperf_output/           # Individual benchmark JSONs
│   ├── aiperf_benchmark_*_isl-128_osl-128_maxcon-1_n-8.json
│   ├── aiperf_benchmark_*_isl-128_osl-128_maxcon-32_n-256.json
│   └── ...
└── reports_output/
    └── benchmarks_aiperf/              # Generated reports
        ├── aiperf_benchmark_display_*.md
        └── data/
            └── aiperf_benchmark_stats_*.csv

.workflow_venvs/.venv_benchmarks_aiperf/
└── artifacts/<model_id>/               # Raw AIPerf output
    └── bench_<isl>_<osl>_<concurrency>/
        ├── profile_export_aiperf.json
        ├── profile_export.jsonl
        └── profile_export_aiperf.csv
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Server not starting | Check Docker logs: `docker logs <container>` |
| All metrics are 0 | Verify server health: `curl http://localhost:8000/health` or delete previously generated json files under `./tt-inference-server/workflow_logs/benchmarks_aiperf_output`|
| Missing dependencies | Delete `.workflow_venvs/.venv_benchmarks_aiperf/` and re-run |
| Authentication errors | Check `JWT_SECRET` is set in `.env` file |
