# Performance Benchmarks

Performance Benchmarks for Tenstorrent LLM (model) implementations. These benchmarks determine if a LLM implementation has expected performance.

## Usage: `--workflow benchmarks`

See [Model Readiness Workflows User Guide](../docs/workflows_user_guide.md#performance-benchmarks)

### `run_benchmarks.py`

Purpose: Main script for performance benchmarks defined in `BENCHMARK_CONFIGS`, called by `run.py` via `run_workflows.py`.

### Workflow

1. Parse CLI runtime arguments
2. Model & Device Validation
3. Wait for vLLM Inference Server to be ready
4. Run all BenchmarkTask in BenchmarkConfig for given model as a subprocess with own python environment

### `benchmark_config.py`

Purpose: defines all static information known ahead of run time for evaluations to be run for each model implementation including: python environment, eval parameters, scoring methods, and expected results.

#### Components

- **`BenchmarkConfig`**: a set of tasks for a specific model implementation.
- **`BenchmarkTask`**: defines python environment to use and mapping of tasks to parameters. The mappings include for each device. Uses vLLM [benchmarks/benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py).
- **`BenchmarkTaskParams`**: Defines how a benchmark should be run.
- **`BENCHMARK_CONFIGS`**: Final dictionary mapping all internal model names to their BenchmarkConfig.

## benchmarks controls

turn off benchmark sweeps with `ONLY_BENCHMARK_TARGETS`:
```
export ONLY_BENCHMARK_TARGETS=1
```
override the benchmark targets with `OVERRIDE_BENCHMARK_TARGETS`, and the benchmark params that are used via a JSON file like benchmarking/benchmark_targets/test_benchmarks_override.json:
```
export OVERRIDE_BENCHMARK_TARGETS=/home/my_user/tt-inference-server/benchmarking/benchmark_targets/test_benchmarks_override.json
```

To check you have the right params set for your benchmarking run, before the benchmarks start running you'll see a summary table of the parameters to be run:
```
2025-05-02 18:54:04,602 - run_benchmarks.py:184 - INFO: Running benchmarks for:
#   isl        osl        max_concurrency num_prompts
--- ---------- ---------- --------------- ------------
1   128        128        1               8
2   128        1024       1               4
3   1024       128        1               4
4   2048       128        1               4
5   3072       128        1               4
6   4096       128        1               2
7   8192       128        1               2
8   16384      128        1               2
9   32000      128        1               2
10  128        128        32              256
11  128        1024       32              128
12  2048       128        32              128
13  2048       2048       32              64
14  3000       64         32              128
15  4000       64         32              128
16  4500       64         32              64
17  8000       64         32              64
18  16000      64         32              64
```

---

# Manual run scripts (not recommended)

The additional script in this directory are for manually running benchmarks. These will be deprecated if there are breaking changes in favor of using the automated workflow.

See [manual_benchmarks.md](manual_benchmarks.md) for further detail on how to run benchmarks manually.
