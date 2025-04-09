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
- **`BENCHMARK_CONFIGS`**: Final dictionary mapping all internal model names to their EvalConfig.

---

# Manual run scripts (not recommended)

The additional script in this directory are for manually running benchmarks. These will be deprecated if there are breaking changes in favor of using the automated workflow.

See [manual_benchmarks.md](manual_benchmarks.md) for further detail on how to run benchmarks manually.
