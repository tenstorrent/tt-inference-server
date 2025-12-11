# Performance Benchmarks

Performance Benchmarks for Tenstorrent LLM (model) implementations. These benchmarks determine if a LLM implementation has expected performance.

## Usage: `--workflow benchmarks`

See [Model Readiness Workflows User Guide](../docs/workflows_user_guide.md#performance-benchmarks)

## Benchmarking Tools

The tt-inference-server supports multiple benchmarking tools. Use the `--tools` argument to select:

### vLLM benchmark_serving.py (default)
```bash
python run.py --model <model> --device <device> --workflow benchmarks --docker-server
# or explicitly:
python run.py --model <model> --device <device> --workflow benchmarks --docker-server --tools vllm
```

### GenAI-Perf - NVIDIA Triton SDK tool
```
python run.py --model <model> --device <device> --workflow benchmarks --docker-server --tools genai
```

### AIPerf (ai-dynamo/aiperf)
[AIPerf](https://github.com/ai-dynamo/aiperf) is a comprehensive benchmarking tool that measures the performance of generative AI models.

```bash
python run.py --model <model> --device <device> --workflow benchmarks --docker-server --tools aiperf
```

For manual usage of AIPerf, see [AIPerf Manual Usage Guide](../docs/aiperf_manual.md).

### `run_benchmarks.py`

Purpose: Main script for vLLM benchmarks defined in `BENCHMARK_CONFIGS`, called by `run.py` via `run_workflows.py`.

### `run_genai_benchmarks.py`

Purpose: Docker orchestration script for GenAI-Perf benchmarks using NVIDIA Triton SDK container.

### `run_benchmarks_aiperf.py`

Purpose: Main script for AIPerf benchmarks with detailed percentile metrics and warm-up logic.

#### Workflow

1. Parse CLI runtime arguments
2. Model & Device Validation
3. Wait for vLLM Inference Server to be ready
4. Run all BenchmarkTask in BenchmarkConfig for given model as a subprocess with own python environment

### `benchmark_config.py`

Purpose: defines all static information known ahead of run time for benchmarks to be run for each model implementation including: python environment, benchmark parameters, and expected results.

#### Components

- **`BenchmarkConfig`**: a set of tasks for a specific model implementation.
- **`BenchmarkTask`**: defines python environment to use and mapping of tasks to parameters for each device.
- **`BenchmarkTaskParams`**: Defines how a benchmark should be run.
- **`BENCHMARK_CONFIGS`**: Final dictionary mapping all internal model names to their BenchmarkConfig.

## Benchmark Targets

The reference targets are based on theoretical peformance estimates for each model architecture and hardware combination, for example Llama-3.3-70B on T3K (TT-LoudBox). Model architecture is the set of weights with a common architecture that can be run interchangably (perhaps with small tweaks to hyperparameters), each model architecture is keyed by the 1st weight for the default implementation in `workflows/model_spec.py`.

For example, `Llama-3.3-70B` is the key for
```python
    ModelSpec(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.T3K},
        weights=[
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
    ),
```

The performance targets for each model-hardware combination are defined in `benchmarking/benchmark_targets/model_performance_reference.json` key used is the ModelSpec's 1st model weights model name. This model name e.g. `Llama-3.3-70B` above, uniquely defines the targets for all models weights of the same model architecture. These base theoretical targets are the same for all implementations for the same model architecture and hardware combination. Targets can be added directly to a specific ModelSpec as needed for additional points of comparison.

Then the target is defined in `benchmarking/benchmark_targets/model_performance_reference.json` for `t3k` and `galaxy` hardware:
```json
    "Llama-3.3-70B": {
        "t3k": [
            {
                "isl": 128,
                "osl": 128,
                "max_concurrency": 1,
                "num_prompts": 8,
                "targets": {
                    "theoretical": {
                        "ttft_ms": 38,
                        "tput_user": 28
                    }
                }
            }
        ],
        "galaxy": [
            {
                "isl": 128,
                "osl": 128,
                "max_concurrency": 1,
                "num_prompts": 8,
                "targets": {
                    "theoretical": {
                        "ttft_ms": 50,
                        "tput_user": 80
                    }
                }
            },
            {
                "isl": 2048,
                "osl": 128,
                "max_concurrency": 1,
                "num_prompts": 8,
                "targets": {
                    "theoretical": {
                        "ttft_ms": 800,
                        "tput_user": 80
                    }
                }
            }
        ]
    },
```

## Benchmarks Configuration Controls

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
