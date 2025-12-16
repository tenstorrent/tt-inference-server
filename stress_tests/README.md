## Overview

This code introduces the **Stress Tests workflow**, a comprehensive testing framework designed to validate model
readiness and performance across different parameter configurations. The stress_tests workflow provides both targeted
single-parameter testing and exhaustive multi-parameter matrix testing to ensure models meet performance specifications
before deployment.

## What is the Stress Tests Workflow?

The Stress Tests workflow is a model validation system that:

- **Tests model performance** across various input/output token configurations
- **Validates concurrency handling** at different load levels
- **Measures performance metrics** like TTFT (Time to First Token) and throughput
- **Compares against performance targets** defined in model configurations
- **Generates comprehensive test reports** for model readiness assessment

### Key Features

🎯 **Three Testing Modes:**

- **Single Mode**: Test specific parameter combinations with explicit control
- **Multiple Mode**: Comprehensive cross-product matrix testing across all valid parameter combinations
- **Custom Mode**: User-defined parameter sweeps with full cross-product generation

🔧 **Flexible Parameter Control:**

- Input/output token lengths
- Concurrency levels (batch sizes)
- Number of prompts (user simulation)
- Context length limits
- Custom parameter combinations via workflow-args

📊 **Performance Validation:**

- Automatic extraction of performance targets from model configs
- Comparison against theoretical and reference benchmarks
- Support for different target levels (theoretical, reference, etc.)

⏱️ **Endurance Testing:**

- 24-hour continuous testing mode for stability validation
- Stress testing with high concurrency and prompt volumes

## How It Works

### Architecture

```
stress_tests/
├── __init__.py                          # Import interface
├── run_stress_tests.py                   # Entry point and orchestration
├── stress_tests_args.py                  # Argument consolidation from multiple sources
├── stress_tests_core.py                  # Core workflow logic and execution
├── stress_tests_config.py                # Parameter space generation and validation
├── stress_tests_benchmarking_script.py  # Benchmarking engine and metrics collection
└── stress_tests_summary_report.py       # Result processing and report generation
```

**Consolidated Architecture:**

- **Entry Point** (`run_stress_tests.py`): Handles CLI integration, JWT authentication, and model spec loading
- **Argument Handling** (`stress_tests_args.py`): Consolidates arguments from argparse, model_spec.cli_args, and
  workflow_args with clear precedence rules
- **Core Logic** (`stress_tests_core.py`): Consolidated workflow execution, parameter generation, environment setup, and
  benchmark orchestration
- **Configuration** (`stress_tests_config.py`): Parameter space extraction from model specs, cross-product generation,
  and constraint enforcement
- **Reporting** (`stress_tests_summary_report.py`): JSON result processing, markdown table generation, and CSV export

### Parameter Generation Logic

**Single Mode:**

- Uses explicit parameters from command line arguments
- Calculates input/output sizes based on max_context_length
- Defaults: max_context_length=8192, max_concurrent=1, num_prompts=1

**Multiple Mode:**

- Generates cross-product of all valid parameter combinations
- Uses model-specific constraints (max_context, max_concurrency)
- Includes validated combinations from performance reference data
- Context sizes: max, ~50%, ~10% of model's max_context
- Output sizes: 128, 2048 tokens
- Concurrency: 1, 2, ~50% of max, validated values from model spec
- Prompt patterns: single prompt (1), load-matched (num_prompts = max_concurrent)

**Custom Mode:**

- Accepts user-specified parameter lists via workflow_args
- Generates full cross-product of custom ISL × OSL × Concurrency values
- Supports partial specification (e.g., only custom ISL with default OSL)
- Automatically applies context limit constraints with adjustment

## Usage Examples

### Basic Usage

#### Single Parameter Test

Test a specific configuration with explicit parameters:

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "run_mode=single" --docker-server
```

#### Custom Context Length

Test with a specific context length:

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "run_mode=single max_context_length=4096" --docker-server
```

#### Load Testing with Concurrency

Test with specific concurrency and prompt count:

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "run_mode=single max_context_length=2048 max_concurrent=16 num_prompts=64" --docker-server
```

#### Comprehensive Matrix Testing

Run all valid parameter combinations:

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "run_mode=multiple" --docker-server
```

### Advanced Usage

#### High-Load Stress Testing

```bash
python run.py --model "Qwen2.5-72B-Instruct" --workflow stress_tests --device t3k --workflow-args "run_mode=single max_context_length=8192 max_concurrent=32 num_prompts=160" --docker-server
```

#### Custom Input/Output Sizes

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "run_mode=single input_size=1024 output_size=512 max_concurrent=8 num_prompts=24" --docker-server
```

#### Endurance Testing (24 hours)

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "endurance_mode=true" --docker-server
```

#### Custom Parameter Sweeps

Test across custom ISL values with fixed OSL:

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "custom-isl-values=1024,2048,4096,8192 custom-osl-values=128" --docker-server
```

Test with custom concurrency sweep (auto-generates ISL/OSL combinations):

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "custom-concurrency-values=1,2,4,8,16,32" --docker-server
```

Full custom cross-product (ISL × OSL × Concurrency):

```bash
python run.py --model "Llama-3.1-8B-Instruct" --workflow stress_tests --device n300 --workflow-args "custom-isl-values=1024,2048 custom-osl-values=128,256 custom-concurrency-values=1,16" --docker-server
```

### Model-Specific Examples

#### Large Model Testing (70B+)

```bash
# Qwen2.5-72B on T3K with reduced context for faster testing
python run.py --model "Qwen2.5-72B-Instruct" --workflow stress_tests --device t3k --workflow-args "run_mode=single max_context_length=2048 max_concurrent=16 num_prompts=32" --docker-server

# Full parameter matrix for comprehensive validation
python run.py --model "Qwen2.5-72B-Instruct" --workflow stress_tests --device t3k --workflow-args "run_mode=multiple" --docker-server
```

## Parameter Reference

All stress test parameters are passed via `--workflow-args` as space-separated key=value pairs.

### Run Mode & Configuration

| Parameter            | Description                          | Default    | Example                   |
|----------------------|--------------------------------------|------------|---------------------------|
| `run_mode`           | Testing mode: `single` or `multiple` | `multiple` | `run_mode=single`         |
| `max_context_length` | Maximum context length in tokens     | 8192       | `max_context_length=4096` |
| `endurance_mode`     | Run continuously for 24 hours        | `false`    | `endurance_mode=true`     |

### Single Mode Parameters

| Parameter        | Description                      | Default            | Example             |
|------------------|----------------------------------|--------------------|---------------------|
| `input_size`     | Input token length (ISL)         | 75% of max_context | `input_size=1024`   |
| `output_size`    | Output token length (OSL)        | 128                | `output_size=256`   |
| `max_concurrent` | Concurrent requests (batch size) | 1                  | `max_concurrent=16` |
| `num_prompts`    | Total number of prompts          | 1                  | `num_prompts=64`    |

### Custom Parameter Specification (Multiple Mode)

| Parameter                     | Description                        | Format                           | Example                               |
|-------------------------------|------------------------------------|----------------------------------|---------------------------------------|
| `custom-isl-values`           | Comma-separated ISL values         | `val1,val2,...`                  | `custom-isl-values=1024,2048,4096`    |
| `custom-osl-values`           | Comma-separated OSL values         | `val1,val2,...`                  | `custom-osl-values=128,256`           |
| `custom-concurrency-values`   | Comma-separated concurrency values | `val1,val2,...`                  | `custom-concurrency-values=1,2,16,32` |
| `custom-num-prompts-strategy` | Strategy for num_prompts           | `match_concurrency` or `fixed:N` | `custom-num-prompts-strategy=fixed:8` |

**Note:** Custom parameters generate a full cross-product and override automatic parameter generation. Supports partial
specification (e.g., only custom-isl-values with auto-generated OSL).

### Advanced Filtering

| Parameter                | Description                                 | Default | Example                       |
|--------------------------|---------------------------------------------|---------|-------------------------------|
| `only_match_concurrency` | Filter to only num_prompts = max_concurrent | `false` | `only_match_concurrency=true` |
| `use_server_tokenizer`   | Use server-side tokenization                | `false` | `use_server_tokenizer=true`   |

### Top-Level Flags

| Parameter                 | Description                           | Example                   |
|---------------------------|---------------------------------------|---------------------------|
| `--disable-trace-capture` | Skip trace capture (already captured) | `--disable-trace-capture` |
| `--percentile-report`     | Include detailed percentile metrics   | `--percentile-report`     |

### Common Usage Patterns

```bash
# Single test
--workflow-args "run_mode=single max_concurrent=1 num_prompts=1"

# Load test (prompts = concurrency)
--workflow-args "run_mode=single max_concurrent=16 num_prompts=16"

# Stress test (prompts > concurrency)
--workflow-args "run_mode=single max_concurrent=32 num_prompts=160"

# Custom ISL sweep
--workflow-args "custom-isl-values=1024,2048,4096 custom-osl-values=128"

# Custom concurrency sweep (auto ISL/OSL)
--workflow-args "custom-concurrency-values=1,2,4,8,16,32"
```

## Output and Reports

### Test Execution Output

```
Stress Tests: Llama-3.1-8B-Instruct_n300 on n300
Mode: single | Total combinations: 1

Test 6144/128 (ISL/OSL) | 1x1 (conc×prompts)
```

### Generated Test Combinations (Multiple Mode)

The workflow automatically generates a comprehensive test matrix and displays it in a clear table format:

```
## Test Parameter Combinations
| # | ISL | OSL | Max Seq | Concurrency | Prompts | Source | Adjusted |
|---|-----|-----|---------|-------------|---------|--------|----------|
|  1 | 127872 | 128 | 128000 |           1 |       1 | auto   |          |
|  2 | 127872 | 2048| 129920 |           1 |       1 | auto   |     ✓    |
|  3 | 63872  | 128 |  64000 |           2 |       2 | cross_ |          |
|  4 | 10752  | 128 |  10880 |          16 |      16 | custom |          |

**Total**: 24 combinations
**Adjusted**: 8 combinations were adjusted for context limit compliance

Note: Source indicates parameter origin (auto: algorithmic, cross_: cross-product, custom: user-specified)
```

### Performance Metrics

Each test automatically collects comprehensive metrics:

**Core Latency Metrics:**

- **TTFT (Time to First Token)**: Latency to first response token
- **TPOT (Time per Output Token)**: Average time between tokens
- **ITL (Inter-Token Latency)**: Token generation consistency
- **E2EL (End-to-End Latency)**: Total request completion time

**Throughput Metrics:**

- **Tokens per second (TPS)**: User-level and system-level throughput
- **Requests per second (RPS)**: Request completion rate

**Percentile Distributions (Automatic):**

All metrics include percentile distributions calculated across all requests:

- **p05** (5th percentile): Best-case performance
- **p25** (25th percentile): Better than average
- **p50** (50th percentile / median): Typical performance
- **p95** (95th percentile): Worst 5% of cases
- **p99** (99th percentile): Tail latency

Percentiles are calculated for: TTFT, TPOT, ITL, and E2EL. Use `--percentile-report` flag for detailed percentile
breakdowns in reports.

**Statistical Measures:**

- Mean and standard deviation for all latency metrics
- Error rates and failed request percentage

## Integration with Model Specs

The stress_tests workflow automatically integrates with model specs to:

1. **Extract Performance Targets**: Reads theoretical and reference performance targets
2. **Validate Parameter Bounds**: Ensures tests stay within model constraints
3. **Use Validated Combinations**: Incorporates pre-tested parameter sets
4. **Apply Device Constraints**: Respects device-specific limitations

### Example Model Spec Integration

```python
# From model_performance_reference.json
"Llama-3.1-8B": {
    "n300": [
        {
            "isl": 128,
            "osl": 128,
            "max_concurrency": 1,
            "num_prompts": 8,
            "targets": {
                "theoretical": {
                    "ttft_ms": 16,
                    "tput_user": 66
                }
            }
        }
    ]
}
```

### Context Limit Enforcement

The workflow intelligently handles context limit constraints:

- **Automatic Adjustment**: When ISL + OSL exceeds max_context, parameters are adjusted using configurable policies
- **Policy Options**: "neutral" (proportional scaling), "preserve_isl", "preserve_osl"
- **Transparency**: All adjustments are logged and reported in the test results

## Benefits

### For Development Teams

- **Early Detection**: Catch performance regressions before deployment
- **Parameter Optimization**: Find optimal configurations for different use cases
- **Load Planning**: Understand capacity limits and scaling characteristics

### For QA/Testing

- **Comprehensive Coverage**: Test all valid parameter combinations automatically
- **Reproducible Results**: Consistent test parameters and environments
- **Performance Baselines**: Establish and track performance targets

### For DevOps/Infrastructure

- **Capacity Planning**: Understand resource requirements at scale
- **SLA Validation**: Verify models meet performance commitments
- **Deployment Readiness**: Comprehensive pre-deployment validation

## Technical Implementation

### Argument Consolidation Architecture

The `StressTestsArgs` dataclass (`stress_tests_args.py`) consolidates arguments from three sources with clear
precedence:

**1. Argparse (CLI flags):**

- `--model-spec-json`: Model specification file path
- `--output-path`: Results output directory
- `--jwt-secret`: Authentication token

**2. Model Spec CLI Args (`model_spec.cli_args`):**

- Core configuration: `model`, `device`, `service_port`
- Trace control: `disable_trace_capture`
- All top-level run.py flags passed through

**3. Parsed Workflow Args (`--workflow-args`):**

- Run configuration: `run_mode`, `max_context_length`, `endurance_mode`
- Test parameters: `input_size`, `output_size`, `max_concurrent`, `num_prompts`
- Custom parameters: `custom-isl-values`, `custom-osl-values`, `custom-concurrency-values`
- Filters: `only_match_concurrency`, `use_server_tokenizer`

**Precedence Rules:**

- Custom parameters (`custom-*`) override `run_mode` automatic generation
- Explicit values always take precedence over calculated defaults
- Model spec constraints enforced regardless of user input (with adjustment, not rejection)

### Core Workflow Engine

The `StressTests` class in `stress_tests_core.py` provides:

- **Environment Setup**: Automatic configuration of required environment variables
- **Parameter Generation**: Intelligent parameter space exploration supporting three modes (single, multiple, custom)
- **Constraint Enforcement**: Context limit validation with configurable adjustment policies (neutral, preserve_isl,
  preserve_osl)
- **Trace Capture Optimization**: Captures all unique (ISL, OSL) pairs once upfront, not per test
- **Benchmark Orchestration**: Subprocess execution of benchmarking script with isolated environments
- **Result Management**: Organized output file generation with detailed naming conventions

### Key Features

- **Smart Defaults**: Automatic parameter calculation based on model specifications (75% ISL, 128 OSL)
- **Trace Capture**: One-time upfront trace collection for all unique context lengths, skipped in subsequent subprocess
  calls
- **Health Monitoring**: Server health validation before test execution with configurable timeout
- **Error Handling**: Comprehensive error reporting and graceful failure handling
- **Progress Tracking**: Real-time test progress with simplified logging (ISL/OSL | conc×prompts format)
- **Partial Specification Support**: Specify only ISL or only OSL, system infers the other
- **Cross-Product Generation**: Full cartesian product of custom parameter lists with deduplication

### Workflow Integration

- Integrates seamlessly with existing `run.py` CLI interface via WorkflowType.STRESS_TESTS
- Uses standard workflow patterns (docker-server, logging, JWT authentication, model spec JSON)
- Supports all workflow arguments and overrides with explicit validation
- Compatible with existing model configuration system (MODEL_SPECS, performance_reference)
- Maintains compatibility with workflow orchestration (run_workflows.py) and reporting systems (
  stress_tests_summary_report.py)
- Included in RELEASE workflow alongside evals and benchmarks

---

This stress_tests workflow provides a robust foundation for model validation and performance testing, ensuring models
meet quality and performance standards before deployment to production environments. The clean architecture enables easy
maintenance and extension while providing comprehensive testing capabilities across the entire parameter space.
