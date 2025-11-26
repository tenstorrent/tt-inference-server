# Wildcard Run Mode: Non-Deterministic Stress Testing

## Overview

The **wildcard** run mode introduces randomized ISL/OSL variation to stress test models at concurrency and context length boundaries. Unlike the `single` and `multiple` modes which use fixed ISL/OSL values per run, wildcard mode generates unique random ISL/OSL pairs for each prompt within a single run.

### Key Differences from Other Modes

| Aspect | Single | Multiple | Wildcard |
|--------|--------|----------|----------|
| ISL/OSL per run | Fixed | Fixed | Per-prompt random |
| Parameter space | Single combination | Cross-product grid | Random around targets |
| Concurrency | Configurable | Auto-generated | max_concurrency from spec |
| num_prompts | Configurable | Auto-generated | 3x max_concurrency |
| Use case | Quick baseline | Comprehensive benchmark | Stress/chaos testing |
| Predictability | High | High | Low (intentionally) |

## How Wildcard Mode Works

### 1. Context Targets

The parameter space is defined by 4 context target sizes (or 2 in dev mode):

```
For a model with max_context = 8192:
  Target 1: 1/32 of max_context = 256 tokens    (smallest)
  Target 2: 25% of max_context = 2048 tokens
  Target 3: 50% of max_context = 4096 tokens
  Target 4: 75% of max_context = 6144 tokens    (largest)
```

**Dev Mode** uses only targets 1 and 2 for faster iteration.

### 2. Random Generation Strategy

For each prompt:

1. **Select a context target** (evenly distributed across all prompts)
2. **Generate random ISL/OSL split**:
   - ISL ratio randomly chosen between 0.2 and 0.9
   - Example: For 2048 target with 0.75 ratio → ISL ≈ 1536, OSL ≈ 512

3. **Apply variance** (±10% by default):
   - ISL: `center * (1 - 0.10)` to `center * (1 + 0.10)`
   - OSL: Same variance applied independently
   - Example: ISL center=1536 → varies in range [1382, 1690]

4. **Enforce constraints**:
   - Ensure ISL + OSL ≤ max_context
   - If violated, adjust using proportional scaling
   - Ensure minimum values (ISL ≥ 1, OSL ≥ 1)

### 3. Concurrency and Prompts

```
num_prompts = 3 × max_concurrency
```

For a model with `max_concurrency = 32`:
- num_prompts = 96
- Each prompt distributed across 4 context targets (24 prompts per target in full mode)

This creates a **3x load multiplier** relative to concurrency, stressing the system's ability to handle high prompt-to-batch ratios.

## Usage

### Basic Wildcard Mode

Run with automatic parameter generation:

```bash
python spec_tests/run_spec_tests.py \
  --model-spec-json model_config.json \
  --output-path ./results \
  --jwt-secret "your_token" \
  --workflow-args "run-mode=wildcard"
```

**Output Example:**
```
Generated 96 wildcard prompt configurations
  Max concurrency: 32
  Context targets: [256, 2048, 4096, 6144]
  Variance: ±10.0%
  Seed: 42
```

### Fixed ISL Mode

Keep ISL constant at 128, vary only OSL:

```bash
--workflow-args "run-mode=wildcard wildcard-fix-isl=128"
```

**Effect:**
- ISL is always 128 tokens
- OSL varies around target values
- Useful for testing output scaling effects

### Fixed OSL Mode

Keep OSL constant at 128, vary only ISL:

```bash
--workflow-args "run-mode=wildcard wildcard-fix-osl=128"
```

**Effect:**
- OSL is always 128 tokens
- ISL varies around target values (minus OSL space)
- Useful for testing input processing scaling

### Dev Mode

Run only 1/32 and 25% targets for fast iteration:

```bash
--workflow-args "run-mode=wildcard wildcard-dev-mode=true"
```

**Effect:**
- Only 2 context targets instead of 4
- Reduces num_prompts accordingly (48 instead of 96 for concurrency=32)
- Faster test cycles during development

### Custom Variance

Adjust the variance percentage (default 10%):

```bash
--workflow-args "run-mode=wildcard wildcard-variance-pct=0.15"
```

**Effect:**
- ±15% variance instead of ±10%
- Larger ISL/OSL variations
- More aggressive stress testing

### Reproducible Results

Use a seed for deterministic random sequences:

```bash
--workflow-args "run-mode=wildcard wildcard-seed=12345"
```

**Effect:**
- Same seed produces identical ISL/OSL sequences
- Useful for consistent baseline comparisons

### Combined Example

```bash
--workflow-args "run-mode=wildcard wildcard-fix-isl=1024 wildcard-dev-mode=true wildcard-seed=42"
```

This runs:
- Fixed ISL = 1024 tokens
- Only 2 context targets (1/32 and 25%)
- Reproducible with seed 42
- 48 prompts for max_concurrency=32

## Example Parameter Sets

For a model with max_context=8192 and max_concurrency=32:

### Example 1: Full Wildcard Run
```
Target 1 (1/32 = 256):
  Prompt 1: ISL=244, OSL=12
  Prompt 2: ISL=186, OSL=70
  Prompt 3: ISL=268, OSL=5
  (... 21 more prompts)

Target 2 (25% = 2048):
  Prompt 25: ISL=1602, OSL=446
  Prompt 26: ISL=1859, OSL=189
  (... 23 more prompts)

Target 3 (50% = 4096):
  Prompt 49: ISL=3355, OSL=741
  (... 24 more prompts)

Target 4 (75% = 6144):
  Prompt 73: ISL=4863, OSL=1281
  (... 24 more prompts)

Total: 96 prompts with varied sizes
```

### Example 2: Fixed ISL=512
```
All prompts have ISL=512

At target 1 (256 total):
  ISL=512 conflicts with target!
  Adjusted to: ISL=200, OSL=56

At target 2 (2048 total):
  ISL=512, OSL varies around 1536

At target 3 (4096 total):
  ISL=512, OSL varies around 3584

At target 4 (6144 total):
  ISL=512, OSL varies around 5632
```

## Configuration Options

### Workflow Arguments

Pass via `--workflow-args` flag:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `run-mode` | str | - | Set to `wildcard` |
| `wildcard-fix-isl` | int | None | Fix ISL at this value |
| `wildcard-fix-osl` | int | None | Fix OSL at this value |
| `wildcard-dev-mode` | bool | False | Use only 2 context targets |
| `wildcard-variance-pct` | float | 0.10 | Variance as percentage (0.10 = ±10%) |
| `wildcard-seed` | int | 42 | Random seed for reproducibility |

### Format

```bash
--workflow-args "key1=value1 key2=value2 key3=value3"
```

Examples:
- Integer: `wildcard-seed=12345`
- Float: `wildcard-variance-pct=0.20`
- Boolean: `wildcard-dev-mode=true`

## Trace Capture

### Automatic Deduplication

Wildcard mode generates many unique (ISL, OSL) pairs. The system automatically:

1. **Deduplicates** unique pairs for trace capture
2. **Warns** if > 50 unique pairs (trace capture may take several minutes)
3. **Respects** `--disable-trace-capture` flag

### Example

If 96 prompts generate 60 unique (ISL, OSL) pairs:

```
WARNING: Wildcard mode generated 60 unique context lengths -
         trace capture may take several minutes
Capturing 60 unique traces before test execution...
(~5-10 seconds per unique pair = ~5-10 minutes total)
```

### Disable Trace Capture

To skip trace capture and proceed directly to benchmarking:

```bash
python spec_tests/run_spec_tests.py \
  --model-spec-json model_config.json \
  --output-path ./results \
  --jwt-secret "your_token" \
  --workflow-args "run-mode=wildcard" \
  --disable-trace-capture
```

## Result Files

Wildcard mode creates result JSON files named:

```
spec_test_{model_id}_{device}_{timestamp}_wildcard_maxcon-{N}_n-{M}.json
```

Example:
```
spec_test_id_llama31_8b_wh_b0_2025-11-26_14-30-45_wildcard_maxcon-32_n-96.json
```

### Result Content

The JSON includes:
- TTFT (Time To First Token) metrics
- Throughput metrics (tokens/sec)
- Per-prompt latencies
- Percentile metrics (p5, p25, p50, p95, p99)
- ISL/OSL information

## Performance Characteristics

### Execution Time

For a typical model with max_concurrency=32:

| Mode | Prompts | Est. Time |
|------|---------|-----------|
| Single | 1 | 5-10 seconds |
| Multiple (full) | 30-50 | 1-2 minutes |
| Wildcard (full) | 96 | 3-5 minutes |
| Wildcard (dev) | 48 | 1.5-2.5 minutes |

### Trace Capture Time

Adds ~5-10 seconds per unique (ISL, OSL) pair:

- Wildcard (full): ~2-4 minutes (40-50 unique pairs typical)
- Wildcard (dev): ~0.5-1 minute (10-15 unique pairs)

### Total Time Estimates

```
Wildcard (full) = trace (~2 min) + benchmarking (~3 min) = ~5-6 min
Wildcard (dev)  = trace (~1 min) + benchmarking (~2 min) = ~2-3 min
```

## Use Cases

### 1. Stress Testing at Boundaries

Find where the model breaks under extreme conditions:

```bash
--workflow-args "run-mode=wildcard wildcard-variance-pct=0.20"
```

- High variance (±20%)
- Random ISL/OSL splits
- Tests system limits

### 2. Sensitivity Analysis

Test how model responds to different ISL patterns:

```bash
--workflow-args "run-mode=wildcard wildcard-fix-osl=128"
```

- Fixed output size
- Varying input sizes
- Isolates input processing effects

### 3. Quick Regression Testing

Fast baseline validation:

```bash
--workflow-args "run-mode=wildcard wildcard-dev-mode=true wildcard-seed=42"
```

- Only 2 context targets
- Reproducible results
- 2-3 minute runtime

### 4. Load Testing

High concurrency with varied prompts:

```bash
--workflow-args "run-mode=wildcard"
```

- 3x concurrency load (num_prompts = 3x batch)
- Non-deterministic sizes
- Realistic mixed workload

## Constraint Enforcement Examples

### Example 1: Valid Combination

```
Model: max_context = 8192
Target: 2048
Generated: ISL = 1500, OSL = 548
Check: 1500 + 548 = 2048 ≤ 8192 ✓
Result: Used as-is
```

### Example 2: Constraint Violation

```
Model: max_context = 8192
Target: 6144 (75%)
Generated: ISL = 5800 (with variance), OSL = 6100 (with variance)
Check: 5800 + 6100 = 11900 > 8192 ✗
Adjustment: Scale both proportionally
  Scale factor = 8192 / 11900 = 0.688
  Adjusted ISL = 5800 * 0.688 = 3990
  Adjusted OSL = 6100 * 0.688 = 4197
  Check: 3990 + 4197 = 8187 ≤ 8192 ✓
Result: Use adjusted values
```

### Example 3: Fixed ISL Constraint

```
Model: max_context = 8192
Configuration: wildcard-fix-isl=128
Target: 256
Calculation:
  ISL = 128 (fixed)
  Remaining for OSL = 256 - 128 = 128
  OSL varies around 128 (±10% variance)
  OSL range: [115, 141]
Result: ISL=128, OSL ∈ [115, 141]
```

## Troubleshooting

### Q: Too many traces taking too long

**A:** Use dev mode or disable trace capture:
```bash
--workflow-args "run-mode=wildcard wildcard-dev-mode=true"
# or
--disable-trace-capture
```

### Q: Want consistent/reproducible results

**A:** Set a seed:
```bash
--workflow-args "run-mode=wildcard wildcard-seed=12345"
```

### Q: Want less variance/more conservative testing

**A:** Lower variance percentage:
```bash
--workflow-args "run-mode=wildcard wildcard-variance-pct=0.05"
```

### Q: Want to test only one ISL or OSL

**A:** Fix one dimension:
```bash
--workflow-args "run-mode=wildcard wildcard-fix-isl=512"
# or
--workflow-args "run-mode=wildcard wildcard-fix-osl=256"
```

### Q: Getting constraint adjustment warnings

**A:** This is normal. The system automatically adjusts ISL/OSL to satisfy constraints. To see details, check logs for:
```
Adjusted X/96 prompts for constraint compliance
```

## Implementation Details

### Files Modified

1. **spec_tests_args.py** - Wildcard configuration parameters
2. **spec_tests_core.py** - Parameter generation and execution (~200 lines)
3. **spec_tests_benchmarking_script.py** - Per-prompt size support
4. **run_spec_tests.py** - Float argument parsing

### Key Methods

- `_generate_wildcard_mode_params()` - Generates all 96 prompt configurations
- `_generate_random_value_around_center()` - Applies variance with bounds
- `enforce_context_limit()` - Ensures constraint satisfaction (reused from multiple mode)

### Data Flow

```
spec_tests_core.py (parameter generation)
  → generates per_prompt_sizes list
  → writes temporary JSON file
  → passes to benchmarking_script.py

benchmarking_script.py
  → reads JSON config
  → passes to generate_cleaned_random_prompts_using_server()
  → each prompt gets unique ISL/OSL
  → runs benchmark with varied prompts
```

## Best Practices

1. **Start with dev mode** for quick iteration
   ```bash
   --workflow-args "run-mode=wildcard wildcard-dev-mode=true"
   ```

2. **Use a fixed seed** for reproducible baselines
   ```bash
   --workflow-args "run-mode=wildcard wildcard-seed=42"
   ```

3. **Disable trace capture** if running multiple times
   ```bash
   --disable-trace-capture
   ```

4. **Fix one dimension** if testing sensitivity
   ```bash
   --workflow-args "run-mode=wildcard wildcard-fix-isl=512"
   ```

5. **Monitor constraints** if seeing many adjustments
   - Indicates ISL/OSL combinations are aggressive
   - Consider lower variance if concerned

## References

- See `spec_tests/spec_tests_core.py` for implementation
- See plan file at `.claude/plans/delegated-shimmying-naur.md` for design details
