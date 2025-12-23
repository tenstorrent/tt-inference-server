# Test Categorization and Marking System

## Overview

This document describes the test marking system implemented for the TT Inference Server test suite. The system enables selective test execution based on test characteristics, hardware requirements, and model categories.

## Architecture

The marking system is implemented through:

1. **Configuration v2.0** in `server_tests_config.json` with:
   - `test_templates`: Reusable test definitions with default markers
   - `test_suites`: Model/device specific configurations
   - `prerequisite_tests`: Tests that run before all others (DeviceLivenessTest)
   - `hardware_defaults`: Device-specific defaults (num_devices, retry_attempts)
   - `model_categories`: Category to model mappings

2. **TestFilter** in `test_categorization_system/test_filter.py` for programmatic test selection with:
   - Auto-derivation of markers from context (model, device, category)
   - Prerequisite test injection
   - Method chaining for filter composition

3. **CLI Support** in `run.py` with marker-based filtering arguments

## Key Features

### Auto-Derived Markers

Markers are automatically derived from context, reducing manual configuration:

- **Model category** → `image`, `audio`, `cnn`, `llm`
- **Specific model** → `sdxl`, `whisper`, `distil_whisper`, etc.
- **Hardware target** → `n150`, `n300`, `t3k`, `galaxy`
- **Template markers** → `load`, `param`, `smoke`, etc.

### Prerequisite Tests

DeviceLivenessTest automatically runs before all other tests for each model/device combination. This ensures hardware is ready before running actual tests.

To skip prerequisites:
```bash
python run.py --model sdxl --device n150 --skip-prerequisites
```

### Test Templates

Common test configurations are defined once in `test_templates` and referenced in test suites:

```json
"test_templates": {
    "ImageGenerationLoadTest": {
        "module": "tests.server_tests.test_cases.image_generation_load_test",
        "markers": ["load", "e2e", "slow", "heavy"],
        "test_config": {
            "test_timeout": 3600,
            "retry_attempts": 1
        }
    }
}
```

Test suites reference templates with overrides:
```json
{
    "template": "ImageGenerationLoadTest",
    "description": "Test image generation time for 20 iterations",
    "targets": {
        "num_inference_steps": 20,
        "image_generation_time": 10
    }
}
```

## Marker Categories

### Test Types
| Marker | Description | Use Case |
|--------|-------------|----------|
| `unit` | Fast, isolated tests | Development feedback loops |
| `integration` | Medium-speed tests with local dependencies | Pre-commit validation |
| `load` | Load testing with concurrent requests | Performance validation |
| `param` | Parameter validation tests | API contract testing |
| `functional` | Functional correctness tests | Feature verification |
| `smoke` | Quick sanity checks | Basic functionality |
| `e2e` | Full system tests | Complete workflows |
| `stability` | Long-running stability tests | Reliability validation |
| `eval` | Model evaluation tests | Quality metrics |

### Performance Characteristics
| Marker | Description |
|--------|-------------|
| `slow` | Tests taking >30 seconds |
| `heavy` | Resource-intensive tests |
| `fast` | Tests completing in <10 seconds |

### Hardware Targets
| Marker | Description |
|--------|-------------|
| `n150` | N150 device (1 chip) |
| `n300` | N300 device (2 chips) |
| `t3k` | T3K device (4 chips) |
| `galaxy` | Galaxy device (32 chips) |

### Model Categories
| Marker | Description |
|--------|-------------|
| `image` | Image generation/processing |
| `audio` | Audio transcription |
| `cnn` | CNN models |
| `llm` | Large Language Models |

### Specific Models
| Marker | Description |
|--------|-------------|
| `sdxl` | Stable Diffusion XL |
| `sdxl_img2img` | SDXL img2img |
| `sdxl_inpaint` | SDXL inpainting |
| `sd35` | Stable Diffusion 3.5 |
| `whisper` | Whisper model |
| `distil_whisper` | Distil Whisper |

## Usage Examples

### Use Case 1: Run All IMAGE/AUDIO Tests on Specific Hardware

```bash
# All IMAGE tests on N150
python run.py --model-category IMAGE --device n150

# All IMAGE and AUDIO tests on all hardware
python run.py --model-category IMAGE AUDIO

# Programmatically
from tests.server_tests.test_categorization_system import TestFilter

filter = TestFilter()
tests = filter.filter_by_model_category(["IMAGE", "AUDIO"]) \
              .filter_by_device("n150") \
              .get_tests()
```

### Use Case 2: Run Specific Model on Specific Hardware with Category

```bash
# SDXL load tests on N150
python run.py --model stable-diffusion-xl-base-1.0 --device n150 --markers load

# Programmatically
filter = TestFilter()
tests = filter.filter_by_model("stable-diffusion-xl-base-1.0") \
              .filter_by_device("n150") \
              .filter_by_markers(["load"]) \
              .get_tests()
```

### Use Case 3: Run Specific Tests with Full Filtering

```bash
# Specific test class for SDXL on N150
python run.py --model stable-diffusion-xl-base-1.0 --device n150 --markers load --test-name ImageGenerationLoadTest

# Programmatically
filter = TestFilter()
tests = filter.filter_by_model("stable-diffusion-xl-base-1.0") \
              .filter_by_device("n150") \
              .filter_by_markers(["load"]) \
              .filter_by_test_name("ImageGenerationLoadTest") \
              .get_tests()
```

### Use Case 4: Run All Performance/Load Tests

```bash
# All load tests across all models
python run.py --markers load

# All load and eval tests
python run.py --markers load eval

# Programmatically
filter = TestFilter()
tests = filter.filter_by_markers(["load"]).get_tests()
```

### Additional Examples

```bash
# Fast smoke tests only
python run.py --markers smoke fast --match-all-markers

# Exclude slow/heavy tests (for CI)
python run.py --device n150 --exclude-markers slow heavy

# List all available markers
python run.py --list-markers

# Preview which tests would run (dry-run)
python run.py --model-category IMAGE --device n150 --list-tests
```

## CLI Reference

```
usage: run.py [-h] [--model MODEL] [--device DEVICE]
              [--model-category CATEGORY [CATEGORY ...]]
              [--markers MARKER [MARKER ...]] [--match-all-markers]
              [--exclude-markers MARKER [MARKER ...]]
              [--test-name TEST_NAME] [--skip-prerequisites]
              [--list-markers] [--list-tests] [--dry-run]

Arguments:
  --model MODEL           Filter by model name
  --device DEVICE         Filter by device (n150, n300, t3k, galaxy)
  --model-category        Filter by model category (IMAGE, AUDIO, CNN)
  --markers               Filter by test markers
  --match-all-markers     Require ALL markers to match
  --exclude-markers       Exclude tests with these markers
  --test-name             Filter by specific test class name
  --skip-prerequisites    Skip prerequisite tests (DeviceLivenessTest)
  --list-markers          List available markers and exit
  --list-tests            List matching tests without running
```

## Adding New Tests

### Step 1: Define Template (if needed)

If your test type doesn't exist, add a template to `test_templates`:

```json
"NewTestType": {
    "module": "tests.server_tests.test_cases.new_test_type",
    "markers": ["e2e", "slow"],
    "test_config": {
        "test_timeout": 3600,
        "retry_attempts": 1,
        "retry_delay": 60
    }
}
```

### Step 2: Add to Test Suite

Add your test case to the appropriate suite in `test_suites`:

```json
{
    "template": "NewTestType",
    "enabled": true,
    "description": "Description of what this test does",
    "targets": {
        "custom_param": "value"
    }
}
```

Markers are **automatically derived**:
- From the template (`e2e`, `slow`)
- From model category (`image`, `audio`, etc.)
- From model marker (`sdxl`, `whisper`, etc.)
- From device (`n150`, `t3k`, etc.)

### Step 3: Add Custom Markers (Optional)

If you need additional markers beyond auto-derived ones, add them in the template:

```json
"test_config": {
    ...
},
"markers": ["e2e", "slow", "regression"]  // "regression" is custom
```

## Configuration Reference

### Hardware Defaults

Located in `hardware_defaults`, defines device-specific values:

```json
"hardware_defaults": {
    "n150": {
        "num_of_devices": 1,
        "liveness_retry_attempts": 80
    },
    "t3k": {
        "num_of_devices": 4,
        "liveness_retry_attempts": 120
    }
}
```

These are automatically applied to:
- DeviceLivenessTest `retry_attempts`
- Test `targets.num_of_devices` (unless overridden)

### Test Suite Structure

```json
{
    "id": "sdxl-n150",           // Unique identifier
    "weights": ["model-name"],   // Model(s) this suite tests
    "device": "n150",            // Target hardware
    "model_marker": "sdxl",      // Specific model marker
    "test_cases": [...]          // List of test case configs
}
```

## Programmatic API

### TestFilter Methods

```python
from tests.server_tests.test_categorization_system import TestFilter

filter = TestFilter()

# Filtering methods (chainable)
filter.filter_by_model_category(["IMAGE", "AUDIO"])
filter.filter_by_model("stable-diffusion-xl-base-1.0")
filter.filter_by_device("n150")
filter.filter_by_markers(["load", "smoke"], match_all=False)
filter.filter_by_test_name("ImageGenerationLoadTest")
filter.exclude_markers(["slow", "heavy"])
filter.include_prerequisites(True)

# Get results
tests = filter.get_tests()           # List of suite dicts
flat = filter.get_flat_tests()       # Flat list of test dicts
count = filter.get_test_count()      # Total test count

# Utilities
filter.print_summary()               # Print filtered test summary
filter.reset()                       # Reset all filters
filter.get_available_markers()       # Get marker definitions
filter.get_all_devices()             # List all devices
filter.get_all_models()              # List all models
```

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  smoke-tests:
    runs-on: self-hosted
    steps:
      - name: Run smoke tests
        run: python tests/server_tests/run.py --markers smoke --device n150

  load-tests:
    runs-on: self-hosted
    needs: smoke-tests
    steps:
      - name: Run load tests
        run: python tests/server_tests/run.py --markers load --device n150

  full-suite:
    runs-on: self-hosted
    needs: load-tests
    steps:
      - name: Run all tests
        run: python tests/server_tests/run.py --device n150
```

### Environment Variables

```bash
SERVICE_PORT=8000      # Service port for tests
TEST_TIMEOUT=60        # Default test timeout
TEST_RETRIES=2         # Default retry count
MODEL=sdxl             # Default model filter
DEVICE=n150            # Default device filter
```

## Best Practices

1. **Use templates**: Define common test configurations once
2. **Let markers auto-derive**: Don't manually add model/device markers
3. **Keep descriptions clear**: Help others understand test purpose
4. **Group by model/device**: One suite per model/device combination
5. **Use meaningful targets**: Document expected values
6. **Run smoke tests first**: Use `--markers smoke` in CI for quick feedback
7. **Exclude heavy tests in dev**: Use `--exclude-markers slow heavy`

## Troubleshooting

### No tests match filters

```bash
# List available options
python run.py --list-markers
python run.py --list-tests

# Check filter results
python run.py --model-category IMAGE --list-tests
```

### Test not loading

Check that:
1. Module path is correct in template
2. Class name matches template name
3. Test file has no syntax errors
4. Required dependencies are installed

### Markers not applied

Verify:
1. Template has `markers` array
2. Suite has `model_marker` field
3. Suite has `device` field
4. Test case references correct template
