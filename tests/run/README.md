# Model Readiness Tests

Model Readiness Tests for Tenstorrent LLM (model) implementations. These tests determine if a LLM implementation behaves as expected during .

## Usage: `--workflow tests`



### `run_tests.py`

Purpose: Main script for model tests defined in `MODEL_CONFIGS`, called by `run.py` via `run_workflows.py`.

### Workflow

1. Parse CLI runtime arguments
2. Model & Device Validation
3. Wait for vLLM Inference Server to be ready
4. Run all TestTasks for given model as a subprocess with own python environment

### `tests_config.py`

Purpose: defines all parameter space to be tested across, i.e., size of random prompts to be sent. 
#### Components

- **`TestsConfig`**: Holds test setup details (e.g., model repo, device) and initializes the parameter space for tests.
- **`TestParamSpace`**: Defines the range of test parameters (e.g., trimmed context length, sequence values) based on the model and device.
- **`TestTask`**:Generates test parameter combinations. Supports both "single" mode (fixed parameters) and "multiple" mode (combinatorial benchmarks).
- **`TestRun`**: Executes each test by building and running a benchmark command, managing logging, and capturing results.
- **`Tests`**: Orchestrates the entire test workflow by setting up the environment, iterating through test tasks, and triggering test runs.

---

# Manual run scripts (not recommended)

These will be deprecated if there are breaking changes in favor of using the automated workflow.
