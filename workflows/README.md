# tt-inference-server workflow runner

This project provides a command-line interface (CLI) to run various workflows related to the Tenstorrent inference server. It supports executing workflows locally or via Docker, handling environment setup, dependency management, and logging for multiple models and workflow types.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [run.py CLI Usage](#runpy-cli-usage)
  - [Docker Volume Options](#docker-volume-options)
  - [Print Docker Command](#print-docker-command)
- [Container Interface](#container-interface)
- [Client Side Scripts](#client-side-scripts)
- [Workflow Setup](#workflow-setup)
- [Project Structure](#project-structure)
- [Error Handling](#error-handling)

## Overview

The inference server has two independent interfaces:

1. **`run.py`** (host-side) -- optionally used to template the `docker run` command, validate the runtime, configure host setup, and run client-side workflows (`benchmarks`, `evals`).
2. **Container interface** (`run_vllm_api_server.py`) -- can be used independently from `run.py` via a direct `docker run` command, accepting `--model` and `--tt-device` to self-resolve the model spec from a bundled JSON. See the [container interface documentation](../vllm-tt-metal-llama3/README.md#container-interface-direct-docker-run).

The module `workflows/run_local.py` is responsible for setting up the local execution environment. It handles tasks such as bootstrapping a virtual environment, installing dependencies, configuring workflow-specific settings, and finally launching the workflow script.

## Features

- **Multiple Workflows**: Run benchmarks, evals, server, release, and report workflows.
- **Execution Modes**: Choose between running workflows locally or in Docker mode.
- **Automatic Setup**: Manages environment setup, including virtual environments and dependency installation.
- **Logging**: Detailed logging for tracking execution, errors, and debugging.

## Example diagram for benchmarks workflow

![inference-server-workflow-diagram-2025-08-14-1106.png](inference-server-workflow-diagram-2025-08-14-1106.png)

The workflows that run end to end tests on the inference server (benchmarks, evals, tests, spec_tests, and stress_tests) all follow the same pattern in configuring how to send HTTP requests to the inference server.

## Prerequisites

    Python 3.8+: Required to run the CLI and setup scripts.
    Docker: Needed if running workflows in Docker mode.
    Git: Required for cloning repositories during setup (e.g., for the llama-cookbook used in meta evals).

## Installation

Clone the Repository:
```
git clone https://github.com/yourusername/tt-inference-server.git
cd tt-inference-server
```

The workflows automatically create their own virtual environments as needed. You can execute the CLI directly using Python:
```
python run.py --model <model_name> --workflow <workflow_type> --tt-device <device_type>
```
Dependencies:

Required dependencies are installed during the workflow setup process. Ensure you have internet connectivity for downloading packages and cloning any necessary repositories.

## run.py CLI Usage

`run.py` is the host-side automation CLI. It can optionally be used to:
1. Template the `docker run` command for the [container interface](../vllm-tt-metal-llama3/README.md#container-interface-direct-docker-run)
2. Validate the runtime environment
3. Configure the host setup (weights download, volume creation)
4. Run client-side workflows (`benchmarks`, `evals`)

```
Usage: python3 run.py --model <model> --workflow <workflow> [options]
```

### Command-line Arguments

**Required Arguments:**

| Argument | Description |
|---|---|
| `--model` | Model to run. Available models are defined in `MODEL_SPECS`. |
| `--workflow` | Workflow to run: `benchmarks`, `evals`, `server`, `release`, `reports`, `tests`. |

**Model and Device Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--tt-device` | Largest supported device on host | Target device: `n150`, `n300`, `p100`, `p150`, `t3k`, `galaxy`. The legacy alias `--device` is still supported. |
| `--impl` | Model spec default | Implementation option. If not specified, the default implementation for the model and device is inferred. |
| `--engine` | Model spec default | Inference engine override: `vllm`, `media`, `forge`. |

**Server Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--docker-server` | false | Run inference server in a Docker container. |
| `--local-server` | false | Run inference server on localhost. |
| `-it`, `--interactive` | false | Run Docker in interactive mode. |
| `--service-port` | `8000` | Service port. Also reads from `SERVICE_PORT` env var. |
| `--no-auth` | false | Disable vLLM API key authorization (skips `JWT_SECRET` requirement). |
| `--print-docker-cmd` | false | Print the Docker run command and exit without starting the server. |

**Docker Volume Options:**

| Argument | Default | Description |
|---|---|---|
| `--host-volume` | None (Docker named volume) | Host directory for persistent cache volume (bind mount). |
| `--host-hf-cache` | None | Host HuggingFace cache directory to mount readonly for model weights. |
| `--host-weights-dir` | None | Host directory with pre-downloaded model weights to mount into the container. |
| `--image-user` | `1000` | UID passed to `docker run --user`. Set to match host user UID for correct bind mount permissions. |

Only one of `--host-volume`, `--host-hf-cache`, `--host-weights-dir` can be specified. See [Docker Volume Options](#docker-volume-options) for details.

**Advanced Arguments:**

| Argument | Description |
|---|---|
| `--dev-mode` | Enable developer mode (bind mounts source code into container). |
| `--override-docker-image` | Override the Docker image used by `--docker-server`. |
| `--device-id` | Tenstorrent device IDs, comma-separated PCI indices (e.g. `0` or `0,1,2`). |
| `--override-tt-config` | Override TT config as JSON string (e.g., `'{"data_parallel": 16}'`). |
| `--vllm-override-args` | Override vLLM arguments as JSON string (e.g., `'{"max_model_len": 4096}'`). |
| `--disable-trace-capture` | Disable trace capture requests to speed up execution. |
| `--workflow-args` | Additional workflow arguments (e.g., `'param1=value1 param2=value2'`). |

### Secrets

Secrets can be provided via a `.env` file in the repository root or as environment variables:

```bash
# Option 1: .env file (automatically loaded by run.py)
HF_TOKEN=hf_...
JWT_SECRET=my-secret-string

# Option 2: environment variables
export HF_TOKEN=hf_...
export JWT_SECRET=my-secret-string
```

### Docker Volume Options

When running with `--docker-server`, `run.py` supports three mutually exclusive strategies for how model weights and caches are persisted. Only one can be specified at a time.

**1. Docker named volume (default)**

No flags needed. A Docker named volume is created automatically for model weights and TT Metal caches. Weights are downloaded inside the container on first start.

```bash
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --docker-server
```

**2. Host persistent volume (`--host-volume`)**

Bind mounts an entire host directory as the container's `cache_root`. All data (weights, TT Metal caches) lives on the host filesystem.

```bash
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --docker-server \
  --host-volume /mnt/data/tt-cache
```

**3. Host HuggingFace cache (`--host-hf-cache`)**

Mounts the host's existing HuggingFace cache directory readonly into the container. Other persistent data (TT Metal caches) uses a Docker named volume.

```bash
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --docker-server \
  --host-hf-cache ~/.cache/huggingface
```

**4. Host weights directory (`--host-weights-dir`)**

Mounts a host directory containing pre-downloaded model weights readonly into the container. Other persistent data uses a Docker named volume.

```bash
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --docker-server \
  --host-weights-dir /mnt/models/Llama-3.1-8B-Instruct
```

### Print Docker Command

Use `--print-docker-cmd` to output the generated `docker run` command without starting the server. This is useful for inspecting or customizing the command before running it manually.

```bash
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --docker-server --print-docker-cmd
```

### Example Commands

Run the evals workflow:
```bash
python3 run.py --model Qwen2.5-72B-Instruct --workflow evals --tt-device N150
```

Run a workflow with a Docker server:
```bash
python3 run.py --model Llama-3.3-70B-Instruct --workflow evals --tt-device T3K --docker-server
```

Run benchmarks workflow:
```bash
python3 run.py --model Llama-3.3-70B-Instruct --workflow benchmarks --tt-device T3K
```

Run server workflow in Docker with interactive mode:
```bash
python3 run.py --model Llama-3.3-70B-Instruct --workflow server --tt-device T3K --docker-server --interactive
```

Run with custom service port and additional workflow arguments:
```bash
python3 run.py --model Qwen2.5-72B-Instruct --workflow evals --tt-device N150 --service-port 9000 --workflow-args "batch_size=4 max_tokens=512"
```

## Container Interface

The inference server container can be used independently from `run.py` via a direct `docker run` command. See the full [container interface documentation](../vllm-tt-metal-llama3/README.md#container-interface-direct-docker-run) for details, including CLI args, secrets, and persistent volume overrides.

## Client Side Scripts

The `run.py` CLI can be used to run client-side workflows (benchmarks and evals) against an external vLLM server that is already running and serving traffic. This is useful when you have a vLLM server deployed separately and want to run evaluations or benchmarks against it without managing the server lifecycle through `run.py`.

### Prerequisites for Client Side Usage

1. **External vLLM Server**: You must have a vLLM server already running and accessible via HTTP/HTTPS
2. **Model Compatibility**: The external server must be serving a model that is defined in the `MODEL_SPECS`
3. **Network Access**: The client machine running `run.py` must have network access to the vLLM server

### Supported Client Side Workflows

The following workflows can be run as client-side scripts against an external vLLM server:

- **benchmarks**: Performance benchmarking against the external server
- **evals**: Model evaluation and testing against the external server

### Configuration

To use `run.py` with an external vLLM server, you need to configure the server endpoint:

1. **Set the SERVICE_PORT environment variable** to match your external server's port:
   ```bash
   export SERVICE_PORT=8000  # Replace with your server's port
   ```

2. **[optional] Set the server JWT secret** for authorization (if set on server):
   ```bash
   export JWT_SECRET=my-string-secret
   ```

### Usage Examples

Run benchmarks against an external vLLM server:
```bash
# Server running on localhost:8000
python3 run.py --model Llama-3.3-70B-Instruct --workflow benchmarks --tt-device T3K --disable-trace-capture

# can use --service-port or SERVICE_PORT env var to set another port
SERVICE_PORT=9000 python3 run.py --model Qwen2.5-72B-Instruct --workflow benchmarks --tt-device N150 --disable-trace-capture
```

Run evaluations against an external vLLM server:
```bash
# Server running on localhost:8000
python3 run.py --model Llama-3.3-70B-Instruct --workflow evals --tt-device T3K --disable-trace-capture

# can use --service-port or SERVICE_PORT env var to set another port
python3 run.py --model Qwen2.5-72B-Instruct --workflow evals --tt-device N150 --disable-trace-capture --service-port 7592
```

Run multiple model inference servers, each must be on a separate card
```bash
# run model on multiple devices
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --tt-device n300 --docker-server --dev-mode --device-id 0
python3 run.py --model Llama-3.1-8B-Instruct --workflow server --tt-device n300 --docker-server --dev-mode --device-id 1
```

### Important Notes

- **Use `--disable-trace-capture`**: When running against an external server, it's recommended to use the `--disable-trace-capture` flag to speed up execution, especially if the server is already running and traces have been captured previously.

- **Model Configuration**: The `--model` parameter must match a model defined in `MODEL_SPECS`, and the external server must be serving that exact model or a compatible variant.

- **Device Parameter**: `--tt-device` selects the target hardware profile. If omitted, `run.py` infers a default from available host hardware. The legacy alias `--device` is still accepted.

- **No Server Management**: When running client-side scripts, `run.py` will not start, stop, or manage any inference servers. It assumes the external server is already running and accessible.

- **Authentication**: If your external vLLM server requires authentication, ensure the necessary tokens or credentials are configured in your environment.

### Troubleshooting

- **Connection Issues**: Verify the external server is accessible by testing with curl or a similar tool:
  ```bash
  curl http://your-server:port/v1/models
  ```

- **Model Mismatch**: Ensure the model served by the external server matches the model specified in the `--model` parameter.

- **Port Conflicts**: Make sure the `SERVICE_PORT` environment variable matches the actual port your external server is listening on.


## Workflow Setup

The module workflows/run_local.py handles local workflow execution through the WorkflowSetup class, which:

    Bootstraps the Environment:
    Checks the Python version, creates a virtual environment using the uv tool, and installs necessary packages.

    Configures Workflow-Specific Settings:
    Depending on the workflow type (benchmarks, evals, tests), it creates dedicated virtual environments, prepares datasets (e.g., for meta evals), and adjusts configuration files as needed.

    Executes the Workflow Script:
    After setup, it constructs the command line and executes the main workflow script with proper logging and output redirection.

## Project Structure
```
.
├── run.py                   # Main CLI entry point.
├── VERSION                  # Contains the current project version.
├── workflows/
│   ├── run_local.py         # Module for local workflow execution.
│   ├── run_docker.py        # Module for Docker-based execution (under development).
│   ├── model_spec.py               # Model configuration definitions.
│   ├── setup_host.py        # Host setup functions.
│   ├── utils.py             # Utility functions (logging, directory checks, etc.).
│   ├── workflow_config.py   # Workflow configuration details.
│   └── ...                  # Other workflow-related modules.
├── evals/
│   ├── eval_config.py       # Evaluation configuration details.
│   └── run_evals.py         # Evals run script.
```
## Error Handling

    Logging:
    Errors are caught in the main try/except block in run.py and are logged with detailed stack traces.

    Not Yet Implemented:
    Some workflows (e.g., benchmarks, server) currently raise NotImplementedError to indicate that further development is needed.


## Model config

All data known for a given model ahead of runtime is defined compactly and inferred where possible in the ModelSpec object defined in `workflows/model_spec.py`.

For example: `Llama-3.3-70B`
```python
    ModelSpec(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.T3K},
        weights={
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        },
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
    ),
```
Key concepts:

* weights: the ordered list of model weights that a model config is valid for. The same config is copied and made available in MODEL_SPECS map for each of the defined weights strs to match.
* default_impl_map: Maps each device type to a bool indicating whether this implementation is the default for that device. The default implementation will be used if one is not specified directly on CLI.
* device_configurations: the hardware supported for the model implementation and model architecture.

The performance targets for each model-hardware combination are defined in `benchmarking/benchmark_targets/model_performance_reference.json` key used is the default_impl ModelSpec's 1st model weights model name. This model name e.g. `Llama-3.3-70B` above, uniquely defines the targets for all models weights of the same model architecture. These base theoretical targets are the same for all implementations for the same model architecture and hardware combination. Targets can be added directly to a specific ModelSpec as needed for additional points of comparison.

The model evaluation targets are defined only for each model weights because they are dependent on the different outputs from models, not on the model implementation or the hardware running it.
