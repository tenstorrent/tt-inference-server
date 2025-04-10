# tt-inference-server workflow runner

This project provides a command-line interface (CLI) to run various workflows related to the Tenstorrent inference server. It supports executing workflows locally or via Docker, handling environment setup, dependency management, and logging for multiple models and workflow types.
Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [run.py CLI Usage](#runpy-cli-usage)
- [Workflow Setup](#workflow-setup)
- [Project Structure](#project-structure)
- [Error Handling](#error-handling)

# Overview

The main entry point of the project is `run.py`. This script enables you to execute different workflows—such as benchmarks, evals, server, release, or report—by specifying the model and workflow type. Depending on your configuration, workflows can run on your host system or inside a Docker container.

The module workflows/run_local.py is responsible for setting up the local execution environment. It handles tasks such as bootstrapping a virtual environment, installing dependencies, configuring workflow-specific settings, and finally launching the workflow script.
Features

    Multiple Workflows: Run benchmarks, evals, server, release, and report workflows.
    Execution Modes: Choose between running workflows locally or in Docker mode.
    Automatic Setup: Manages environment setup, including virtual environments and dependency installation.
    Logging: Detailed logging for tracking execution, errors, and debugging.

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
python run.py --model <model_name> --workflow <workflow_type>
```
Dependencies:

Required dependencies are installed during the workflow setup process. Ensure you have internet connectivity for downloading packages and cloning any necessary repositories.

## run.py CLI Usage

Execute the CLI using run.py with the appropriate command-line arguments.
```
Command-line Arguments

    --model (required):
    Specifies the model to run. The available models are defined in MODEL_CONFIGS.

    --workflow (required):
    Specifies the workflow to run. Valid options include:
        benchmarks
        evals
        server
        release
        report

    --docker (optional):
    Enable Docker mode to run the workflow inside a Docker container.

    --device (optional):
    Specifies the device to use. Choices include:
        N150
        N300
        T3K

    --workflow-args (optional):
    Additional workflow arguments (e.g., param1=value1 param2=value2).

    --jwt-secret (optional):
    JWT secret for generating tokens. Defaults to the JWT_SECRET environment variable if not provided.

    --hf-token (optional):
    Hugging Face token. Defaults to the HF_TOKEN environment variable if not provided.

Example Commands

Run the evals workflow locally:

    python3 run.py --model my_model --workflow evals --hf-token your_hf_token_here

    e.g.:

    python3 run.py --model Qwen2.5-72B-Instruct --workflow evals

Run a workflow in Docker mode:

    python3 run.py --model Llama-3.3-70B-Instruct --workflow evals --docker
```
Note: Docker mode is not yet fully implemented and will currently raise a NotImplementedError.


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
│   ├── model_config.py      # Model configuration definitions.
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