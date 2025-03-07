# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from pathlib import Path
from enum import IntEnum, auto


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
    TESTS = auto()
    REPORTS = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid TaskType: {name}")


def get_repo_root_path(marker: str = ".git") -> Path:
    """Return the root directory of the repository by searching for a marker file or directory."""
    current_path = Path(__file__).resolve().parent  # Start from the script's directory
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Repository root not found. No '{marker}' found in parent directories."
    )


model_config = {
    "DeepSeek-R1-Distill-Llama-70B": {
        "HF_MODEL_REPO_ID": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "DeepSeek-R1-Distill-Llama-70B",
        "REPACKED": 0,
    },
    "Qwen2.5-72B": {
        "HF_MODEL_REPO_ID": "Qwen/Qwen2.5-72B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 360,
        "MIN_RAM": 360,
        "MODEL_NAME": "Qwen2.5-72B",
        "REPACKED": 0,
    },
    "Qwen2.5-72B-Instruct": {
        "HF_MODEL_REPO_ID": "Qwen/Qwen2.5-72B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 360,
        "MIN_RAM": 360,
        "MODEL_NAME": "Qwen2.5-72B-Instruct",
        "REPACKED": 0,
    },
    "Qwen2.5-7B": {
        "HF_MODEL_REPO_ID": "Qwen/Qwen2.5-7B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 28,
        "MIN_RAM": 35,
        "MODEL_NAME": "Qwen2.5-7B",
        "REPACKED": 0,
    },
    "Qwen2.5-7B-Instruct": {
        "HF_MODEL_REPO_ID": "Qwen/Qwen2.5-7B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 28,
        "MIN_RAM": 35,
        "MODEL_NAME": "Qwen2.5-7B-Instruct",
        "REPACKED": 0,
    },
    "Llama-3.3-70B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.3-70B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "Llama-3.3-70B",
        "REPACKED": 1,
    },
    "Llama-3.3-70B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.3-70B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "Llama-3.3-70B-Instruct",
        "REPACKED": 1,
    },
    "Llama-3.2-11B-Vision": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.2-11B-Vision",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 44,
        "MIN_RAM": 55,
        "MODEL_NAME": "Llama-3.2-11B-Vision",
        "REPACKED": 0,
    },
    "Llama-3.2-11B-Vision-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 44,
        "MIN_RAM": 55,
        "MODEL_NAME": "Llama-3.2-11B-Vision-Instruct",
        "REPACKED": 0,
    },
    "Llama-3.2-1B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.2-1B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 4,
        "MIN_RAM": 5,
        "MODEL_NAME": "Llama-3.2-1B",
        "REPACKED": 0,
    },
    "Llama-3.2-1B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.2-1B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 4,
        "MIN_RAM": 5,
        "MODEL_NAME": "Llama-3.2-1B-Instruct",
        "REPACKED": 0,
    },
    "Llama-3.2-3B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.2-3B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 12,
        "MIN_RAM": 15,
        "MODEL_NAME": "Llama-3.2-3B",
        "REPACKED": 0,
    },
    "Llama-3.2-3B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.2-3B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 12,
        "MIN_RAM": 15,
        "MODEL_NAME": "Llama-3.2-3B-Instruct",
        "REPACKED": 0,
    },
    "Llama-3.1-70B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.1-70B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "Llama-3.1-70B",
        "REPACKED": 1,
    },
    "Llama-3.1-70B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.1-70B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "Llama-3.1-70B-Instruct",
        "REPACKED": 1,
    },
    "Llama-3.1-8B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.1-8B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 32,
        "MIN_RAM": 40,
        "MODEL_NAME": "Llama-3.1-8B",
        "REPACKED": 0,
    },
    "Llama-3.1-8B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3.1-8B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 32,
        "MIN_RAM": 40,
        "MODEL_NAME": "Llama-3.1-8B-Instruct",
        "REPACKED": 0,
    },
    "Llama-3-70B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3-70B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "Llama-3-70B",
        "REPACKED": 1,
    },
    "Llama-3-70B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3-70B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 350,
        "MIN_RAM": 350,
        "MODEL_NAME": "Llama-3-70B-Instruct",
        "REPACKED": 1,
    },
    "Llama-3-8B": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3-8B",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 32,
        "MIN_RAM": 40,
        "MODEL_NAME": "Llama-3-8B",
        "REPACKED": 0,
    },
    "Llama-3-8B-Instruct": {
        "HF_MODEL_REPO_ID": "meta-llama/Llama-3-8B-Instruct",
        "IMPL_ID": "tt-metal",
        "MIN_DISK": 32,
        "MIN_RAM": 40,
        "MODEL_NAME": "Llama-3-8B-Instruct",
        "REPACKED": 0,
    },
}


workflow_config_map = {
    WorkflowType.BENCHMARKS: {
        "name": "benchmarks",
        "run_script_path": get_repo_root_path() / "benchmarking" / "run_benchmarks.py",
        "python_version": "3.10",
    },
    WorkflowType.EVALS: {
        "name": "evals",
        "run_script_path": get_repo_root_path() / "evals" / "run_evals.py",
        "python_version": "3.10",
    },
}


def get_default_workflow_root_log_dir():
    # docker env uses CACHE_ROOT
    default_dir_name = "workflow_logs"
    cache_root = os.getenv("CACHE_ROOT")
    if cache_root:
        default_workflow_root_log_dir = Path(cache_root) / default_dir_name
    else:
        default_workflow_root_log_dir = get_repo_root_path() / default_dir_name
    return default_workflow_root_log_dir
