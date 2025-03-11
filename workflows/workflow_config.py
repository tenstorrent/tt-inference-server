# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from pathlib import Path
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
    TESTS = auto()
    REPORTS = auto()
    SERVER = auto()

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


def get_default_workflow_root_log_dir():
    # docker env uses CACHE_ROOT
    default_dir_name = "workflow_logs"
    cache_root = os.getenv("CACHE_ROOT")
    if cache_root:
        default_workflow_root_log_dir = Path(cache_root) / default_dir_name
    else:
        default_workflow_root_log_dir = get_repo_root_path() / default_dir_name
    return default_workflow_root_log_dir


@dataclass(frozen=True)
class WorkflowConfig:
    """
    All static configuration and metadata required to execute a workflow.
    """

    workflow_type: WorkflowType
    run_script_path: Path
    name: Optional[str] = None
    workflow_path: Optional[Path] = None
    python_version: str = "3.10"
    venv_path: Optional[Path] = None
    venv_python: Optional[Path] = None
    venv_pip: Optional[Path] = None
    workflow_log_dir: Optional[Path] = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        if not self.run_script_path:
            raise ValueError("run_script_path must be provided for WorkflowConfig.")
        if not self.workflow_type:
            raise ValueError("workflow_type must be provided for WorkflowConfig.")
        assert isinstance(self.run_script_path, Path)

    def _infer_data(self):
        if self.name is None:
            object.__setattr__(self, "name", self.workflow_type.name.lower())

        if self.workflow_path is None:
            object.__setattr__(self, "workflow_path", self.run_script_path.parent)

        if self.venv_path is None:
            object.__setattr__(
                self, "venv_path", self.workflow_path / f".venv_{self.name}"
            )

        if self.venv_python is None:
            object.__setattr__(self, "venv_python", self.venv_path / "bin" / "python")

        if self.venv_pip is None:
            object.__setattr__(self, "venv_pip", self.venv_path / "bin" / "pip")

        if self.workflow_log_dir is None:
            object.__setattr__(
                self, "workflow_log_dir", get_default_workflow_root_log_dir()
            )


WORKFLOW_BENCHMARKS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.BENCHMARKS,
    run_script_path=get_repo_root_path() / "benchmarking" / "run_benchmarks.py",
)
WORKFLOW_EVALS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.EVALS,
    run_script_path=get_repo_root_path() / "evals" / "run_evals.py",
)
WORKFLOW_SERVER_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.SERVER,
    run_script_path=get_repo_root_path()
    / "vllm-tt-metal-llama3"
    / "src"
    / "run_vllm_api_server.py",
)

# Define WorkflowConfig instances in a list
workflow_config_list = [
    WORKFLOW_BENCHMARKS_CONFIG,
    WORKFLOW_EVALS_CONFIG,
    WORKFLOW_SERVER_CONFIG,
]

# Generate a dictionary keyed by the workflow name for each WorkflowConfig instance
WORKFLOW_CONFIGS = {config.workflow_type: config for config in workflow_config_list}
