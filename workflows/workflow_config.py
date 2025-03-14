# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict


from workflows.utils import get_repo_root_path, get_default_workflow_root_log_dir


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
    TESTS = auto()
    REPORTS = auto()
    SERVER = auto()
    TESTS = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid TaskType: {name}")


class WorkflowVenvType(IntEnum):
    EVALS = auto()
    EVALS_META = auto()
    EVALS_VISION = auto()
    BENCHMARKS = auto()
    SERVER = auto()
    TESTS = auto()


@dataclass(frozen=True)
class VenvConfig:
    venv_type: WorkflowVenvType
    name: Optional[str] = None
    python_version: Optional[str] = "3.10"
    venv_path: Optional[Path] = None
    venv_python: Optional[Path] = None
    venv_pip: Optional[Path] = None
    workflow_path: Optional[Path] = None

    def _infer_data(self):
        if self.venv_python is None:
            object.__setattr__(self, "venv_python", self.venv_path / "bin" / "python")

        if self.venv_python is None:
            object.__setattr__(self, "venv_python", self.venv_path / "bin" / "python")

        if self.venv_pip is None:
            object.__setattr__(self, "venv_pip", self.venv_path / "bin" / "pip")


@dataclass(frozen=True)
class WorkflowConfig:
    """
    All static configuration and metadata required to execute a workflow.
    """

    workflow_type: WorkflowType
    run_script_path: Path
    workflow_venvs_list: List[VenvConfig]
    name: Optional[str] = None
    workflow_path: Optional[Path] = None
    workflow_venv_dict: Dict[str, VenvConfig] = None
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

        if self.workflow_log_dir is None:
            object.__setattr__(
                self, "workflow_log_dir", get_default_workflow_root_log_dir()
            )

        for wf_venv in self.workflow_venvs_list:
            object.__setattr__(wf_venv, "name", wf_venv.venv_type.name.lower())

            object.__setattr__(
                wf_venv, "venv_path", self.workflow_path / f".venv_{wf_venv.name}"
            )
            wf_venv._infer_data()
            object.__setattr__(
                self,
                "workflow_venv_dict",
                {wf_venv.venv_type: wf_venv for wf_venv in self.workflow_venvs_list},
            )


WORKFLOW_BENCHMARKS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.BENCHMARKS,
    run_script_path=get_repo_root_path() / "benchmarking" / "run_benchmarks.py",
    workflow_venvs_list=[VenvConfig(venv_type=WorkflowVenvType.BENCHMARKS)],
)
WORKFLOW_EVALS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.EVALS,
    run_script_path=get_repo_root_path() / "evals" / "run_evals.py",
    workflow_venvs_list=[
        VenvConfig(venv_type=WorkflowVenvType.EVALS),
        VenvConfig(venv_type=WorkflowVenvType.EVALS_META),
        VenvConfig(venv_type=WorkflowVenvType.EVALS_VISION),
    ],
)
WORKFLOW_SERVER_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.SERVER,
    run_script_path=get_repo_root_path()
    / "vllm-tt-metal-llama3"
    / "src"
    / "run_vllm_api_server.py",
    workflow_venvs_list=[VenvConfig(venv_type=WorkflowVenvType.SERVER)],
)

WORKFLOW_TESTS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.TESTS,
    run_script_path=get_repo_root_path() / "tests" / "run_tests.py",
    workflow_venvs_list=[VenvConfig(venv_type=WorkflowVenvType.TESTS)],
)

# Define WorkflowConfig instances in a list
workflow_config_list = [
    WORKFLOW_BENCHMARKS_CONFIG,
    WORKFLOW_EVALS_CONFIG,
    WORKFLOW_SERVER_CONFIG,
    WORKFLOW_TESTS_CONFIG,
]

# Generate a dictionary keyed by the workflow name for each WorkflowConfig instance
WORKFLOW_CONFIGS = {config.workflow_type: config for config in workflow_config_list}
