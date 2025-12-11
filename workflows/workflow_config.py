# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from workflows.utils import (
    get_default_workflow_root_log_dir,
    get_repo_root_path,
    map_configs_by_attr,
)
from workflows.workflow_types import WorkflowType, WorkflowVenvType


@dataclass(frozen=True)
class WorkflowConfig:
    """
    All static configuration and metadata required to execute a workflow.
    """

    workflow_type: WorkflowType
    workflow_run_script_venv_type: WorkflowVenvType
    run_script_path: Path
    name: Optional[str] = None
    workflow_log_dir: Optional[Path] = None
    # TODO: remove workflow_path if unused
    workflow_path: Optional[Path] = None

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
                self,
                "workflow_log_dir",
                get_default_workflow_root_log_dir() / f"{self.name}_logs",
            )


WORKFLOW_BENCHMARKS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.BENCHMARKS,
    run_script_path=get_repo_root_path() / "benchmarking" / "run_benchmarks.py",
    workflow_run_script_venv_type=WorkflowVenvType.BENCHMARKS_RUN_SCRIPT,
)

WORKFLOW_BENCHMARKS_AIPERF_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.BENCHMARKS,
    run_script_path=get_repo_root_path() / "benchmarking" / "run_benchmarks_aiperf.py",
    workflow_run_script_venv_type=WorkflowVenvType.BENCHMARKS_AIPERF,
    name="benchmarks_aiperf",
)
WORKFLOW_EVALS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.EVALS,
    run_script_path=get_repo_root_path() / "evals" / "run_evals.py",
    workflow_run_script_venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
)
WORKFLOW_STRESS_TESTS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.STRESS_TESTS,
    run_script_path=get_repo_root_path() / "stress_tests" / "run_stress_tests.py",
    workflow_run_script_venv_type=WorkflowVenvType.STRESS_TESTS_RUN_SCRIPT,
)
WORKFLOW_TESTS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.TESTS,
    run_script_path=get_repo_root_path() / "tests" / "run_tests.py",
    workflow_run_script_venv_type=WorkflowVenvType.TESTS_RUN_SCRIPT,
)
WORKFLOW_SPEC_TESTS_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.SPEC_TESTS,
    run_script_path=get_repo_root_path() / "tests" / "server_tests" / "run.py",
    workflow_run_script_venv_type=WorkflowVenvType.TESTS_RUN_SCRIPT,
)
WORKFLOW_SERVER_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.SERVER,
    run_script_path=get_repo_root_path()
    / "vllm-tt-metal-llama3"
    / "src"
    / "run_vllm_api_server.py",
    workflow_run_script_venv_type=None,
)

WORKFLOW_REPORT_CONFIG = WorkflowConfig(
    workflow_type=WorkflowType.REPORTS,
    run_script_path=get_repo_root_path() / "workflows" / "run_reports.py",
    workflow_run_script_venv_type=WorkflowVenvType.REPORTS_RUN_SCRIPT,
)

# Define WorkflowConfig instances in a list
workflow_config_list = [
    WORKFLOW_BENCHMARKS_CONFIG,
    WORKFLOW_EVALS_CONFIG,
    WORKFLOW_STRESS_TESTS_CONFIG,
    WORKFLOW_TESTS_CONFIG,
    WORKFLOW_SERVER_CONFIG,
    WORKFLOW_SPEC_TESTS_CONFIG,
    WORKFLOW_REPORT_CONFIG,
]

# Generate a dictionary keyed by the workflow name for each WorkflowConfig instance
WORKFLOW_CONFIGS = map_configs_by_attr(
    config_list=workflow_config_list, attr="workflow_type"
)
