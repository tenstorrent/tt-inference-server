# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from workflows.workflow_types import WorkflowVenvType
from workflows.utils import map_configs_by_attr
from workflows.model_spec import MODEL_SPECS


@dataclass(frozen=True)
class TestTask:
    task_name: str
    test_path: Path
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.TESTS_RUN_SCRIPT
    test_args: Tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        pass

    def validate_data(self):
        pass


@dataclass(frozen=True)
class TestConfig:
    hf_model_repo: str
    tasks: List[TestTask]


_test_config_list = [
    TestConfig(
        hf_model_repo="Qwen/Qwen3-32B",
        tasks=[
            TestTask(
                task_name="vllm_params",
                test_path=Path(
                    "tests/server_tests/test_cases/test_vllm_server_parameters.py"
                ),
                test_args=("s", "v"),
            ),
        ],
    ),
    TestConfig(
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        tasks=[
            TestTask(
                task_name="vllm_params",
                test_path=Path(
                    "tests/server_tests/test_cases/test_vllm_server_parameters.py"
                ),
                test_args=("s", "v"),
            ),
        ],
    ),
    TestConfig(
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        tasks=[
            TestTask(
                task_name="vllm_params",
                test_path=Path(
                    "tests/server_tests/test_cases/test_vllm_server_parameters.py"
                ),
                test_args=("s", "v"),
            ),
        ],
    ),
]


_test_config_map = map_configs_by_attr(
    config_list=_test_config_list, attr="hf_model_repo"
)
TEST_CONFIGS = {
    model_spec.model_name: _test_config_map[model_spec.hf_model_repo]
    for _, model_spec in MODEL_SPECS.items()
    if model_spec.hf_model_repo in _test_config_map
}
