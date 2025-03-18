# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List, Tuple

from workflows.workflow_types import WorkflowVenvType, BenchmarkTaskType
from workflows.utils import map_configs_by_attr


@dataclass(frozen=True)
class BenchmarkTask:
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_VLLM_API
    isl_osl_pairs: List[Tuple[int, int]] = None
    workflow_venv_type: WorkflowVenvType = (
        WorkflowVenvType.BENCHMARKS_HTTP_CLIENT_VLLM_API
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    hf_model_repo: str
    tasks: List[BenchmarkTask]


_benchmark_config_list = [
    BenchmarkConfig(
        hf_model_repo="Qwen/Qwen2.5-7B-Instruct",
        tasks=[
            BenchmarkTask(isl_osl_pairs=[(128, 128)]),
        ],
    ),
    BenchmarkConfig(
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        tasks=[
            BenchmarkTask(isl_osl_pairs=[(128, 128)]),
        ],
    ),
    BenchmarkConfig(
        hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
        tasks=[
            BenchmarkTask(isl_osl_pairs=[(128, 128)]),
        ],
    ),
]

BENCHMARK_CONFIGS = map_configs_by_attr(
    config_list=_benchmark_config_list, attr="hf_model_repo"
)
