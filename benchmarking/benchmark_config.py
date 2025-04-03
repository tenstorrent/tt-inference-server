# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from dataclasses import dataclass
from typing import List, Dict

from workflows.workflow_types import WorkflowVenvType, BenchmarkTaskType, DeviceTypes
from workflows.model_config import MODEL_CONFIGS
from workflows.utils import BenchmarkTaskParams


@dataclass(frozen=True)
class BenchmarkTask:
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_VLLM_API
    workflow_venv_type: WorkflowVenvType = (
        WorkflowVenvType.BENCHMARKS_HTTP_CLIENT_VLLM_API
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    model_name: str
    tasks: List[BenchmarkTask]


BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (1024, 128),
    (2048, 128),
    (3072, 128),
    (4096, 128),
    (8192, 128),
    (16384, 128),
    (32000, 128),
]

MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (2048, 128),
    (2048, 2048),
    (3000, 64),
    (4000, 64),
    (4500, 64),
    (8000, 64),
    (16000, 64),
]

# HF_MODELS = {v.hf_model_repo for k, v in MODEL_CONFIGS.items()}


def get_num_prompts(input_len, output_len, max_concurrency):
    if output_len > 1024:
        return 2 * max_concurrency
    if output_len > 128 and output_len <= 1024:
        return 4 * max_concurrency
    if output_len <= 128:
        return 8 * max_concurrency
    raise ValueError(f"Invalid output_len: {output_len}")


# define benchmark configs for each model and each device configuration
# uses:
# 1. BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS
# 2. MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
# num_prompts is set dynamically based on OSL because that mostly sets how long the benchmark takes
if os.getenv("ONLY_TARGET_BENCHMARKS"):
    BENCHMARK_CONFIGS = {
        model_name: BenchmarkConfig(
            model_name=model_name,
            tasks=[
                BenchmarkTask(
                    param_map={
                        _device: [
                            BenchmarkTaskParams(
                                isl=isl,
                                osl=osl,
                                max_concurrency=1,
                                num_prompts=get_num_prompts(isl, osl, 1),
                            )
                            for isl, osl in BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS
                        ]
                        + [
                            BenchmarkTaskParams(
                                isl=isl,
                                osl=osl,
                                max_concurrency=_max_concurrency,
                                num_prompts=get_num_prompts(isl, osl, _max_concurrency),
                            )
                            for isl, osl in MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
                        ]
                        for _device, _max_concurrency in model_config.max_concurrency_map.items()
                    }
                )
            ],
        )
        for model_name, model_config in MODEL_CONFIGS.items()
    }
else:
    BENCHMARK_CONFIGS = {
        model_name: BenchmarkConfig(
            model_name=model_name,
            tasks=[
                BenchmarkTask(
                    param_map={
                        _device: [
                            BenchmarkTaskParams(
                                isl=isl,
                                osl=osl,
                                max_concurrency=1,
                                num_prompts=get_num_prompts(isl, osl, 1),
                            )
                            for isl, osl in BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS
                        ]
                        + [
                            BenchmarkTaskParams(
                                isl=isl,
                                osl=osl,
                                max_concurrency=_max_concurrency,
                                num_prompts=get_num_prompts(isl, osl, _max_concurrency),
                            )
                            for isl, osl in MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
                        ]
                        for _device, _max_concurrency in model_config.max_concurrency_map.items()
                    }
                )
            ],
        )
        for model_name, model_config in MODEL_CONFIGS.items()
    }
