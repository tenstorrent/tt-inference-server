# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import json
from pathlib import Path
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
    # skip the benchmark sweeps and only run the benchmarks defined in the model config
    BENCHMARK_CONFIGS = {
        model_name: BenchmarkConfig(
            model_name=model_name,
            tasks=[BenchmarkTask(param_map=model_config.perf_reference_map)],
        )
        for model_name, model_config in MODEL_CONFIGS.items()
    }
elif os.getenv("OVERRIDE_BENCHMARKS"):
    """
    override benchmark configs for each model and each device configuration
    uses: benchmarks/model_benchmarks_override.json
    this file uses the same format as the workflows/model_performance_reference.json
    e.g.: 
    {
        "Llama-3.3-70B": {
            "t3k": [
                {
                    "isl": 128,
                    "osl": 128,
                    "max_concurrency": 1,
                    "num_prompts": 8
                }
            ]
        }
    }
    """
    filepath = Path(__file__).resolve().parent / "model_benchmarks_override.json"
    assert filepath.exists(), f"Override benchmark file not found: {filepath}"
    with open(filepath, "r") as f:
        data = json.load(f)

    BENCHMARK_CONFIGS = {
        model_name: BenchmarkConfig(
            model_name=model_name,
            tasks=[
                BenchmarkTask(
                    param_map={
                        DeviceTypes.from_string(device_str): [
                            BenchmarkTaskParams(
                                isl=params.get("isl"),
                                osl=params.get("osl"),
                                max_concurrency=params.get("max_concurrency"),
                                num_prompts=params.get("num_prompts"),
                            )
                            for params in params_list
                        ]
                        for device_str, params_list in override_map.items()
                    }
                )
            ],
        )
        for model_name, override_map in data.items()
    }
else:
    BENCHMARK_CONFIGS = {}
    for model_name, model_config in MODEL_CONFIGS.items():
        perf_ref_task = BenchmarkTask(param_map=model_config.perf_reference_map)
        # get (isl, osl, max_concurrency) from perf_ref_task
        perf_ref_task_runs = {
            _device: [
                (params.isl, params.osl, params.max_concurrency) for params in perf_refs
            ]
            for _device, perf_refs in model_config.perf_reference_map.items()
        }
        # make benchmark sweeps table for each device
        benchmark_task_runs = BenchmarkTask(
            param_map={
                _device: [
                    BenchmarkTaskParams(
                        isl=isl,
                        osl=osl,
                        max_concurrency=1,
                        num_prompts=get_num_prompts(isl, osl, 1),
                    )
                    for isl, osl in BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS
                    if (isl, osl, 1) not in perf_ref_task_runs.get(_device, [])
                ]
                + [
                    BenchmarkTaskParams(
                        isl=isl,
                        osl=osl,
                        max_concurrency=_max_concurrency,
                        num_prompts=get_num_prompts(isl, osl, _max_concurrency),
                    )
                    for isl, osl in MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
                    if (isl, osl, _max_concurrency)
                    not in perf_ref_task_runs.get(_device, [])
                ]
                for _device, _max_concurrency in model_config.max_concurrency_map.items()
            }
        )
        BENCHMARK_CONFIGS[model_name] = BenchmarkConfig(
            model_name=model_name,
            tasks=[perf_ref_task, benchmark_task_runs],
        )
