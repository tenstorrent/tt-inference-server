# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from dataclasses import dataclass
from typing import List, Dict

from workflows.workflow_types import WorkflowVenvType, BenchmarkTaskType, DeviceTypes
from workflows.model_spec import MODEL_SPECS
from workflows.utils import BenchmarkTaskParams, BenchmarkTaskParamsCNN


@dataclass(frozen=True)
class BenchmarkTask:
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_VLLM_API
    workflow_venv_type: WorkflowVenvType = (
        WorkflowVenvType.BENCHMARKS_HTTP_CLIENT_VLLM_API
    )

@dataclass(frozen=True)
class BenchmarkTaskCNN(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_CNN_API
    workflow_venv_type: WorkflowVenvType = None  # no workflow venv needed for CNN benchmarks

@dataclass(frozen=True)
class BenchmarkTaskAudio(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_AUDIO_API
    workflow_venv_type: WorkflowVenvType = None  # no workflow venv needed for audio benchmarks

@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str
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

# Image resolution pairs for multimodal benchmarks
# Format here is isl, osl, image_height, image_width, images_per_prompt
ISL_OSL_IMAGE_RESOLUTION_PAIRS = [
    (128, 128, 512, 512, 1),   # Base resolution
    (128, 128, 1024, 1024, 1),
    (128, 128, 1024, 512, 1),
    (128, 128, 512, 1024, 1),
]


def get_num_prompts(input_len, output_len, max_concurrency):
    # Large sequences (slowest) -> fewest prompts
    if output_len > 1024 or input_len > 4000:
        return 2 * max_concurrency

    # Medium sequences
    if (output_len > 128 and output_len <= 1024) or (
        input_len > 128 and input_len <= 4000
    ):
        return 4 * max_concurrency

    # Small sequences (fastest) -> most prompts
    if output_len <= 128:
        return 8 * max_concurrency

    raise ValueError(f"Invalid output_len: {output_len}")


# define benchmark configs for each model and each device configuration
# uses:
# 1. BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS
# 2. MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
# 3. ISL_OSL_IMAGE_RESOLUTION_PAIRS
# num_prompts is set dynamically based on OSL because that mostly sets how long the benchmark takes
if os.getenv("ONLY_BENCHMARK_TARGETS"):
    # skip the benchmark sweeps and only run the benchmarks defined in the model config
    BENCHMARK_CONFIGS = {
        model_id: BenchmarkConfig(
            model_id=model_id,
            tasks=[BenchmarkTask(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})],
        )
        for model_id, model_spec in MODEL_SPECS.items()
    }
else:
    BENCHMARK_CONFIGS = {}
    for model_id, model_spec in MODEL_SPECS.items():
        # Create performance reference task using the device_model_spec
        perf_ref_task = BenchmarkTask(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})
        if (model_spec.model_type.name == "CNN"):
            perf_ref_task = BenchmarkTaskCNN(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})
        elif (model_spec.model_type.name == "ASR"):
            perf_ref_task = BenchmarkTaskAudio(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})
        
        # get (isl, osl, max_concurrency) from perf_ref_task
        perf_ref_task_runs = {
            model_spec.device_type: [
                (params.isl, params.osl, params.image_height, params.image_width, params.images_per_prompt, params.max_concurrency) if params.task_type == "image"
                else (params.num_inference_steps,) if params.task_type == "cnn"
                else (params.max_concurrency,) if params.task_type == "audio"
                else (params.isl, params.osl, params.max_concurrency) 
                for params in model_spec.device_model_spec.perf_reference
            ]
        }
        
        # Since each ModelConfig now represents a single device, use that device and its max_concurrency
        _device = model_spec.device_type
        _max_concurrency = model_spec.device_model_spec.max_concurrency
        
        # make benchmark sweeps table for this device
        if (model_spec.model_type.name == "CNN"):
            benchmark_task_runs = BenchmarkTaskCNN(
                param_map={
                    _device: [
                        BenchmarkTaskParamsCNN(
                            num_inference_steps=20,
                            num_eval_runs=15
                        )
                    ]
                }
            )
        elif (model_spec.model_type.name == "ASR"):
            benchmark_task_runs = BenchmarkTaskAudio(
                param_map={
                    _device: [
                        BenchmarkTaskParams(
                            max_concurrency=1,
                            num_prompts=15,
                            task_type="audio"
                        ),
                        BenchmarkTaskParams(
                            max_concurrency=32,
                            num_prompts=15,
                            task_type="audio"
                        ),
                    ]
                }
            )
        else:
            benchmark_task_runs = BenchmarkTask(
                param_map={
                    _device: 
                    [
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
                    + (
                        [
                            BenchmarkTaskParams(
                                isl=isl,
                                osl=osl,
                                max_concurrency=1,
                                num_prompts=get_num_prompts(isl, osl, 1),
                                task_type="image",
                                image_height=height,
                                image_width=width,
                                images_per_prompt=images_per_prompt,
                            )
                            for isl, osl, height, width, images_per_prompt in ISL_OSL_IMAGE_RESOLUTION_PAIRS
                            if (isl, osl, height, width, images_per_prompt, 1) not in perf_ref_task_runs.get(_device, [])
                        ] if "image" in model_spec.supported_modalities else []
                    )
                    + (
                        [
                            BenchmarkTaskParams(
                                isl=isl,
                                osl=osl,
                                max_concurrency=_max_concurrency,
                                num_prompts=get_num_prompts(isl, osl, _max_concurrency),
                                task_type="image",
                                image_height=height,
                                image_width=width,
                                images_per_prompt=images_per_prompt,
                            )
                            for isl, osl, height, width, images_per_prompt in ISL_OSL_IMAGE_RESOLUTION_PAIRS
                            if (isl, osl, height, width, images_per_prompt, _max_concurrency) not in perf_ref_task_runs.get(_device, [])
                        ] if "image" in model_spec.supported_modalities else []
                    )
                }
            )

        BENCHMARK_CONFIGS[model_id] = BenchmarkConfig(
            model_id=model_id,
            tasks=[perf_ref_task, benchmark_task_runs],
        )
