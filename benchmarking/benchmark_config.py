# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from dataclasses import dataclass
from typing import Dict, List

from workflows.model_spec import MODEL_SPECS, ModelType
from workflows.utils_report import BenchmarkTaskParams, BenchmarkTaskParamsCNN
from workflows.workflow_types import BenchmarkTaskType, DeviceTypes, WorkflowVenvType


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
    workflow_venv_type: WorkflowVenvType = (
        None  # no workflow venv needed for CNN benchmarks
    )


@dataclass(frozen=True)
class BenchmarkTaskEmbedding(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_CNN_API
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.BENCHMARKS_EMBEDDING


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str
    tasks: List[BenchmarkTask]


BENCHMARK_ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (1024, 128),
    (2048, 128),
    (2048, 2048),
    (4096, 128),
    (8192, 128),
    (16384, 128),
    (32768, 128),
    (65536, 128),
]

BENCHMARK_CONCURRENCY = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
]


# Image resolution pairs for multimodal benchmarks
# Format here is isl, osl, image_height, image_width, images_per_prompt
ISL_OSL_IMAGE_RESOLUTION_PAIRS = [
    (128, 128, 512, 512, 1),  # Base resolution
    (128, 128, 1024, 1024, 1),
    (128, 128, 1024, 512, 1),
    (128, 128, 512, 1024, 1),
]


def _expand_text_sweep_params(
    isl: int,
    osl: int,
    max_context: int,
    model_max_concurrency: int,
) -> List[BenchmarkTaskParams]:
    allowed_max_concurrency = get_benchmark_max_concurrency(
        isl, osl, max_context, model_max_concurrency
    )
    return [
        BenchmarkTaskParams(
            isl=isl,
            osl=osl,
            max_concurrency=concurrency,
            num_prompts=get_num_prompts(isl, osl, concurrency),
        )
        for concurrency in BENCHMARK_CONCURRENCY
        if concurrency <= allowed_max_concurrency
    ]


def _expand_image_sweep_params(
    isl: int,
    osl: int,
    image_height: int,
    image_width: int,
    images_per_prompt: int,
    max_context: int,
    model_max_concurrency: int,
    model_name: str,
) -> List[BenchmarkTaskParams]:
    vision_tokens = calculate_vision_tokens(
        image_height=image_height,
        image_width=image_width,
        images_per_prompt=images_per_prompt,
        model_name=model_name,
    )
    allowed_max_concurrency = get_benchmark_max_concurrency(
        isl,
        osl,
        max_context,
        model_max_concurrency,
        vision_tokens=vision_tokens,
    )
    return [
        BenchmarkTaskParams(
            isl=isl,
            osl=osl,
            max_concurrency=concurrency,
            num_prompts=get_num_prompts(isl, osl, concurrency),
            task_type="image",
            image_height=image_height,
            image_width=image_width,
            images_per_prompt=images_per_prompt,
        )
        for concurrency in BENCHMARK_CONCURRENCY
        if concurrency <= allowed_max_concurrency
    ]


def get_num_prompts(input_len, output_len, max_concurrency):
    # Large sequences (slowest) -> fewest prompts
    if output_len > 1024 or input_len > 16384:
        return 1 * max_concurrency
    
    if input_len > 4096:
        return 2 * max_concurrency

    # Medium sequences
    if (output_len > 128 and output_len <= 1024) or (
        input_len > 128 and input_len <= 4096
    ):
        return 4 * max_concurrency

    # Small sequences (fastest) -> most prompts
    if output_len <= 128:
        return 8 * max_concurrency

    raise ValueError(f"Invalid output_len: {output_len}")


def calculate_vision_tokens(
    image_height, image_width, images_per_prompt, model_name=None
):
    """
    Calculate vision tokens from image dimensions on the client side.

    Different VLM models use different methods to calculate vision tokens:
    - Gemma-3 models: Fixed 256 tokens per image (images normalized to 896x896)
    - Qwen2.5-VL: (height // 28) * (width // 28)
    - Qwen3-VL: (height // 32) * (width // 32)

    Args:
        image_height: Height of the image in pixels
        image_width: Width of the image in pixels
        images_per_prompt: Number of images per prompt
        model_name: Model name to determine calculation method (e.g., "gemma-3-27b-it", "Qwen/Qwen2.5-VL-3B-Instruct")

    Returns:
        Total number of vision tokens
    """
    if image_height is None or image_width is None or images_per_prompt is None:
        return 0

    if model_name is None:
        return 0

    model_name_lower = model_name.lower()

    # Gemma-3 models: Fixed 256 tokens per image
    if "gemma-3" in model_name_lower or "medgemma" in model_name_lower:
        tokens_per_image = 256
    # Qwen2.5-VL models
    elif "qwen2.5-vl" in model_name_lower or "qwen2-5-vl" in model_name_lower:
        tokens_per_image = (image_height // 28) * (image_width // 28)
    # Qwen3-VL models
    elif "qwen3-vl" in model_name_lower or "qwen3" in model_name_lower:
        tokens_per_image = (image_height // 32) * (image_width // 32)
    else:
        # Default: return 0 for unknown models
        return 0

    return tokens_per_image * images_per_prompt


def get_benchmark_max_concurrency(
    isl, osl, max_context, model_max_concurrency=32, vision_tokens=0
):
    """
    Calculate the maximum concurrency for benchmarks based on context limits.

    For VLM models, vision tokens must be included in the calculation to ensure
    accurate max_concurrency values that account for the full context usage.

    Args:
        isl: Input sequence length (text tokens)
        osl: Output sequence length
        max_context: Maximum context length supported by the model
        model_max_concurrency: Maximum concurrency supported by the model (default: 32)
        vision_tokens: Number of vision tokens per request (default: 0 for LLM-only)

    Returns:
        Maximum concurrency that fits within the context limit
    """
    # Calculate total sequence length per request (text + vision tokens)
    total_seq_len = isl + osl + vision_tokens

    # If a single request exceeds max_context, return 1 (minimum viable)
    if total_seq_len > max_context:
        return 1

    # Calculate maximum concurrency that fits within context limit
    max_concurrency_by_context = max_context // total_seq_len

    # Return the minimum of context-limited and model-limited concurrency
    return min(max_concurrency_by_context, model_max_concurrency)


def cap_benchmark_params(
    params: BenchmarkTaskParams,
    max_context: int,
    model_max_concurrency: int,
    model_name: str = None,
) -> BenchmarkTaskParams:
    """
    Cap max_concurrency based on context limits (including vision tokens for VLM models)
    and recalculate num_prompts accordingly.

    Args:
        params: Original benchmark task parameters
        max_context: Maximum context length supported by the model
        model_max_concurrency: Maximum concurrency supported by the model
        model_name: Model name for vision token calculation (optional)

    Returns:
        Updated BenchmarkTaskParams with capped concurrency and recalculated num_prompts
    """
    # Skip capping for CNN/Audio tasks that don't have isl/osl
    if params.isl is None or params.osl is None:
        return params

    # Calculate vision tokens for VLM models
    vision_tokens = 0
    if params.task_type == "image" and params.image_height and params.image_width:
        vision_tokens = calculate_vision_tokens(
            params.image_height,
            params.image_width,
            params.images_per_prompt or 0,
            model_name,
        )

    # Calculate the allowed max_concurrency based on sequence length (including vision tokens)
    calculated_max_concurrency = get_benchmark_max_concurrency(
        params.isl, params.osl, max_context, model_max_concurrency, vision_tokens
    )

    # Cap the max_concurrency if it exceeds the calculated limit
    capped_max_concurrency = min(params.max_concurrency, calculated_max_concurrency)

    # If concurrency was capped, recalculate num_prompts
    if capped_max_concurrency < params.max_concurrency:
        recalculated_num_prompts = get_num_prompts(
            params.isl, params.osl, capped_max_concurrency
        )

        # Create new params with capped values
        return BenchmarkTaskParams(
            isl=params.isl,
            osl=params.osl,
            max_concurrency=capped_max_concurrency,
            num_prompts=recalculated_num_prompts,
            task_type=params.task_type,
            image_height=params.image_height,
            image_width=params.image_width,
            images_per_prompt=params.images_per_prompt,
            targets=params.targets,
            theoretical_ttft_ms=params.theoretical_ttft_ms,
            theoretical_tput_user=params.theoretical_tput_user,
            target_peak_perf=params.target_peak_perf,
        )

    # No capping needed, return original params
    return params


# define benchmark configs for each model and each device configuration
# uses:
# 1. BENCHMARK_ISL_OSL_PAIRS
# 2. ISL_OSL_IMAGE_RESOLUTION_PAIRS
# num_prompts is set dynamically based on OSL because that mostly sets how long the benchmark takes
BENCHMARK_CONFIGS = {}
for model_id, model_spec in MODEL_SPECS.items():
    # Since each ModelConfig now represents a single device, use that device and its max_concurrency
    device = model_spec.device_type
    model_max_concurrency = model_spec.device_model_spec.max_concurrency
    max_context = model_spec.device_model_spec.max_context
    perf_reference = model_spec.device_model_spec.perf_reference

    # Apply capping to each perf reference entry (including vision tokens for VLM models)
    capped_perf_reference = [
        cap_benchmark_params(
            params, max_context, model_max_concurrency, model_spec.model_name
        )
        for params in perf_reference
    ]

    # Create performance reference task with capped values
    if model_spec.model_type == ModelType.CNN:
        perf_ref_task = BenchmarkTaskCNN(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.EMBEDDING:
        perf_ref_task = BenchmarkTaskEmbedding(param_map={device: capped_perf_reference})
    else:
        perf_ref_task = BenchmarkTask(param_map={device: capped_perf_reference})

    tasks = [perf_ref_task]
    # optionally skip the benchmark sweeps and only run the perf reference targets
    if not bool(os.getenv("ONLY_BENCHMARK_TARGETS")):
        # Make benchmark sweeps table for this device
        if model_spec.model_type == ModelType.CNN:
            benchmark_task_runs = BenchmarkTaskCNN(
                param_map={
                    device: [
                        BenchmarkTaskParamsCNN(num_inference_steps=20, num_eval_runs=15)
                    ]
                }
            )
        elif model_spec.model_type == ModelType.EMBEDDING:
            benchmark_task_runs = BenchmarkTaskEmbedding(
                param_map={device: [BenchmarkTaskParams()]}
            )
        else:
            benchmark_task_runs = BenchmarkTask(
                param_map={
                    device: [
                        expanded_params
                        for isl, osl in BENCHMARK_ISL_OSL_PAIRS
                        for expanded_params in _expand_text_sweep_params(
                            isl=isl,
                            osl=osl,
                            max_context=max_context,
                            model_max_concurrency=model_max_concurrency,
                        )
                    ]
                    + (
                        # additional vision language model image + text benchmarks
                        [
                            expanded_params
                            for isl, osl, height, width, images_per_prompt in ISL_OSL_IMAGE_RESOLUTION_PAIRS
                            for expanded_params in _expand_image_sweep_params(
                                isl=isl,
                                osl=osl,
                                image_height=height,
                                image_width=width,
                                images_per_prompt=images_per_prompt,
                                max_context=max_context,
                                model_max_concurrency=model_max_concurrency,
                                model_name=model_spec.model_name,
                            )
                        ]
                        if "image" in model_spec.supported_modalities
                        else []
                    )
                }
            )

        tasks.append(benchmark_task_runs)

    BENCHMARK_CONFIGS[model_id] = BenchmarkConfig(model_id=model_id, tasks=tasks)
