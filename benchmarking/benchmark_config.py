# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from functools import lru_cache

from workflows.workflow_types import WorkflowVenvType, BenchmarkTaskType, DeviceTypes
from workflows.model_spec import MODEL_SPECS, ModelType
from workflows.utils import BenchmarkTaskParams, BenchmarkTaskParamsCNN

logger = logging.getLogger(__name__)


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
    (8000, 64),
    (16000, 64),
    (32000, 64),
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


@lru_cache(maxsize=32)
def get_image_token_count_from_processor(
    model_name_or_path: str,
    image_height: int,
    image_width: int,
) -> Optional[int]:
    """
    Determine image token count using the model's HuggingFace processor.
    
    Loads the processor (not model weights) and processes a dummy image to determine
    the actual token count. This works for any vision-language model and handles
    model-specific preprocessing (resizing, normalization, patching, etc.).
    
    Args:
        model_name_or_path: HuggingFace model ID (e.g., "google/gemma-3-4b-it")
        image_height: Height of the image in pixels
        image_width: Width of the image in pixels
    
    Returns:
        Number of image tokens, or None if processor cannot be loaded
        
    Note:
        - Only downloads processor config (~KB), not model weights (~GB)
        - Results are cached automatically via @lru_cache
        - Cached in ~/.cache/huggingface/ after first download
    """
    try:
        from transformers import AutoProcessor
        from PIL import Image
        import numpy as np
        
        logger.info(f"Loading processor for {model_name_or_path}...")
        
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        
        # Create dummy image and minimal text prompt
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
        )
        dummy_text = "Test"
        
        # Process inputs with image
        inputs_with_image = processor(
            text=dummy_text,
            images=dummy_image,
            return_tensors="pt"
        )
        
        # Process text-only inputs for comparison
        inputs_text_only = processor(
            text=dummy_text,
            return_tensors="pt"
        )
        
        # Calculate image tokens: total - text
        image_tokens = inputs_with_image["input_ids"].shape[1] - inputs_text_only["input_ids"].shape[1]
        
        logger.info(
            f"✓ Determined {image_tokens} image tokens for {model_name_or_path} "
            f"at {image_height}×{image_width}"
        )
        
        return image_tokens
        
    except Exception as e:
        logger.error(
            f"Failed to load processor for {model_name_or_path}: {e}"
        )
        return None


def calculate_image_token_count(
    image_height: int,
    image_width: int,
    model_name: str,
    hf_model_repo: str
) -> int:
    """
    Calculate the number of tokens an image will consume in a vision-language model.
    
    Strategy:
    1. Try processor-based method (accurate, requires transformers in venv)
    2. Fall back to hardcoded values based on model name (works without transformers)
    
    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        model_name: Model name for fallback formula selection
        hf_model_repo: HuggingFace model ID for processor-based method
    
    Returns:
        Number of image tokens
        
    Note:
        - During config generation (system Python): Uses hardcoded fallback values
        - During runtime (workflow venv): Can use processor if transformers available
        - Fallback values verified against official model documentation
    """
    # Try processor-based method first (requires transformers)
    token_count = get_image_token_count_from_processor(
        hf_model_repo, image_height, image_width
    )
    if token_count is not None:
        return token_count
    
    # Fallback to hardcoded values (no transformers required)
    model_lower = model_name.lower()
    
    # Gemma-3: 256 tokens (verified via vLLM logs)
    # Ref: https://ai.google.dev/gemma/docs/core/model_card_3
    if "gemma-3" in model_lower or "gemma3" in model_lower:
        return 256
    
    # Qwen2.5-VL: Dynamic based on 14×14 patches with 28-pixel rounding
    # Ref: https://www.emergentmind.com/topics/qwen2-5-vl-model
    if "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        rounded_h = (image_height // 28) * 28
        rounded_w = (image_width // 28) * 28
        return (rounded_h // 14) * (rounded_w // 14)
    
    # Qwen3-VL: Dynamic with 32× compression
    # Ref: https://qwen-ai.chat/models/qwen3-vl/
    if "qwen3-vl" in model_lower or "qwen3_vl" in model_lower:
        return (image_height // 32) * (image_width // 32)
    
    # Default: Use Gemma-3 value as safe fallback
    logger.warning(f"Unknown model {model_name}, defaulting to 256 image tokens")
    return 256


def get_benchmark_max_concurrency(isl, osl, max_context, model_max_concurrency=32, num_image_tokens=0):
    """
    Calculate the maximum concurrency for benchmarks based on context limits.
    
    Args:
        isl: Input sequence length (text tokens)
        osl: Output sequence length (text tokens)
        max_context: Maximum context length supported by the model
        model_max_concurrency: Maximum concurrency supported by the model (default: 32)
        num_image_tokens: Number of image tokens per request (default: 0 for text-only)
    
    Returns:
        Maximum concurrency that fits within the context limit
    """
    # Calculate total sequence length per request including image tokens
    total_seq_len = isl + osl + num_image_tokens
    
    # If a single request exceeds max_context, return 1 (minimum viable)
    if total_seq_len > max_context:
        return 1
    
    # Calculate maximum concurrency that fits within context limit
    max_concurrency_by_context = max_context // total_seq_len
    
    # Return the minimum of context-limited and model-limited concurrency
    return min(max_concurrency_by_context, model_max_concurrency)


def _build_benchmark_configs():
    """
    Build benchmark configurations for all models.
    
    This function is called lazily to avoid importing transformers at module load time.
    The transformers library is only available after the venv is set up.
    
    Returns:
        Dict of benchmark configurations keyed by model_id
    """
    # define benchmark configs for each model and each device configuration
    # uses:
    # 1. BATCH_1_BENCHMARK_COMMON_ISL_OSL_PAIRS
    # 2. MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
    # 3. ISL_OSL_IMAGE_RESOLUTION_PAIRS
    # num_prompts is set dynamically based on OSL because that mostly sets how long the benchmark takes
    if os.getenv("ONLY_BENCHMARK_TARGETS"):
        # skip the benchmark sweeps and only run the benchmarks defined in the model config
        return {
            model_id: BenchmarkConfig(
                model_id=model_id,
                tasks=[BenchmarkTask(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})],
            )
            for model_id, model_spec in MODEL_SPECS.items()
        }
    else:
        configs = {}
        for model_id, model_spec in MODEL_SPECS.items():
            # Create performance reference task using the device_model_spec
            perf_ref_task = BenchmarkTask(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})
            if model_spec.model_type == ModelType.CNN:
                perf_ref_task = BenchmarkTaskCNN(param_map={model_spec.device_type: model_spec.device_model_spec.perf_reference})
            
            # get (isl, osl, max_concurrency) from perf_ref_task
            perf_ref_task_runs = {
                model_spec.device_type: [
                    (params.isl, params.osl, params.image_height, params.image_width, params.images_per_prompt, params.max_concurrency) if params.task_type == "image"
                    else (params.num_inference_steps,) if params.task_type == "cnn"
                    else (params.isl, params.osl, params.max_concurrency) 
                    for params in model_spec.device_model_spec.perf_reference
                ]
            }
            
            # Since each ModelConfig now represents a single device, use that device and its max_concurrency
            _device = model_spec.device_type
            _model_max_concurrency = model_spec.device_model_spec.max_concurrency
            _max_context = model_spec.device_model_spec.max_context
            
            # make benchmark sweeps table for this device
            if model_spec.model_type == ModelType.CNN:
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
                                max_concurrency=get_benchmark_max_concurrency(isl, osl, _max_context, _model_max_concurrency),
                                num_prompts=get_num_prompts(isl, osl, get_benchmark_max_concurrency(isl, osl, _max_context, _model_max_concurrency)),
                            )
                            for isl, osl in MAX_CONCURRENCY_BENCHMARK_COMMON_ISL_OSL_PAIRS
                            if (isl, osl, get_benchmark_max_concurrency(isl, osl, _max_context, _model_max_concurrency))
                            not in perf_ref_task_runs.get(_device, [])
                        ]
                        + 
                        (
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
                                    max_concurrency=get_benchmark_max_concurrency(
                                        isl, osl, _max_context, _model_max_concurrency,
                                        num_image_tokens=calculate_image_token_count(height, width, model_spec.model_name, model_spec.hf_model_repo) * images_per_prompt
                                    ),
                                    num_prompts=get_num_prompts(
                                        isl, osl, 
                                        get_benchmark_max_concurrency(
                                            isl, osl, _max_context, _model_max_concurrency,
                                            num_image_tokens=calculate_image_token_count(height, width, model_spec.model_name, model_spec.hf_model_repo) * images_per_prompt
                                        )
                                    ),
                                    task_type="image",
                                    image_height=height,
                                    image_width=width,
                                    images_per_prompt=images_per_prompt,
                                )
                                for isl, osl, height, width, images_per_prompt in ISL_OSL_IMAGE_RESOLUTION_PAIRS
                                if (isl, osl, height, width, images_per_prompt, get_benchmark_max_concurrency(
                                    isl, osl, _max_context, _model_max_concurrency,
                                    num_image_tokens=calculate_image_token_count(height, width, model_spec.model_name, model_spec.hf_model_repo) * images_per_prompt
                                )) not in perf_ref_task_runs.get(_device, [])
                            ] if "image" in model_spec.supported_modalities else []
                        )
                    }
                )

            configs[model_id] = BenchmarkConfig(
                model_id=model_id,
                tasks=[perf_ref_task, benchmark_task_runs],
            )
        return configs


# Lazy-loaded benchmark configs cache
_BENCHMARK_CONFIGS_CACHE = None


def __getattr__(name):
    """
    Module-level __getattr__ for lazy loading BENCHMARK_CONFIGS.
    
    This ensures the configs are only built when first accessed, allowing
    fallback values to be used during initial import (before venv setup).
    """
    global _BENCHMARK_CONFIGS_CACHE
    
    if name == "BENCHMARK_CONFIGS":
        if _BENCHMARK_CONFIGS_CACHE is None:
            _BENCHMARK_CONFIGS_CACHE = _build_benchmark_configs()
        return _BENCHMARK_CONFIGS_CACHE
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
