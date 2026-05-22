# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

from benchmarking.spec_decode_common import SpecDecodeRunSpec
from workflows.model_spec import MODEL_SPECS
from workflows.utils_report import BenchmarkTaskParams, BenchmarkTaskParamsCNN
from workflows.workflow_types import (
    BenchmarkTaskType,
    DeviceTypes,
    ModelType,
    WorkflowVenvType,
)


@dataclass(frozen=True)
class BenchmarkTask:
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_VLLM_API
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.BENCHMARKS_VLLM


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
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_VLLM_API
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.BENCHMARKS_VLLM


@dataclass(frozen=True)
class BenchmarkTaskVideo(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_VIDEO_API
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.BENCHMARKS_VIDEO


@dataclass(frozen=True)
class BenchmarkTaskTTS(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_CNN_API
    workflow_venv_type: WorkflowVenvType = (
        None  # no workflow venv needed for TTS benchmarks
    )


@dataclass(frozen=True)
class BenchmarkTaskImage(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_CNN_API
    workflow_venv_type: WorkflowVenvType = (
        None  # no workflow venv needed for image generation benchmarks
    )


@dataclass(frozen=True)
class BenchmarkTaskAudio(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = BenchmarkTaskType.HTTP_CLIENT_CNN_API
    workflow_venv_type: WorkflowVenvType = (
        None  # no workflow venv needed for audio transcription benchmarks
    )


@dataclass(frozen=True)
class BenchmarkTaskStructuredOutput(BenchmarkTask):
    param_map: Dict[DeviceTypes, List[BenchmarkTaskParams]]
    task_type: BenchmarkTaskType = (
        BenchmarkTaskType.HTTP_CLIENT_VLLM_STRUCTURED_OUTPUT_API
    )
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.BENCHMARKS_VLLM


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str
    tasks: List[BenchmarkTask]


BENCHMARK_ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (1024, 128),
    (2048, 128),
    (4096, 128),
    (8192, 128),
    (16384, 128),
    (32768, 128),
    (65536, 128),
]
SMOKE_TEST_BENCHMARK_PAIR = (16, 4)


# Profile definitions for the speculative-decoding benchmark
# (--workflow benchmarks --tools spec_decode). Selection happens at run time:
# --limit-samples-mode smoke-test → "smoke"; otherwise → "full".
#
# Two datasets are wired up:
#
#   * Spec-Bench (hemingkx/Spec-Bench, 480 prompts) — exposes 13 row-level
#     category values via question.jsonl. We use these one-at-a-time at
#     concurrency=1 to isolate per-content-type acceptance rates.
#   * SPEED-Bench (nvidia/SPEED-Bench, 2026) — two splits:
#       - "qualitative": semantically diverse prompts for drafter accuracy
#       - "throughput_{1k,2k,8k,16k,32k}": fixed-ISL prompts for system
#         throughput across input lengths under realistic batching.
#
# vLLM's SpecBench/SpeedBench loaders do exact-match category filtering on
# the JSONL ``category`` column. Passing a name that isn't in that column
# (e.g. "mt_bench" — that's the dataset name, not a row label — or "default"
# for SPEED-Bench) yields 0 prompts. ``category=None`` skips filtering and
# loads every row in the (sub)set, which is what we use for SPEED-Bench's
# wide qualitative + throughput sweeps.

# Spec-Bench's 13 row-level categories. The first 8 are MT-Bench's
# sub-categories, flattened into top-level rows by the Spec-Bench authors.
SPEC_BENCH_CATEGORIES = (
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
    "translation",
    "summarization",
    "qa",
    "math_reasoning",
    "rag",
)

# SPEED-Bench subsets (matches vllm 0.21 argparse choices).
SPEED_BENCH_QUALITATIVE = "qualitative"
SPEED_BENCH_THROUGHPUT_SUBSETS = (
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)

# Concurrency points for the throughput sweep. Spans single-stream to
# moderate batching so the diminishing-returns curve — where the workload
# becomes compute-bound and speculative decoding stops winning — is visible.
THROUGHPUT_CONCURRENCY_SWEEP = (1, 16, 64)

SPEC_DECODE_PROFILES: Dict[str, List[SpecDecodeRunSpec]] = {
    # CI-level smoke profile: tiny, finishes in seconds. Exercises both
    # dataset code paths with real category/subset values so a misconfigured
    # name (the historical "mt_bench" / "default" / "throughput" bugs) fails
    # loudly instead of silently loading 0 prompts.
    "smoke": [
        SpecDecodeRunSpec(
            dataset_kind="spec_bench",
            category="writing",
            output_len=128,
            max_concurrency=1,
            num_prompts=4,
        ),
        SpecDecodeRunSpec(
            dataset_kind="speed_bench",
            category=None,
            output_len=128,
            max_concurrency=1,
            num_prompts=4,
            speed_bench_subset=SPEED_BENCH_QUALITATIVE,
        ),
    ],
    # Full sweep:
    #   - All 13 Spec-Bench categories × 2 output lengths × conc=1
    #     → per-content-type acceptance rate, single-stream
    #   - SPEED-Bench qualitative whole-split × conc=1 × osl=2048
    #     → broad cross-domain AR check on the 880-prompt diverse split
    #   - SPEED-Bench throughput_{1k..32k} × conc{1,16,64} × osl=1024
    #     → maps E2E speedup across ISL and concurrency so the compute-bound
    #       regime is observable
    "full": [
        SpecDecodeRunSpec(
            dataset_kind="spec_bench",
            category=category,
            output_len=output_len,
            max_concurrency=1,
            num_prompts=16,
        )
        for category in SPEC_BENCH_CATEGORIES
        for output_len in (128, 512)
    ]
    + [
        SpecDecodeRunSpec(
            dataset_kind="speed_bench",
            category=None,
            output_len=2048,
            max_concurrency=1,
            num_prompts=64,
            speed_bench_subset=SPEED_BENCH_QUALITATIVE,
        )
    ]
    + [
        SpecDecodeRunSpec(
            dataset_kind="speed_bench",
            category=None,
            output_len=1024,
            max_concurrency=concurrency,
            num_prompts=max(32, 4 * concurrency),
            speed_bench_subset=subset,
        )
        for subset in SPEED_BENCH_THROUGHPUT_SUBSETS
        for concurrency in THROUGHPUT_CONCURRENCY_SWEEP
    ],
}


# Image resolution pairs for multimodal benchmarks
# Format here is isl, osl, image_height, image_width, images_per_prompt
ISL_OSL_IMAGE_RESOLUTION_PAIRS = [
    (128, 128, 512, 512, 1),  # Base resolution
    (128, 128, 1024, 1024, 1),
    (128, 128, 1024, 512, 1),
    (128, 128, 512, 1024, 1),
]


# format: (dataset, structured_output_ratio)
# vllm implements the following datasets: json, json-unique, grammar, regex, choice, xgrammar_bench, so they are all listed
# to see structured outputs charactization overhead, only json, json-unique and xgrammar_bench is needed, so other datasets are commented out
STRUCTURED_OUTPUT_PAIRS = [
    ("json", 1.0),
    ("json", 0.0),
    ("json-unique", 1.0),
    ("json-unique", 0.0),
    # ("grammar", 1.0),
    # ("regex", 1.0),
    # ("choice", 1.0),
    ("xgrammar_bench", 1.0),
    ("xgrammar_bench", 0.0),
]
STRUCTURED_OUTPUT_NUM_PROMPTS = 100
STRUCTURED_OUTPUT_OSL = 128
STRUCTURED_OUTPUT_MAX_CONCURRENCY = 4


def _expand_text_sweep_params(
    isl: int,
    osl: int,
    max_context: int,
    max_tokens_all_users: int,
    model_max_concurrency: int,
) -> List[BenchmarkTaskParams]:
    if isl + osl > max_context:
        return []

    allowed_max_concurrency = get_benchmark_max_concurrency(
        isl, osl, max_context, max_tokens_all_users, model_max_concurrency
    )
    concurrencies = [1]
    if allowed_max_concurrency > 1:
        concurrencies.append(allowed_max_concurrency)

    return [
        BenchmarkTaskParams(
            isl=isl,
            osl=osl,
            max_concurrency=concurrency,
            num_prompts=get_num_prompts(isl, osl, concurrency),
        )
        for concurrency in concurrencies
    ]


def _expand_image_sweep_params(
    isl: int,
    osl: int,
    image_height: int,
    image_width: int,
    images_per_prompt: int,
    max_context: int,
    max_tokens_all_users: int,
    model_max_concurrency: int,
    model_name: str,
) -> List[BenchmarkTaskParams]:
    vision_tokens = calculate_vision_tokens(
        image_height=image_height,
        image_width=image_width,
        images_per_prompt=images_per_prompt,
        model_name=model_name,
    )
    if isl + osl + vision_tokens > max_context:
        return []

    allowed_max_concurrency = get_benchmark_max_concurrency(
        isl,
        osl,
        max_context,
        max_tokens_all_users,
        model_max_concurrency,
        vision_tokens=vision_tokens,
    )
    concurrencies = [1]
    if allowed_max_concurrency > 1:
        concurrencies.append(allowed_max_concurrency)

    return [
        BenchmarkTaskParams(
            isl=isl,
            osl=osl,
            max_concurrency=concurrency,
            num_prompts=get_num_prompts(isl, osl, concurrency),
            task_type="vlm",
            image_height=image_height,
            image_width=image_width,
            images_per_prompt=images_per_prompt,
        )
        for concurrency in concurrencies
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
    isl,
    osl,
    max_context,
    max_tokens_all_users,
    model_max_concurrency=32,
    vision_tokens=0,
):
    """
    Calculate the maximum concurrency for benchmarks based on context limits.

    For VLM models, vision tokens must be included in the calculation to ensure
    accurate max_concurrency values that account for the full context usage.

    Args:
        isl: Input sequence length (text tokens)
        osl: Output sequence length
        max_context: Maximum context length supported by the model
        max_tokens_all_users: Maximum supported number of tokens in a batch at any given time
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

    # Calculate maximum concurrency that fits within total token budget
    max_concurrency_by_context = max_tokens_all_users // total_seq_len

    # Return the minimum of context-limited and model-limited concurrency
    return min(max_concurrency_by_context, model_max_concurrency)


def powers_of_two_up_to(max_value: int) -> List[int]:
    """
    Return [1, 2, 4, ...] up to and including max_value.
    """
    if max_value < 1:
        return []
    values: List[int] = []
    v = 1
    while v <= max_value:
        values.append(v)
        v *= 2
    return values


def _benchmark_param_dedupe_key(params: BenchmarkTaskParams) -> Tuple:
    # Include the fields that define benchmark uniqueness.
    return (
        getattr(params, "task_type", "text"),
        int(params.isl) if params.isl is not None else None,
        int(params.osl) if params.osl is not None else None,
        int(params.max_concurrency) if params.max_concurrency is not None else None,
        int(getattr(params, "image_height", 0) or 0),
        int(getattr(params, "image_width", 0) or 0),
        int(getattr(params, "images_per_prompt", 0) or 0),
        int(getattr(params, "num_inference_steps", 0) or 0),
        int(getattr(params, "num_eval_runs", 0) or 0),
    )


def select_smoke_test_benchmark_config(
    benchmark_config: BenchmarkConfig, device: DeviceTypes
) -> BenchmarkConfig:
    if benchmark_config.tasks:
        benchmark_target_task = benchmark_config.tasks[0]
        benchmark_targets = benchmark_target_task.param_map.get(device)
        if benchmark_targets:
            benchmark_target_param_map = dict(benchmark_target_task.param_map)
            benchmark_target_param_map[device] = list(benchmark_targets)
            return BenchmarkConfig(
                model_id=benchmark_config.model_id,
                tasks=[
                    replace(benchmark_target_task, param_map=benchmark_target_param_map)
                ],
            )

    smoke_isl, smoke_osl = SMOKE_TEST_BENCHMARK_PAIR
    smoke_num_prompts = get_num_prompts(smoke_isl, smoke_osl, 1)
    for task in benchmark_config.tasks[1:]:
        for params in task.param_map.get(device, []):
            if params.isl is None or params.osl is None:
                continue
            if getattr(params, "task_type", "text") != "text":
                continue

            smoke_param_map = dict(task.param_map)
            smoke_param_map[device] = [
                replace(
                    params,
                    isl=smoke_isl,
                    osl=smoke_osl,
                    max_concurrency=1,
                    num_prompts=smoke_num_prompts,
                )
            ]
            return BenchmarkConfig(
                model_id=benchmark_config.model_id,
                tasks=[replace(task, param_map=smoke_param_map)],
            )

    return BenchmarkConfig(model_id=benchmark_config.model_id, tasks=[])


def expand_concurrency_sweep_params(
    params_list: Iterable[BenchmarkTaskParams],
    *,
    max_context: int,
    max_tokens_all_users: int,
    model_max_concurrency: int,
    model_name: str,
    candidate_concurrencies: List[int],
    ensure_allowed_max: bool = True,
) -> List[BenchmarkTaskParams]:
    """
    Expand params_list to include candidate concurrencies (e.g. powers-of-2),
    capped by per-param allowed max concurrency.

    For image params, vision tokens are included in context accounting.
    CNN/audio/embedding params (without isl/osl) are returned unchanged.
    """
    expanded: List[BenchmarkTaskParams] = []
    seen = set()

    for params in params_list:
        # CNN/audio style params don't have isl/osl; keep them unchanged.
        if params.isl is None or params.osl is None:
            key = _benchmark_param_dedupe_key(params)
            if key not in seen:
                expanded.append(params)
                seen.add(key)
            continue

        isl = int(params.isl)
        osl = int(params.osl)

        # Reuse existing capping logic (includes vision tokens for VLM models).
        base_data = dict(vars(params))
        probe_data = dict(base_data)
        probe_data["max_concurrency"] = int(model_max_concurrency)
        probe_data["num_prompts"] = get_num_prompts(
            isl, osl, int(model_max_concurrency)
        )
        capped_probe = cap_benchmark_params(
            BenchmarkTaskParams(**probe_data),
            max_context=max_context,
            max_tokens_all_users=max_tokens_all_users,
            model_max_concurrency=model_max_concurrency,
            model_name=model_name,
        )
        allowed_max = int(capped_probe.max_concurrency)

        concurrencies = [
            int(c) for c in candidate_concurrencies if int(c) <= allowed_max
        ]
        if ensure_allowed_max and allowed_max not in concurrencies:
            concurrencies.append(allowed_max)
        concurrencies = sorted(set(concurrencies))

        for concurrency in concurrencies:
            new_data = dict(base_data)
            new_data["max_concurrency"] = int(concurrency)
            new_data["num_prompts"] = get_num_prompts(isl, osl, int(concurrency))

            new_params = BenchmarkTaskParams(**new_data)
            key = _benchmark_param_dedupe_key(new_params)
            if key not in seen:
                expanded.append(new_params)
                seen.add(key)

    return expanded


def cap_benchmark_params(
    params: BenchmarkTaskParams,
    max_context: int,
    max_tokens_all_users: int,
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
    if params.task_type == "vlm" and params.image_height and params.image_width:
        vision_tokens = calculate_vision_tokens(
            params.image_height,
            params.image_width,
            params.images_per_prompt or 0,
            model_name,
        )

    # Calculate the allowed max_concurrency based on sequence length (including vision tokens)
    calculated_max_concurrency = get_benchmark_max_concurrency(
        params.isl,
        params.osl,
        max_context,
        max_tokens_all_users,
        model_max_concurrency,
        vision_tokens,
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
    max_tokens_all_users = model_spec.device_model_spec.max_tokens_all_users
    perf_reference = model_spec.device_model_spec.perf_reference

    # Apply capping to each perf reference entry (including vision tokens for VLM models)
    capped_perf_reference = [
        cap_benchmark_params(
            params,
            max_context,
            max_tokens_all_users,
            model_max_concurrency,
            model_spec.model_name,
        )
        for params in perf_reference
    ]

    # Create performance reference task with capped values
    if model_spec.model_type == ModelType.CNN:
        perf_ref_task = BenchmarkTaskCNN(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.EMBEDDING:
        perf_ref_task = BenchmarkTaskEmbedding(
            param_map={device: capped_perf_reference}
        )
    elif model_spec.model_type == ModelType.VIDEO:
        perf_ref_task = BenchmarkTaskVideo(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
        perf_ref_task = BenchmarkTaskTTS(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.IMAGE:
        perf_ref_task = BenchmarkTaskImage(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.AUDIO:
        perf_ref_task = BenchmarkTaskAudio(param_map={device: capped_perf_reference})
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
        elif model_spec.model_type == ModelType.VIDEO:
            benchmark_task_runs = BenchmarkTaskVideo(
                param_map={device: [BenchmarkTaskParams()]}
            )
        elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
            benchmark_task_runs = BenchmarkTaskTTS(
                param_map={
                    device: [
                        BenchmarkTaskParams(
                            max_concurrency=model_max_concurrency,
                            num_prompts=8,
                            task_type="tts",
                        )
                    ]
                }
            )
        elif model_spec.model_type == ModelType.IMAGE:
            benchmark_task_runs = BenchmarkTaskImage(
                param_map={device: [BenchmarkTaskParams()]}
            )
        elif model_spec.model_type == ModelType.AUDIO:
            benchmark_task_runs = BenchmarkTaskAudio(
                param_map={device: [BenchmarkTaskParams()]}
            )
        else:
            benchmark_task_runs = BenchmarkTask(
                param_map={
                    device: [
                        expanded_params
                        for isl, osl in BENCHMARK_ISL_OSL_PAIRS
                        if isl + osl <= max_context
                        for expanded_params in _expand_text_sweep_params(
                            isl=isl,
                            osl=osl,
                            max_context=max_context,
                            max_tokens_all_users=max_tokens_all_users,
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
                                max_tokens_all_users=max_tokens_all_users,
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

    # Structured-output benchmarks: llms and vlms, can be extended
    structured_output_eligible = model_spec.model_type in (ModelType.LLM, ModelType.VLM)
    if structured_output_eligible:
        tasks.append(
            BenchmarkTaskStructuredOutput(
                param_map={
                    device: [
                        BenchmarkTaskParams(
                            osl=STRUCTURED_OUTPUT_OSL,
                            max_concurrency=STRUCTURED_OUTPUT_MAX_CONCURRENCY,
                            num_prompts=STRUCTURED_OUTPUT_NUM_PROMPTS,
                            task_type="structured_output",
                            structured_dataset=dataset,
                            structured_output_ratio=ratio,
                        )
                        for dataset, ratio in STRUCTURED_OUTPUT_PAIRS
                    ]
                }
            )
        )

    BENCHMARK_CONFIGS[model_id] = BenchmarkConfig(model_id=model_id, tasks=tasks)
