# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import logging
import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

from workflows.utils_report import BenchmarkTaskParams, BenchmarkTaskParamsCNN
from workflows.workflow_types import (
    BenchmarkTaskType,
    DeviceTypes,
    InferenceEngine,
    ModelType,
    WorkflowVenvType,
)

logger = logging.getLogger(__name__)


# Forge models need a newer vllm client that can load their tokenizers; other
# engines use the shared client.
_VLLM_BENCHMARK_VENV_BY_ENGINE = {
    InferenceEngine.FORGE.value: WorkflowVenvType.BENCHMARKS_VLLM_FORGE,
}


def select_vllm_benchmark_venv(model_spec) -> WorkflowVenvType:
    """Pick the vllm benchmark client venv for ``model_spec``."""
    return _VLLM_BENCHMARK_VENV_BY_ENGINE.get(
        model_spec.inference_engine, WorkflowVenvType.BENCHMARKS_VLLM
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
    (8192, 1024),
    (10000, 1024),
    (16384, 128),
    (32768, 128),
    (65536, 128),
    (131072, 128),
]
# Additional high-ISL sweep points appended only for remote SUPER_CLUSTER
# endpoints, whose token budget is context*concurrency (see
# DeviceModelSpec._infer_data) so concurrency does not collapse at high ISL.
# They extend the sweep toward ~250K ISL while staying below the 256K cap, and
# are still filtered per model by ``isl + osl <= max_context`` at build time
# (e.g. reachable by Kimi's 256K context, skipped for a 128K-context model).
# NOTE: To support future models with larger context lengths, add near-max ISL-OSL
# pairs here; they will be automatically filtered by the per-model context cap.
SUPER_CLUSTER_EXTRA_ISL_OSL_PAIRS = [
    (128, 256000 - 128),
    (196608, 128),  # 192K
    (256000 - 128, 128),  # 240K
]
# Remote SUPER_CLUSTER endpoints serve high-ISL sweep points at full
# concurrency, but get_num_prompts scales prompts as a small multiple of
# concurrency, so long-sequence points issue too few requests (e.g. 1x = 64)
# to characterize steady-state throughput. Floor the prompt count so each
# SUPER_CLUSTER sweep point sends at least this many multiples of the model's
# batch size (the spec's max_concurrency).
SUPER_CLUSTER_MIN_NUM_PROMPTS_BATCH_MULTIPLE = 2
SMOKE_TEST_BENCHMARK_PAIR = (16, 4)

# Per-model explicit operating-point sets, overriding the shared sweep above.
#
# The shared sweep is a generic ISL ladder at OSL 128/1024 with concurrency
# [1, model_max_concurrency]. When a model has a requirements document that
# names an explicit (ISL, OSL, concurrency) set, the generic sweep can end up
# measuring nothing that maps to a requirement while spending its budget on
# points nobody asked for.
#
# qwen3.8-27b (agentic coding) Rev 0.11 is such a case. It requires OSL 252 --
# the TraceLab per-step median agentic output -- at concurrency 1, 8 and 16, and
# caps concurrency at 8 for >=128K context. Against the shared sweep, none of
# the 19 points it generated for this model matched any requirement row (OSL was
# never 252), five of them were concurrency 32 which no requirement asks for,
# and those five consumed 10 of the job's 18 hours by each hitting the 7200 s
# per-point timeout without producing a result. See the gap analysis in tt-metal
# models/autoports/qwen_qwen3_6_27b/doc/benchmark_requirements_gap.
#
# Entries are (isl, osl, max_concurrency). num_prompts still comes from
# get_num_prompts, and the per-model ``isl + osl <= max_context`` filter still
# applies, so a model whose context cannot hold a point simply drops it.
MODEL_EXPLICIT_TEXT_SWEEP: Dict[str, List[Tuple[int, int, int]]] = {
    "Qwen3.8-27B": [
        # Rev 0.11 section 3, batch 1 -- agentic coding without subagents.
        (128, 252, 1),
        (1024, 252, 1),
        (4096, 252, 1),
        (16384, 252, 1),
        (32768, 252, 1),
        (65536, 252, 1),
        (131072, 252, 1),
        # Rev 0.11 names ISL 262144 at OSL 252, which exceeds this model's
        # 262144 context and would be dropped by the isl+osl filter. Use the
        # largest ISL that still leaves room for the required output, so the
        # near-maximum-context point is actually measured.
        (262144 - 252, 252, 1),
        # batch 8 -- agentic coding with subagents.
        (4096, 252, 8),
        (32768, 252, 8),
        (131072, 252, 8),
        # batch 16 -- with subagents at short/mid context only; Rev 0.11 caps
        # concurrency at 8 for >=128K, so there is deliberately no 131072 row.
        (4096, 252, 16),
        (32768, 252, 16),
    ],
}


def _normalize_model_key(value: str) -> str:
    """Lowercase alphanumeric form of a model identifier.

    The same model reaches this code as "Qwen3.8-27B", "Qwen/Qwen3.8-27B" or a
    slugified model_id depending on whether the spec came from the catalog, a
    runtime spec JSON or off-catalog synthesis. Comparing normalized forms means
    the override cannot miss on punctuation or casing -- and a miss is otherwise
    invisible, because the run would silently fall back to the generic sweep.
    """
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


_MODEL_EXPLICIT_TEXT_SWEEP_NORMALIZED = {
    _normalize_model_key(name): points
    for name, points in MODEL_EXPLICIT_TEXT_SWEEP.items()
}


def get_explicit_text_sweep(model_spec) -> List[Tuple[int, int, int]]:
    """Explicit requirement point set for this model, or None."""
    for candidate in (
        getattr(model_spec, "model_name", None),
        str(getattr(model_spec, "hf_model_repo", "") or "").rsplit("/", 1)[-1],
        getattr(model_spec, "model_id", None),
    ):
        points = _MODEL_EXPLICIT_TEXT_SWEEP_NORMALIZED.get(
            _normalize_model_key(candidate)
        )
        if points:
            return points
    return None


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
    min_num_prompts: int = 0,
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
            num_prompts=get_num_prompts(
                isl,
                osl,
                concurrency,
                min_num_prompts=min_num_prompts if concurrency > 1 else 0,
            ),
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


def get_num_prompts(input_len, output_len, max_concurrency, *, min_num_prompts=0):
    # Large sequences (slowest) -> fewest prompts
    if output_len > 1024 or input_len > 16384:
        base = 1 * max_concurrency
    elif input_len > 4096:
        base = 2 * max_concurrency
    # Medium sequences
    elif (output_len > 128 and output_len <= 1024) or (
        input_len > 128 and input_len <= 4096
    ):
        base = 4 * max_concurrency
    # Small sequences (fastest) -> most prompts
    elif output_len <= 128:
        base = 8 * max_concurrency
    else:
        raise ValueError(f"Invalid output_len: {output_len}")

    return max(base, min_num_prompts)


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
    min_num_prompts: int = 0,
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
            new_data["num_prompts"] = get_num_prompts(
                isl,
                osl,
                int(concurrency),
                min_num_prompts=min_num_prompts if concurrency > 1 else 0,
            )

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
    min_num_prompts: int = 0,
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
            params.isl,
            params.osl,
            capped_max_concurrency,
            min_num_prompts=min_num_prompts,
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


def build_benchmark_config(model_spec) -> BenchmarkConfig:
    """Build benchmark tasks directly from a resolved model spec.

    Runtime model specs supplied through ``--runtime-model-spec-json`` may not be
    present in import-time ``MODEL_SPECS``. They still carry the same
    ``device_model_spec`` fields needed to generate benchmark tasks.
    """

    # Since each ModelConfig now represents a single device, use that device and its max_concurrency
    device = model_spec.device_type
    model_max_concurrency = model_spec.device_model_spec.max_concurrency
    max_context = model_spec.device_model_spec.max_context
    max_tokens_all_users = model_spec.device_model_spec.max_tokens_all_users
    perf_reference = model_spec.device_model_spec.perf_reference

    # SUPER_CLUSTER remote endpoints extend the sweep toward ~250K ISL; other
    # devices use the standard pairs. Per-model ``isl + osl <= max_context``
    # filtering still applies below.
    text_isl_osl_pairs = list(BENCHMARK_ISL_OSL_PAIRS)
    sweep_min_num_prompts = 0
    if device == DeviceTypes.SUPER_CLUSTER:
        text_isl_osl_pairs += SUPER_CLUSTER_EXTRA_ISL_OSL_PAIRS
        sweep_min_num_prompts = (
            SUPER_CLUSTER_MIN_NUM_PROMPTS_BATCH_MULTIPLE * model_max_concurrency
        )

    vllm_benchmark_venv = select_vllm_benchmark_venv(model_spec)

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
    elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
        perf_ref_task = BenchmarkTaskTTS(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.IMAGE:
        perf_ref_task = BenchmarkTaskImage(param_map={device: capped_perf_reference})
    elif model_spec.model_type == ModelType.AUDIO:
        perf_ref_task = BenchmarkTaskAudio(param_map={device: capped_perf_reference})
    else:
        perf_ref_task = BenchmarkTask(
            param_map={device: capped_perf_reference},
            workflow_venv_type=vllm_benchmark_venv,
        )

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
            explicit_points = get_explicit_text_sweep(model_spec)
            logger.info(
                "benchmark sweep for model_name=%r hf_repo=%r: %s",
                model_spec.model_name,
                model_spec.hf_model_repo,
                (
                    f"explicit requirement set, {len(explicit_points)} points"
                    if explicit_points
                    else f"generic sweep, {len(text_isl_osl_pairs)} isl/osl pairs"
                ),
            )
            if explicit_points is not None:
                text_sweep_params = [
                    BenchmarkTaskParams(
                        isl=isl,
                        osl=osl,
                        max_concurrency=concurrency,
                        num_prompts=get_num_prompts(isl, osl, concurrency),
                    )
                    for isl, osl, concurrency in explicit_points
                    if isl + osl <= max_context
                ]
            else:
                text_sweep_params = [
                    expanded_params
                    for isl, osl in text_isl_osl_pairs
                    if isl + osl <= max_context
                    for expanded_params in _expand_text_sweep_params(
                        isl=isl,
                        osl=osl,
                        max_context=max_context,
                        max_tokens_all_users=max_tokens_all_users,
                        model_max_concurrency=model_max_concurrency,
                        min_num_prompts=sweep_min_num_prompts,
                    )
                ]
            benchmark_task_runs = BenchmarkTask(
                param_map={
                    device: text_sweep_params
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
                },
                workflow_venv_type=vllm_benchmark_venv,
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
                },
                workflow_venv_type=vllm_benchmark_venv,
            )
        )

    return BenchmarkConfig(model_id=model_spec.model_id, tasks=tasks)


def get_benchmark_config(model_spec) -> BenchmarkConfig:
    """Build benchmark tasks from the resolved model spec.

    ``--runtime-model-spec-json`` is already resolved into ``model_spec`` before
    this helper runs. Do not consult the import-time catalog here: the runtime
    JSON must override even when its ``model_id`` collides with a built-in spec.
    """
    return build_benchmark_config(model_spec)
