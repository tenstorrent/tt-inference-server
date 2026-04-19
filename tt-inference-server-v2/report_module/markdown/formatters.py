# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Display-dict factories for each benchmark task type.

Each function takes a raw metrics dict (one row from a parsed benchmark JSON)
and returns a Dict[str, str] mapping human-readable column headers to cell
values.  These dicts are consumed by ``table_builder.get_markdown_table``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Tuple

from report_module.types import NOT_MEASURED_STR
from workflows.utils import (
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)

logger = logging.getLogger(__name__)


def format_backend_value(backend: str) -> str:
    if backend == "vllm":
        return "vLLM"
    if backend == "genai-perf":
        return "genai"
    return backend if backend else NOT_MEASURED_STR


def _build_display_dict(
    result: Dict[str, Any],
    display_cols: List[Tuple[str, str]],
    value_overrides: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Generic helper that maps internal keys to display headers."""
    overrides = value_overrides or {}
    display_dict: Dict[str, str] = {}
    for col_name, display_header in display_cols:
        if col_name in overrides:
            display_dict[display_header] = overrides[col_name]
            continue
        value = result.get(col_name, NOT_MEASURED_STR)
        if col_name == "backend":
            value = format_backend_value(value)
        display_dict[display_header] = str(value)
    return display_dict


# -- Concrete formatters per task type ----------------------------------------

def create_text_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("max_con", "Concurrency"),
        ("num_requests", "N Req"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "Tput User (TPS)"),
        ("tps_decode_throughput", "Tput Decode (TPS)"),
        ("tps_prefill_throughput", "Tput Prefill (TPS)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
        ("total_token_throughput", "Total Token Throughput (tokens/duration)"),
    ]
    return _build_display_dict(result, display_cols)


def create_vlm_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_con", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "Tput User (TPS)"),
        ("tps_decode_throughput", "Tput Decode (TPS)"),
        ("tps_prefill_throughput", "Tput Prefill (TPS)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    overrides: Dict[str, str] = {}
    for key, fallback in [("isl", "input_sequence_length"), ("osl", "output_sequence_length")]:
        overrides[key] = str(result.get(key, result.get(fallback, NOT_MEASURED_STR)))

    return _build_display_dict(result, display_cols, value_overrides=overrides)


class _ModelSpecWrapper:
    """Adapter so whisper helpers can accept a bare ``ModelSpec``.

    ``is_streaming_enabled_for_whisper`` / ``is_preprocessing_enabled_for_whisper``
    expect an object exposing ``.model_spec``; this wrapper provides that
    attribute without changing the helpers' public contract.
    """

    __slots__ = ("model_spec",)

    def __init__(self, spec: Any) -> None:
        self.model_spec = spec


def create_audio_display_dict(
    result: Dict[str, Any], model_spec: Any
) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("streaming_enabled", "Streaming enabled"),
        ("preprocessing_enabled", "Preprocessing enabled"),
        ("accuracy_check", "Accuracy Check"),
        ("t/s/u", "T/S/U"),
        ("rtr", "RTR"),
    ]
    wrapper = _ModelSpecWrapper(model_spec)
    overrides = {
        "streaming_enabled": str(is_streaming_enabled_for_whisper(wrapper)),
        "preprocessing_enabled": str(is_preprocessing_enabled_for_whisper(wrapper)),
    }
    return _build_display_dict(result, display_cols, value_overrides=overrides)


def create_tts_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("rtr", "RTR"),
        ("p90_ttft", "P90 TTFT (ms)"),
        ("p95_ttft", "P95 TTFT (ms)"),
    ]
    return _build_display_dict(result, display_cols)


def create_embedding_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("max_con", "Max Concurrency"),
        ("embedding_dimension", "Embedding Dimension"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "Tput User (TPS)"),
        ("tps_decode_throughput", "Tput Decode (TPS)"),
        ("tps_prefill_throughput", "Tput Prefill (TPS)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]
    return _build_display_dict(result, display_cols)


def create_image_generation_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("num_inference_steps", "Inference Steps"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("inference_steps_per_second", "Steps/Sec"),
    ]
    return _build_display_dict(result, display_cols)


def create_cnn_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("num_inference_steps", "Num Inference Steps"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("task_type", "Task Type"),
    ]
    return _build_display_dict(result, display_cols)


def create_video_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    logger.info(f"Video result: {json.dumps(result, indent=2)}")
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("num_inference_steps", "Num Inference Steps"),
        ("mean_ttft_ms", "TTFT (ms)"),
    ]
    return _build_display_dict(result, display_cols)


TASK_TYPE_FORMATTER_MAP: Dict[str, Callable[..., Dict[str, str]]] = {
    "text": create_text_display_dict,
    "vlm": create_vlm_display_dict,
    "tts": create_tts_display_dict,
    "text_to_speech": create_tts_display_dict,
    "embedding": create_embedding_display_dict,
    "image": create_image_generation_display_dict,
    "image_generation": create_image_generation_display_dict,
    "cnn": create_cnn_display_dict,
    "video": create_video_display_dict,
}
