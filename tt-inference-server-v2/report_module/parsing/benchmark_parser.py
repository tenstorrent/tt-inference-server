# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Benchmark JSON file parsing — extracted from benchmarking/summary_report.py.

Handles vLLM, AIPerf, GenAI-Perf, CNN, image, audio, TTS, embedding, and video
benchmark result files.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from report_module.parsing.common import format_metrics
from report_module.types import NOT_MEASURED_STR
from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import ModelType

logger = logging.getLogger(__name__)


def _map_model_type_to_task_type(model_type: ModelType) -> Optional[str]:
    mapping = {
        ModelType.LLM: "text",
        ModelType.CNN: "cnn",
        ModelType.AUDIO: "audio",
        ModelType.IMAGE: "image",
        ModelType.VLM: "vlm",
        ModelType.EMBEDDING: "embedding",
        ModelType.VIDEO: "video",
        ModelType.TEXT_TO_SPEECH: "text_to_speech",
    }
    return mapping.get(model_type)


def _get_task_type(model_id: str) -> str:
    model_name = model_id.lower().split("_")[-1]
    for _, model_spec in MODEL_SPECS.items():
        if model_name in model_spec.model_name.lower() and model_spec.model_type:
            result = _map_model_type_to_task_type(model_spec.model_type)
            if result:
                return result
    return "unknown"


_IMAGE_GEN_KEYWORDS = ("stable-diffusion", "sdxl", "sd-", "sd3")

_AIPERF_IMAGE_PATTERN = re.compile(
    r"^aiperf_benchmark_"
    r"(?P<model>.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|n150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_isl-(?P<isl>\d+)_osl-(?P<osl>\d+)_maxcon-(?P<maxcon>\d+)_n-(?P<n>\d+)"
    r"_images-(?P<images_per_prompt>\d+)_height-(?P<image_height>\d+)_width-(?P<image_width>\d+)"
    r"\.json$"
)

_AIPERF_TEXT_PATTERN = re.compile(
    r"^aiperf_benchmark_"
    r"(?P<model>.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|n150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_isl-(?P<isl>\d+)_osl-(?P<osl>\d+)_maxcon-(?P<maxcon>\d+)_n-(?P<n>\d+)"
    r"\.json$"
)

_IMAGE_PATTERN = re.compile(
    r"^(?:genai_)?benchmark_"
    r"(?P<model>.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|TG|GALAXY|n150|n300|p100|p150|galaxy_t3k|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_isl-(?P<isl>\d+)_osl-(?P<osl>\d+)_maxcon-(?P<maxcon>\d+)_n-(?P<n>\d+)"
    r"_images-(?P<images_per_prompt>\d+)_height-(?P<image_height>\d+)_width-(?P<image_width>\d+)"
    r"\.json$"
)

_TEXT_PATTERN = re.compile(
    r"^(?:genai_)?benchmark_"
    r"(?P<model>.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|n150x4|TG|GALAXY|n150|n300|p100|p150|galaxy_t3k|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_isl-(?P<isl>\d+)_osl-(?P<osl>\d+)_maxcon-(?P<maxcon>\d+)_n-(?P<n>\d+)"
    r"\.json$"
)

_CNN_PATTERN = re.compile(
    r"^benchmark_"
    r"(?P<model_id>id_.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d+\.?\d*)"
    r"\.json$"
)


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    """Extract benchmark parameters from a benchmark filename."""
    match = _AIPERF_IMAGE_PATTERN.search(filename)
    if match:
        model_name = match.group("model")
        is_image_gen = any(kw in model_name.lower() for kw in _IMAGE_GEN_KEYWORDS)
        return {
            "model_name": model_name,
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "input_sequence_length": int(match.group("isl")),
            "output_sequence_length": int(match.group("osl")),
            "max_con": int(match.group("maxcon")),
            "num_requests": int(match.group("n")),
            "images_per_prompt": int(match.group("images_per_prompt")),
            "image_height": int(match.group("image_height")),
            "image_width": int(match.group("image_width")),
            "task_type": "image" if is_image_gen else "vlm",
            "backend": "aiperf",
        }

    match = _AIPERF_TEXT_PATTERN.search(filename)
    if match:
        return {
            "model_name": match.group("model"),
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "input_sequence_length": int(match.group("isl")),
            "output_sequence_length": int(match.group("osl")),
            "max_con": int(match.group("maxcon")),
            "num_requests": int(match.group("n")),
            "task_type": "text",
            "backend": "aiperf",
        }

    match = _IMAGE_PATTERN.search(filename)
    if match:
        model_name = match.group("model")
        is_image_gen = any(kw in model_name.lower() for kw in _IMAGE_GEN_KEYWORDS)
        return {
            "model_name": model_name,
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "input_sequence_length": int(match.group("isl")),
            "output_sequence_length": int(match.group("osl")),
            "max_con": int(match.group("maxcon")),
            "num_requests": int(match.group("n")),
            "images_per_prompt": int(match.group("images_per_prompt")),
            "image_height": int(match.group("image_height")),
            "image_width": int(match.group("image_width")),
            "task_type": "image" if is_image_gen else "vlm",
        }

    match = _TEXT_PATTERN.search(filename)
    if match:
        return {
            "model_name": match.group("model"),
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "input_sequence_length": int(match.group("isl")),
            "output_sequence_length": int(match.group("osl")),
            "max_con": int(match.group("maxcon")),
            "num_requests": int(match.group("n")),
            "task_type": "text",
        }

    match = _CNN_PATTERN.search(filename)
    if match:
        model_id = match.group("model_id")
        return {
            "model_id": model_id,
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "task_type": _get_task_type(model_id),
        }

    raise ValueError(f"Could not extract parameters from filename: {filename}")


def process_benchmark_file(filepath: str) -> Dict[str, Any]:
    """Process a single benchmark file and extract normalised metrics."""
    logger.info(f"Processing benchmark file: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)
    params = extract_params_from_filename(filename)

    if params.get("backend") == "aiperf":
        return _process_aiperf(data, params, filename)

    benchmarks_data = data.get("benchmarks: ", data)
    if benchmarks_data and benchmarks_data.get("benchmarks"):
        task = params.get("task_type")
        if task == "cnn":
            return _process_cnn_or_image(data, benchmarks_data, params, filename, "cnn")
        if task == "image":
            return _process_cnn_or_image(data, benchmarks_data, params, filename, "image")

    task = params.get("task_type")
    if task in ("text_to_speech", "tts"):
        return _process_tts(data, params, filename)
    if task == "audio":
        return _process_audio(data, params, filename)
    if task == "embedding":
        return _process_embedding(data, params, filename)
    if task == "video":
        return _process_video(data, params, filename)

    return _process_default(data, params, filename)


def _process_aiperf(data, params, filename):
    mean_tpot_ms = data.get("mean_tpot_ms", 0)
    mean_tps = (1000.0 / mean_tpot_ms) if mean_tpot_ms and mean_tpot_ms > 0 else None
    std_tps = None
    if mean_tps and data.get("std_tpot_ms"):
        std_tps = mean_tps - (1000.0 / (mean_tpot_ms + data["std_tpot_ms"]))

    actual_max_con = min(params["max_con"], params["num_requests"])
    tps_decode = mean_tps * actual_max_con if mean_tps else None
    tps_prefill = None
    ttft = data.get("mean_ttft_ms")
    if ttft and ttft > 0:
        tps_prefill = (params["input_sequence_length"] * actual_max_con) / (ttft / 1000)

    metrics = {
        "timestamp": params["timestamp"],
        "model_name": params["model_name"],
        "model_id": data.get("model_id", ""),
        "backend": "aiperf",
        "device": params.get("device", ""),
        "input_sequence_length": params["input_sequence_length"],
        "output_sequence_length": params["output_sequence_length"],
        "max_con": actual_max_con,
        "mean_ttft_ms": ttft,
        "std_ttft_ms": data.get("std_ttft_ms"),
        "mean_tpot_ms": mean_tpot_ms,
        "std_tpot_ms": data.get("std_tpot_ms"),
        "mean_tps": mean_tps,
        "std_tps": std_tps,
        "tps_decode_throughput": tps_decode,
        "tps_prefill_throughput": tps_prefill,
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "request_throughput": data.get("request_throughput"),
        "total_token_throughput": data.get("total_token_throughput"),
        "total_input_tokens": data.get("total_input_tokens"),
        "total_output_tokens": data.get("total_output_tokens"),
        "num_prompts": data.get("num_prompts", ""),
        "num_requests": params["num_requests"],
        "filename": filename,
        "task_type": params["task_type"],
    }
    if params["task_type"] in ("image", "vlm"):
        metrics["images_per_prompt"] = params.get("images_per_prompt", 1)
        metrics["image_height"] = params.get("image_height", 0)
        metrics["image_width"] = params.get("image_width", 0)
    return format_metrics(metrics)


def _process_cnn_or_image(data, benchmarks_data, params, filename, task_type):
    bench = benchmarks_data["benchmarks"]
    metrics = {
        "timestamp": params["timestamp"],
        "model": data.get("model", ""),
        "model_name": data.get("model", ""),
        "model_id": data.get("model", ""),
        "backend": task_type,
        "device": params["device"],
        "num_requests": bench.get("num_requests", 0),
        "num_inference_steps": bench.get("num_inference_steps", 0),
        "mean_ttft_ms": bench.get("ttft", 0) * 1000,
        "inference_steps_per_second": bench.get("inference_steps_per_second", 0),
        "filename": filename,
        "task_type": task_type,
    }
    return format_metrics(metrics)


def _process_tts(data, params, filename):
    bench = data.get("benchmarks", {})
    metrics = {
        "timestamp": params["timestamp"],
        "model": data.get("model", ""),
        "model_name": data.get("model", ""),
        "model_id": data.get("model", ""),
        "backend": "text_to_speech",
        "device": params["device"],
        "num_requests": bench.get("num_requests", 0),
        "mean_ttft_ms": bench.get("ttft", 0) * 1000,
        "filename": filename,
        "task_type": "tts",
        "rtr": bench.get("rtr", 0),
        "p90_ttft": bench.get("ttft_p90", 0) * 1000 if bench.get("ttft_p90") else None,
        "p95_ttft": bench.get("ttft_p95", 0) * 1000 if bench.get("ttft_p95") else None,
        "wer": bench.get("wer"),
    }
    return format_metrics(metrics)


def _process_audio(data, params, filename):
    benchmarks_data = data.get("benchmarks: ", data)
    bench = benchmarks_data.get("benchmarks", {})
    metrics = {
        "timestamp": params["timestamp"],
        "model": data.get("model", ""),
        "model_name": data.get("model", ""),
        "model_id": data.get("model", ""),
        "backend": "audio",
        "device": params["device"],
        "num_requests": bench.get("num_requests", 0),
        "mean_ttft_ms": bench.get("ttft", 0) * 1000,
        "filename": filename,
        "task_type": "audio",
        "accuracy_check": bench.get("accuracy_check", 0),
        "t/s/u": bench.get("t/s/u", 0),
        "rtr": bench.get("rtr", 0),
        "streaming_enabled": data.get("streaming_enabled", False),
        "preprocessing_enabled": data.get("preprocessing_enabled", False),
    }
    return format_metrics(metrics)


def _process_embedding(data, params, filename):
    benchmarks_data = data.get("benchmarks: ", data)
    bench = benchmarks_data.get("benchmarks", {})
    metrics = {
        "timestamp": params["timestamp"],
        "model": data.get("model", ""),
        "model_name": data.get("model", ""),
        "model_id": data.get("model", ""),
        "backend": "embedding",
        "device": params["device"],
        "filename": filename,
        "task_type": "embedding",
        "num_requests": bench.get("num_requests", 0),
        "input_sequence_length": bench.get("isl", 0),
        "output_sequence_length": NOT_MEASURED_STR,
        "max_con": bench.get("concurrency", 0),
        "embedding_dimension": bench.get("embedding_dimension", NOT_MEASURED_STR),
        "mean_ttft_ms": NOT_MEASURED_STR,
        "mean_tpot_ms": NOT_MEASURED_STR,
        "mean_tps": bench.get("tput_user", 0.0),
        "tps_decode_throughput": NOT_MEASURED_STR,
        "tps_prefill_throughput": bench.get("tput_prefill", 0.0),
        "mean_e2el_ms": bench.get("e2el", 0.0),
        "request_throughput": bench.get("req_tput", 0.0),
    }
    return format_metrics(metrics)


def _process_video(data, params, filename):
    benchmarks_data = data.get("benchmarks: ", data)
    bench = benchmarks_data.get("benchmarks", {})
    metrics = {
        "timestamp": params["timestamp"],
        "model": data.get("model", ""),
        "model_name": data.get("model", ""),
        "model_id": data.get("model", ""),
        "backend": "video",
        "device": params["device"],
        "filename": filename,
        "task_type": "video",
        "num_requests": bench.get("num_requests", 0),
        "mean_ttft_ms": bench.get("ttft", 0) * 1000,
        "inference_steps_per_second": bench.get("inference_steps_per_second", 0),
        "num_inference_steps": bench.get("num_inference_steps", 0),
    }
    return format_metrics(metrics)


def _process_default(data, params, filename):
    mean_tpot_ms = data.get("mean_tpot_ms")
    if mean_tpot_ms:
        mean_tpot = max(mean_tpot_ms, 1e-6)
        mean_tps = 1000.0 / mean_tpot
        std_tps = (mean_tps - (1000.0 / (mean_tpot + data["std_tpot_ms"]))) if data.get("std_tpot_ms") else None
    else:
        mean_tps = None
        std_tps = None

    actual_max_con = min(params["max_con"], params["num_requests"])
    tps_decode = mean_tps * actual_max_con if mean_tps else None
    tps_prefill = (params["input_sequence_length"] * actual_max_con) / (data.get("mean_ttft_ms") / 1000)

    metrics = {
        "timestamp": params["timestamp"],
        "model_name": params["model_name"],
        "model_id": data.get("model_id", ""),
        "backend": data.get("backend", ""),
        "device": params.get("device", ""),
        "input_sequence_length": params["input_sequence_length"],
        "output_sequence_length": params["output_sequence_length"],
        "max_con": actual_max_con,
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "std_ttft_ms": data.get("std_ttft_ms"),
        "mean_tpot_ms": mean_tpot_ms,
        "std_tpot_ms": data.get("std_tpot_ms"),
        "mean_tps": mean_tps,
        "std_tps": std_tps,
        "tps_decode_throughput": tps_decode,
        "tps_prefill_throughput": tps_prefill,
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "request_throughput": data.get("request_throughput"),
        "total_input_tokens": data.get("total_input_tokens"),
        "total_output_tokens": data.get("total_output_tokens"),
        "total_token_throughput": data.get("total_token_throughput"),
        "num_prompts": data.get("num_prompts", ""),
        "num_requests": params["num_requests"],
        "filename": filename,
        "task_type": params["task_type"],
    }
    if params["task_type"] in ("image", "vlm"):
        metrics["images_per_prompt"] = params["images_per_prompt"]
        metrics["image_height"] = params["image_height"]
        metrics["image_width"] = params["image_width"]
    return format_metrics(metrics)


def process_benchmark_files(files: List[str]) -> List[Dict[str, Any]]:
    """Process multiple benchmark files, returning sorted metric dicts."""
    results: List[Dict[str, Any]] = []
    logger.info(f"Processing {len(files)} benchmark files")
    for filepath in files:
        try:
            metrics = process_benchmark_file(filepath)
            results.append(metrics)
        except Exception:
            logger.exception(f"Error processing file {filepath}")
    if not results:
        raise ValueError("No benchmark files were successfully processed")
    return sorted(results, key=lambda x: x.get("timestamp", ""))
