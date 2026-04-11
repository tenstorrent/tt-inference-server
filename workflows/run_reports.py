# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import csv
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from evals.eval_config import EVAL_CONFIGS
from stress_tests.stress_tests_summary_report import (
    generate_report as stress_test_generate_report_helper,
)
from server_tests.utils.vllm_parameter_json_to_md import (
    main as generate_vllm_parameter_report,
)
from workflows.acceptance_criteria import (
    acceptance_criteria_check,
    evaluate_benchmark_targets,
    format_acceptance_summary_markdown,
)
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import MODEL_SPECS, ModelSpec
from workflows.perf_targets import get_perf_target_rows
from workflows.release_report_markdown import build_release_report_markdown
from workflows.reports_schema import validate_report_file, write_reports_schema
from workflows.runtime_config import RuntimeConfig
from workflows.utils import (
    get_default_workflow_root_log_dir,
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from workflows.workflow_config import (
    WORKFLOW_REPORT_CONFIG,
)

# from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import DeviceTypes, ModelType, ReportCheckTypes

logger = logging.getLogger(__name__)

# Media clients (audio, cnn) constants
FUNCTIONAL_TARGET = 10
COMPLETE_TARGET = 2
DATE_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"
NOT_MEASURED_STR = "n/a"


# Benchmark summary helpers owned by this workflow module.
def format_backend_value(backend: str) -> str:
    """Format backend value for display in summary tables."""
    if backend == "vllm":
        return "vLLM"
    if backend == "genai-perf":
        return "genai"
    return backend if backend else NOT_MEASURED_STR


def _read_optional_csv_rows(
    csv_path: Optional[Union[str, Path]], label: str
) -> Optional[List[Dict[str, str]]]:
    """Read optional CSV output when it exists."""
    if not csv_path:
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        if csv_path.parent.exists():
            logger.info(
                f"{label} CSV data is optional and was not found at {csv_path}. Continuing without it."
            )
        return None

    with csv_path.open("r", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _map_model_type_to_task_type(model_type: ModelType) -> Optional[str]:
    if model_type == ModelType.LLM:
        return "text"
    if model_type == ModelType.CNN:
        return "cnn"
    if model_type == ModelType.AUDIO:
        return "audio"
    if model_type == ModelType.IMAGE:
        return "image"
    if model_type == ModelType.VLM:
        return "vlm"
    if model_type == ModelType.EMBEDDING:
        return "embedding"
    if model_type == ModelType.VIDEO:
        return "video"
    if model_type == ModelType.TEXT_TO_SPEECH:
        return "text_to_speech"
    return None


def _get_task_type(model_id: str) -> str:
    model_name = model_id.lower().split("_")[-1]
    for model_spec in MODEL_SPECS.values():
        if model_name in model_spec.model_name.lower() and model_spec.model_type:
            task_type = _map_model_type_to_task_type(model_spec.model_type)
            if task_type:
                return task_type
    return "unknown"


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    aiperf_image_pattern = r"""
        ^aiperf_benchmark_
        (?P<model>.+?)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|n150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        _images-(?P<images_per_prompt>\d+)
        _height-(?P<image_height>\d+)
        _width-(?P<image_width>\d+)
        \.json$
    """

    match = re.search(aiperf_image_pattern, filename, re.VERBOSE)
    if match:
        model_name = match.group("model")
        is_image_generation = any(
            img_gen in model_name.lower()
            for img_gen in ["stable-diffusion", "sdxl", "sd-", "sd3"]
        )
        task_type = "image" if is_image_generation else "vlm"
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
            "task_type": task_type,
            "backend": "aiperf",
        }

    aiperf_text_pattern = r"""
        ^aiperf_benchmark_
        (?P<model>.+?)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|n150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        \.json$
    """

    match = re.search(aiperf_text_pattern, filename, re.VERBOSE)
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

    image_pattern = r"""
        ^(?:genai_)?benchmark_
        (?P<model>.+?)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|TG|GALAXY|n150|n300|p100|p150|galaxy_t3k|t3k|tg|galaxy))?
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        _images-(?P<images_per_prompt>\d+)
        _height-(?P<image_height>\d+)
        _width-(?P<image_width>\d+)
        \.json$
    """

    match = re.search(image_pattern, filename, re.VERBOSE)
    if match:
        logger.info(f"Found image benchmark pattern in filename: {filename}")
        model_name = match.group("model")
        is_image_generation = any(
            img_gen in model_name.lower()
            for img_gen in ["stable-diffusion", "sdxl", "sd-", "sd3"]
        )
        task_type = "image" if is_image_generation else "vlm"
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
            "task_type": task_type,
        }

    text_pattern = r"""
        ^(?:genai_)?benchmark_
        (?P<model>.+?)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|n150x4|TG|GALAXY|n150|n300|p100|p150|galaxy_t3k|t3k|tg|galaxy))?
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        \.json$
    """
    match = re.search(text_pattern, filename, re.VERBOSE)
    if match:
        logger.info(f"Found text benchmark pattern in filename: {filename}")
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

    cnn_pattern = r"""
        ^benchmark_
        (?P<model_id>id_.+?)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|p150x8|p300x2|P300x2|p300|P300|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?
        _(?P<timestamp>\d+\.?\d*)
        \.json$
    """

    match = re.search(cnn_pattern, filename, re.VERBOSE)
    if match:
        logger.info(
            f"Found id/device/timestamp benchmark pattern in filename: {filename}"
        )
        model_id = match.group("model_id")
        return {
            "model_id": model_id,
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "task_type": _get_task_type(model_id),
        }

    raise ValueError(f"Could not extract parameters from filename: {filename}")


def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    formatted_metrics = {}
    sig_digits_map = {
        "mean_ttft_ms": 1,
        "mean_tpot_ms": 1,
        "mean_tps": 2,
        "mean_e2el_ms": 1,
        "tps_decode_throughput": 1,
        "tps_prefill_throughput": 1,
        "request_throughput": 3,
        "total_token_throughput": 2,
    }

    for key, value in metrics.items():
        if value is None or value == NOT_MEASURED_STR:
            formatted_metrics[key] = NOT_MEASURED_STR
        elif isinstance(value, float):
            formatted_metrics[key] = round(float(value), sig_digits_map.get(key, 2))
        else:
            formatted_metrics[key] = value

    return formatted_metrics


def _resolve_optional_benchmark_concurrency(
    benchmark_values: Dict[str, Any], fallback: Any = None
) -> Any:
    if not isinstance(benchmark_values, dict):
        return fallback

    concurrency = benchmark_values.get(
        "concurrency", benchmark_values.get("max_concurrency")
    )
    if concurrency not in (None, "", 0, NOT_MEASURED_STR):
        return concurrency

    if fallback not in (None, "", 0, NOT_MEASURED_STR):
        return fallback

    if benchmark_values.get("num_requests") == 1:
        return 1

    return None


def process_benchmark_file(filepath: str) -> Dict[str, Any]:
    """Process a single benchmark file and extract relevant metrics."""
    logger.info(f"Processing benchmark file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    filename = os.path.basename(filepath)
    params = extract_params_from_filename(filename)

    if params.get("backend") == "aiperf":
        logger.info(f"Processing AIPerf benchmark file: {filepath}")
        mean_tpot_ms = data.get("mean_tpot_ms", 0)
        if mean_tpot_ms and mean_tpot_ms > 0:
            mean_tps = 1000.0 / mean_tpot_ms
            std_tps = None
            if data.get("std_tpot_ms"):
                std_tps = mean_tps - (1000.0 / (mean_tpot_ms + data.get("std_tpot_ms")))
        else:
            mean_tps = None
            std_tps = None

        actual_max_con = min(params["max_con"], params["num_requests"])
        tps_decode_throughput = mean_tps * actual_max_con if mean_tps else None
        tps_prefill_throughput = None
        if data.get("mean_ttft_ms") and data.get("mean_ttft_ms") > 0:
            tps_prefill_throughput = (
                params["input_sequence_length"] * actual_max_con
            ) / (data.get("mean_ttft_ms") / 1000)

        metrics = {
            "timestamp": params["timestamp"],
            "model_name": params["model_name"],
            "model_id": data.get("model_id", ""),
            "backend": "aiperf",
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
            "tps_decode_throughput": tps_decode_throughput,
            "tps_prefill_throughput": tps_prefill_throughput,
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

    benchmarks_data = data.get("benchmarks: ", data)
    if benchmarks_data and benchmarks_data.get("benchmarks"):
        logger.info(f"Processing CNN/SDXL-style benchmark file: {filename}")
        benchmark_values = benchmarks_data.get("benchmarks", {})
        if params.get("task_type") == "cnn":
            logger.info(f"Processing CNN benchmark file: {filename}")
            num_requests = benchmark_values.get("num_requests", 0)
            max_concurrency = _resolve_optional_benchmark_concurrency(benchmark_values)
            metrics = {
                "timestamp": params["timestamp"],
                "model": data.get("model", ""),
                "model_name": data.get("model", ""),
                "model_id": data.get("model", ""),
                "backend": "cnn",
                "device": params["device"],
                "num_requests": num_requests,
                "max_con": max_concurrency,
                "max_concurrency": max_concurrency,
                "num_inference_steps": benchmark_values.get("num_inference_steps", 0),
                "mean_ttft_ms": benchmark_values.get("ttft", 0) * 1000,
                "inference_steps_per_second": benchmark_values.get(
                    "inference_steps_per_second", 0
                ),
                "tput_user": benchmark_values.get("inference_steps_per_second", 0),
                "tput": benchmark_values.get("inference_steps_per_second", 0),
                "end_to_end_latency_ms": benchmark_values.get("end_to_end_latency_ms"),
                "e2el_ms": benchmark_values.get(
                    "e2el_ms", benchmark_values.get("end_to_end_latency_ms")
                ),
                "filename": filename,
                "task_type": "cnn",
            }
            return format_metrics(metrics)
        if params.get("task_type") == "image":
            logger.info(f"Processing IMAGE benchmark file: {filename}")
            num_requests = benchmark_values.get("num_requests", 0)
            max_concurrency = _resolve_optional_benchmark_concurrency(
                benchmark_values,
                fallback=min(params.get("max_con", 0), num_requests)
                if params.get("max_con")
                else None,
            )
            metrics = {
                "timestamp": params["timestamp"],
                "model": data.get("model", ""),
                "model_name": data.get("model", ""),
                "model_id": data.get("model", ""),
                "backend": "image",
                "device": params["device"],
                "num_requests": num_requests,
                "max_con": max_concurrency,
                "max_concurrency": max_concurrency,
                "num_inference_steps": benchmark_values.get("num_inference_steps", 0),
                "mean_ttft_ms": benchmark_values.get("ttft", 0) * 1000,
                "inference_steps_per_second": benchmark_values.get(
                    "inference_steps_per_second", 0
                ),
                "tput_user": benchmark_values.get("inference_steps_per_second", 0),
                "tput": benchmark_values.get("inference_steps_per_second", 0),
                "end_to_end_latency_ms": benchmark_values.get("end_to_end_latency_ms"),
                "e2el_ms": benchmark_values.get(
                    "e2el_ms", benchmark_values.get("end_to_end_latency_ms")
                ),
                "filename": filename,
                "task_type": "image",
            }
            return format_metrics(metrics)

    if params.get("task_type") in ("text_to_speech", "tts"):
        logger.info(f"Processing TTS benchmark file: {filename}")
        benchmarks_data = data.get("benchmarks", {})
        metrics = {
            "timestamp": params["timestamp"],
            "model": data.get("model", ""),
            "model_name": data.get("model", ""),
            "model_id": data.get("model", ""),
            "backend": "text_to_speech",
            "device": params["device"],
            "num_requests": benchmarks_data.get("num_requests", 0),
            "mean_ttft_ms": benchmarks_data.get("ttft", 0) * 1000,
            "filename": filename,
            "task_type": "tts",
            "rtr": benchmarks_data.get("rtr", 0),
            "p90_ttft": benchmarks_data.get("ttft_p90", 0) * 1000
            if benchmarks_data.get("ttft_p90")
            else None,
            "p95_ttft": benchmarks_data.get("ttft_p95", 0) * 1000
            if benchmarks_data.get("ttft_p95")
            else None,
            "wer": benchmarks_data.get("wer", None),
        }
        return format_metrics(metrics)

    if params.get("task_type") == "audio":
        logger.info(f"Processing AUDIO benchmark file: {filename}")
        benchmarks_data = data.get("benchmarks: ", data)
        benchmark_values = benchmarks_data.get("benchmarks", {})
        num_requests = benchmark_values.get("num_requests", 0)
        max_concurrency = _resolve_optional_benchmark_concurrency(benchmark_values)
        metrics = {
            "timestamp": params["timestamp"],
            "model": data.get("model", ""),
            "model_name": data.get("model", ""),
            "model_id": data.get("model", ""),
            "backend": "audio",
            "device": params["device"],
            "num_requests": num_requests,
            "num_eval_runs": num_requests,
            "max_con": max_concurrency,
            "max_concurrency": max_concurrency,
            "mean_ttft_ms": benchmark_values.get("ttft", 0) * 1000,
            "filename": filename,
            "task_type": "audio",
            "accuracy_check": benchmark_values.get("accuracy_check", 0),
            "t/s/u": benchmark_values.get("t/s/u", 0),
            "tput_user": benchmark_values.get("t/s/u", 0),
            "ttft_streaming_ms": benchmark_values.get("ttft_streaming_ms"),
            "rtr": benchmark_values.get("rtr", 0),
            "streaming_enabled": data.get("streaming_enabled", False),
            "preprocessing_enabled": data.get("preprocessing_enabled", False),
        }
        return format_metrics(metrics)

    if params.get("task_type") == "embedding":
        benchmarks_data = data.get("benchmarks: ", data)
        benchmark_values = benchmarks_data.get("benchmarks", {})
        max_concurrency = _resolve_optional_benchmark_concurrency(benchmark_values)
        metrics = {
            "timestamp": params["timestamp"],
            "model": data.get("model", ""),
            "model_name": data.get("model", ""),
            "model_id": data.get("model", ""),
            "backend": "embedding",
            "device": params["device"],
            "filename": filename,
            "task_type": "embedding",
            "num_requests": benchmark_values.get("num_requests", 0),
            "isl": benchmark_values.get("isl", 0),
            "input_sequence_length": benchmark_values.get("isl", 0),
            "osl": NOT_MEASURED_STR,
            "output_sequence_length": NOT_MEASURED_STR,
            "max_con": max_concurrency,
            "max_concurrency": max_concurrency,
            "embedding_dimension": benchmark_values.get(
                "embedding_dimension", NOT_MEASURED_STR
            ),
            "mean_ttft_ms": NOT_MEASURED_STR,
            "mean_tpot_ms": NOT_MEASURED_STR,
            "mean_tps": benchmark_values.get("tput_user", 0.0),
            "tput_user": benchmark_values.get("tput_user", 0.0),
            "tps_decode_throughput": NOT_MEASURED_STR,
            "tps_prefill_throughput": benchmark_values.get("tput_prefill", 0.0),
            "tput_prefill": benchmark_values.get("tput_prefill", 0.0),
            "mean_e2el_ms": benchmark_values.get("e2el", 0.0),
            "e2el_ms": benchmark_values.get("e2el", 0.0),
            "request_throughput": benchmark_values.get("req_tput", 0.0),
        }
        return format_metrics(metrics)

    if params.get("task_type") == "video":
        logger.info(f"Processing VIDEO benchmark file: {filename}")
        benchmarks_data = data.get("benchmarks: ", data)
        benchmark_values = benchmarks_data.get("benchmarks", {})
        num_requests = benchmark_values.get("num_requests", 0)
        max_concurrency = _resolve_optional_benchmark_concurrency(benchmark_values)
        metrics = {
            "timestamp": params["timestamp"],
            "model": data.get("model", ""),
            "model_name": data.get("model", ""),
            "model_id": data.get("model", ""),
            "backend": "video",
            "device": params["device"],
            "filename": filename,
            "task_type": "video",
            "num_requests": num_requests,
            "max_con": max_concurrency,
            "max_concurrency": max_concurrency,
            "mean_ttft_ms": benchmark_values.get("ttft", 0) * 1000,
            "inference_steps_per_second": benchmark_values.get(
                "inference_steps_per_second", 0
            ),
            "tput_user": benchmark_values.get("inference_steps_per_second", 0),
            "tput": benchmark_values.get("inference_steps_per_second", 0),
            "num_inference_steps": benchmark_values.get("num_inference_steps", 0),
            "end_to_end_latency_ms": benchmark_values.get("end_to_end_latency_ms"),
            "e2el_ms": benchmark_values.get(
                "e2el_ms", benchmark_values.get("end_to_end_latency_ms")
            ),
        }
        return format_metrics(metrics)

    logger.info(
        f"Default benchmark file processing (task_type={params.get('task_type')})"
    )
    mean_tpot_ms = data.get("mean_tpot_ms")
    if data.get("mean_tpot_ms"):
        mean_tpot = max(data.get("mean_tpot_ms"), 1e-6)
        mean_tps = 1000.0 / mean_tpot
        if data.get("std_tpot_ms"):
            std_tps = mean_tps - (1000.0 / (mean_tpot + data.get("std_tpot_ms")))
        else:
            std_tps = None
    else:
        mean_tps = None
        std_tps = None
    actual_max_con = min(params["max_con"], params["num_requests"])
    tps_decode_throughput = mean_tps * actual_max_con if mean_tps else None
    tps_prefill_throughput = (params["input_sequence_length"] * actual_max_con) / (
        data.get("mean_ttft_ms") / 1000
    )

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
        "tps_decode_throughput": tps_decode_throughput,
        "tps_prefill_throughput": tps_prefill_throughput,
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
        metrics.update(
            {
                "images_per_prompt": params["images_per_prompt"],
                "image_height": params["image_height"],
                "image_width": params["image_width"],
            }
        )

    return format_metrics(metrics)


def process_benchmark_files(files: List[str], pattern: str) -> List[Dict[str, Any]]:
    """Process benchmark files from multiple files matching the given pattern."""
    results = []

    logger.info(f"Processing {len(files)} files")
    for filepath in files:
        logger.info(f"Processing: {filepath} ...")
        try:
            metrics = process_benchmark_file(filepath)
            results.append(metrics)
        except Exception as error:
            logger.exception(f"Error processing file {filepath}: {error}")

    if not results:
        raise ValueError("No benchmark files were successfully processed")

    return sorted(results, key=lambda value: value["timestamp"])


def save_to_csv(results: List[Dict[str, Any]], file_path: Union[Path, str]) -> None:
    if not results:
        return

    headers = list(results[0].keys())
    try:
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for result in results:
                row = [str(result.get(header, NOT_MEASURED_STR)) for header in headers]
                writer.writerow(row)
        print(f"\nResults saved to: {file_path}")
    except Exception as error:
        print(f"Error saving CSV file: {error}")


def create_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("max_con", "Concurrency"),
        ("num_requests", "N Req"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "Interactivity (tok/s/user)"),
        ("tps_decode_throughput", "Output Tput (tok/s)"),
        ("tps_prefill_throughput", "Input Tput (tok/s)"),
        ("total_token_throughput", "Total Tput (tok/s)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        value = result.get(col_name, NOT_MEASURED_STR)
        if col_name == "backend":
            value = format_backend_value(value)
        display_dict[display_header] = str(value)

    return display_dict


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
        ("mean_tps", "Interactivity (tok/s/user)"),
        ("tps_decode_throughput", "Output Tput (tok/s)"),
        ("tps_prefill_throughput", "Input Tput (tok/s)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        if col_name == "isl":
            value = result.get(
                "isl", result.get("input_sequence_length", NOT_MEASURED_STR)
            )
        elif col_name == "osl":
            value = result.get(
                "osl", result.get("output_sequence_length", NOT_MEASURED_STR)
            )
        else:
            value = result.get(col_name, NOT_MEASURED_STR)
        if col_name == "backend":
            value = format_backend_value(value)
        display_dict[display_header] = str(value)

    return display_dict


def create_audio_display_dict(result: Dict[str, Any], model_spec) -> Dict[str, str]:
    """Create display dictionary for audio benchmarks."""
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

    class ModelSpecWrapper:
        def __init__(self, wrapped_model_spec):
            self.model_spec = wrapped_model_spec

    wrapper = ModelSpecWrapper(model_spec)
    whisper_config_values = {
        "streaming_enabled": str(is_streaming_enabled_for_whisper(wrapper)),
        "preprocessing_enabled": str(is_preprocessing_enabled_for_whisper(wrapper)),
    }

    display_dict = {}
    for col_name, display_header in display_cols:
        if col_name in whisper_config_values:
            display_dict[display_header] = whisper_config_values[col_name]
            continue
        value = result.get(col_name, NOT_MEASURED_STR)
        if col_name == "backend":
            value = format_backend_value(value)
        display_dict[display_header] = str(value)

    return display_dict


def create_tts_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    """Create display dictionary for TTS benchmarks."""
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("rtr", "RTR"),
        ("p90_ttft", "P90 TTFT (ms)"),
        ("p95_ttft", "P95 TTFT (ms)"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        value = result.get(col_name, NOT_MEASURED_STR)
        if col_name == "backend":
            value = format_backend_value(value)
        display_dict[display_header] = str(value)

    return display_dict


def create_embedding_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("max_con", "Max Concurrency"),
        ("embedding_dimension", "Embedding Dimension"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "Interactivity (tok/s/user)"),
        ("tps_decode_throughput", "Output Tput (tok/s)"),
        ("tps_prefill_throughput", "Input Tput (tok/s)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        display_dict[display_header] = str(result.get(col_name, NOT_MEASURED_STR))

    return display_dict


def create_image_generation_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    """Create display dictionary for image generation benchmarks."""
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("num_inference_steps", "Inference Steps"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("inference_steps_per_second", "Steps/Sec"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        value = result.get(col_name, NOT_MEASURED_STR)
        if col_name == "backend":
            value = format_backend_value(value)
        display_dict[display_header] = str(value)

    return display_dict


def create_cnn_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("num_inference_steps", "Num Inference Steps"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("task_type", "Task Type"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        display_dict[display_header] = str(result.get(col_name, NOT_MEASURED_STR))

    return display_dict


def create_video_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    display_cols: List[Tuple[str, str]] = [
        ("backend", "Source"),
        ("num_requests", "Num Requests"),
        ("num_inference_steps", "Num Inference Steps"),
        ("mean_ttft_ms", "TTFT (ms)"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        display_dict[display_header] = str(result.get(col_name, NOT_MEASURED_STR))

    return display_dict


def sanitize_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _cell_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    return 1


def wcswidth(text: str) -> int:
    """Return the number of monospace columns text will occupy."""
    return sum(_cell_width(ch) for ch in text)


def pad_right(text: str, width: int) -> str:
    return text + " " * max(width - wcswidth(text), 0)


def pad_left(text: str, width: int) -> str:
    return " " * max(width - wcswidth(text), 0) + text


def pad_center(text: str, width: int) -> str:
    total = width - wcswidth(text)
    left = total // 2
    return " " * max(left, 0) + text + " " * max(total - left, 0)


def get_markdown_table(display_dicts: List[Dict[str, str]]) -> str:
    if not display_dicts:
        return ""

    headers = list(display_dicts[0].keys())
    numeric_cols = {
        header: all(
            re.match(r"^-?\d+(\.\d+)?$", str(display_dict.get(header, "")).strip())
            for display_dict in display_dicts
        )
        for header in headers
    }

    max_left = {}
    max_right = {}
    for header in headers:
        max_left[header] = 0
        max_right[header] = 0
        if numeric_cols[header]:
            for display_dict in display_dicts:
                value = str(display_dict.get(header, "")).strip()
                left, _, right = value.partition(".")
                max_left[header] = max(max_left[header], len(left))
                max_right[header] = max(max_right[header], len(right))

    def format_numeric(value: str, header: str) -> str:
        left, _, right = value.partition(".")
        left = left.rjust(max_left[header])
        if max_right[header] > 0:
            right = right.ljust(max_right[header])
            return f"{left}.{right}"
        return left

    col_widths = {}
    for header in headers:
        if numeric_cols[header]:
            numeric_width = (
                max_left[header]
                + (1 if max_right[header] > 0 else 0)
                + max_right[header]
            )
            col_widths[header] = max(wcswidth(header), numeric_width)
        else:
            max_content = max(
                wcswidth(sanitize_cell(str(display_dict.get(header, ""))))
                for display_dict in display_dicts
            )
            col_widths[header] = max(wcswidth(header), max_content)

    header_row = (
        "| "
        + " | ".join(
            pad_center(sanitize_cell(header), col_widths[header]) for header in headers
        )
        + " |"
    )
    separator_row = (
        "|" + "|".join("-" * (col_widths[header] + 2) for header in headers) + "|"
    )

    value_rows = []
    for display_dict in display_dicts:
        cells = []
        for header in headers:
            raw_value = sanitize_cell(str(display_dict.get(header, "")).strip())
            if numeric_cols[header]:
                numeric_value = format_numeric(raw_value, header)
                cell = pad_left(numeric_value, col_widths[header])
            else:
                cell = pad_right(raw_value, col_widths[header])
            cells.append(cell)
        value_rows.append("| " + " | ".join(cells) + " |")

    end_notes = "\n\nNote: all metrics are means across benchmark run unless otherwise stated.\n"

    def clean_header(header: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", header).strip()

    def describe_headers_from_keys(keys: List[str]) -> str:
        explanation_map = {
            "ISL": "Input Sequence Length (tokens)",
            "OSL": "Output Sequence Length (tokens)",
            "Concurrency": "number of concurrent requests (batch size)",
            "N Req": "total number of requests (sample size, N)",
            "TTFT": "Time To First Token (ms)",
            "TPOT": "Time Per Output Token (ms)",
            "Interactivity": "Output token throughput per user (tok/s/user)",
            "Output Tput": "Output token (decode) throughput, across all users (tok/s)",
            "Input Tput": "Input token (prefill) throughput (tok/s)",
            "E2EL": "End-to-End Latency (ms)",
            "Req Tput": "Request Throughput (RPS)",
        }
        return "\n".join(
            f"> {key}: {explanation_map[key]}" for key in keys if key in explanation_map
        )

    key_list = [clean_header(key) for key in headers]
    explain_str = describe_headers_from_keys(key_list)
    return "\n".join([header_row, separator_row] + value_rows) + end_notes + explain_str


def save_markdown_table(
    markdown_str: str,
    filepath: Union[Path, str],
    add_title: Optional[str] = None,
    add_notes: Optional[List[str]] = None,
) -> None:
    path = Path(filepath)
    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")

    path.parent.mkdir(parents=True, exist_ok=True)

    content = []
    if add_title:
        content.extend([f"# {add_title}", ""])
    content.append(markdown_str)
    if add_notes:
        content.extend(add_notes)

    try:
        path.write_text("\n".join(content), encoding="utf-8")
        print(f"Successfully saved markdown table to: {path}")
    except Exception as error:
        print(f"Error saving markdown table: {str(error)}")


def generate_benchmark_summary_report(
    files, output_dir, report_id, metadata=None, model_spec=None
):
    metadata = metadata or {}
    assert len(files) > 0, "No benchmark files found."
    results = process_benchmark_files(files, pattern="benchmark_*.json")

    def get_sort_key(result):
        isl = result.get("input_sequence_length", 0)
        osl = result.get("output_sequence_length", 0)
        concurrency = result.get("max_con", 1)
        backend = result.get("backend", "")
        source_priority = {
            "vllm": 0,
            "openai-chat": 0,
            "aiperf": 1,
            "genai-perf": 2,
        }.get(backend, 3)
        images = result.get("images_per_prompt", 0)
        height = result.get("image_height", 0)
        width = result.get("image_width", 0)
        return (isl, osl, concurrency, images, height, width, source_priority)

    results.sort(key=get_sort_key)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = metadata["model_name"]
    device = results[0].get("device")
    if "device" in metadata:
        assert metadata["device"] == device, "Device mismatch in metadata"

    data_file_path = output_dir / "data" / f"benchmark_stats_{report_id}.csv"
    data_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(results, data_file_path)

    text_results = [result for result in results if result.get("task_type") == "text"]
    vlm_results = [result for result in results if result.get("task_type") == "vlm"]
    image_results = [result for result in results if result.get("task_type") == "image"]
    audio_results = [result for result in results if result.get("task_type") == "audio"]
    tts_results = [result for result in results if result.get("task_type") == "tts"]
    embedding_results = [
        result for result in results if result.get("task_type") == "embedding"
    ]
    cnn_results = [result for result in results if result.get("task_type") == "cnn"]
    video_results = [result for result in results if result.get("task_type") == "video"]

    markdown_sections = []
    if text_results:
        text_display_results = [create_display_dict(result) for result in text_results]
        text_markdown_str = get_markdown_table(text_display_results)
        markdown_sections.append(
            f"#### Text-to-Text Performance Benchmark Sweeps for {model_name} on {device}\n\n{text_markdown_str}"
        )

    if vlm_results:
        vlm_display_results = [
            create_vlm_display_dict(result) for result in vlm_results
        ]
        vlm_markdown_str = get_markdown_table(vlm_display_results)
        markdown_sections.append(
            f"#### VLM Benchmark Sweeps for {model_name} on {device}\n\n{vlm_markdown_str}"
        )

    if image_results:
        image_display_results = [
            create_image_generation_display_dict(result) for result in image_results
        ]
        image_markdown_str = get_markdown_table(image_display_results)
        markdown_sections.append(
            f"#### Image Generation Benchmark Sweeps for {model_name} on {device}\n\n{image_markdown_str}"
        )

    if audio_results:
        audio_display_results = [
            create_audio_display_dict(result, model_spec) for result in audio_results
        ]
        audio_markdown_str = get_markdown_table(audio_display_results)
        markdown_sections.append(
            f"#### Audio Benchmark Sweeps for {model_name} on {device}\n\n{audio_markdown_str}"
        )

    if tts_results:
        tts_display_results = [
            create_tts_display_dict(result) for result in tts_results
        ]
        tts_markdown_str = get_markdown_table(tts_display_results)
        markdown_sections.append(
            f"#### Text-to-Speech Benchmark Sweeps for {model_name} on {device}\n\n{tts_markdown_str}"
        )

    if embedding_results:
        embedding_display_results = [
            create_embedding_display_dict(result) for result in embedding_results
        ]
        embedding_markdown_str = get_markdown_table(embedding_display_results)
        markdown_sections.append(
            f"#### Embedding Benchmark Sweeps for {model_name} on {device}\n\n{embedding_markdown_str}"
        )

    if cnn_results:
        cnn_display_results = [
            create_cnn_display_dict(result) for result in cnn_results
        ]
        cnn_markdown_str = get_markdown_table(cnn_display_results)
        markdown_sections.append(
            f"#### CNN Benchmark Sweeps for {model_name} on {device}\n\n{cnn_markdown_str}"
        )

    if video_results:
        video_display_results = [
            create_video_display_dict(result) for result in video_results
        ]
        video_markdown_str = get_markdown_table(video_display_results)
        markdown_sections.append(
            f"#### Video Benchmark Sweeps for {model_name} on {device}\n\n{video_markdown_str}"
        )

    if markdown_sections:
        display_md_str = (
            f"### Performance Benchmark Sweeps for {model_name} on {device}\n\n"
            + "\n\n".join(markdown_sections)
        )
    else:
        display_results = [create_display_dict(result) for result in results]
        markdown_str = get_markdown_table(display_results)
        display_md_str = (
            f"### Performance Benchmark Sweeps for {model_name} on {device}\n\n"
            f"{markdown_str}"
        )

    disp_md_path = Path(output_dir) / f"benchmark_display_{report_id}.md"
    save_markdown_table(display_md_str, disp_md_path)
    return display_md_str, results, disp_md_path, data_file_path


def generate_embedding_report_data(model_spec, eval_run_id):
    """Generate embedding-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for embedding evaluation results
    """
    # Embedding models use results_*.json pattern
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_audio_report_data(model_spec, eval_run_id):
    """Generate audio-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for audio evaluation results
    """
    # Audio models use *_results.json pattern (created by lmms-eval)
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/*_results.json"
    return file_name_pattern


def generate_cnn_report_data(model_spec, eval_run_id):
    """Generate CNN-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for CNN evaluation results
    """
    # CNN models use results_*.json pattern
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_video_report_data(model_spec, eval_run_id):
    """Generate video-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for CNN evaluation results
    """
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_image_generation_report_data(model_spec, eval_run_id):
    """Generate image-generation-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for image generation evaluation results
    """
    # Image generation models use results_*.json pattern
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_tts_report_data(model_spec, eval_run_id):
    """Generate TTS-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for TTS evaluation results
    """
    # TTS models use results_*.json pattern (same as image/cnn)
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def get_embedding_benchmark_targets(model_spec, device_str, logger):
    """Get embedding-specific benchmark targets.

    Args:
        model_spec: Model specification
        device_str: Device string
        logger: Logger instance

    Returns:
        Benchmark target data for embedding models
    """
    device_json_list = get_perf_target_rows(model_spec.model_name, device_str)

    if not device_json_list:
        logger.warning(
            f"No performance targets found for embedding model {model_spec.model_name} on {device_str}"
        )

    return device_json_list


def get_audio_benchmark_targets(model_spec, device_str, logger):
    """Get audio-specific benchmark targets.

    Args:
        model_spec: Model specification
        device_str: Device string
        logger: Logger instance

    Returns:
        Benchmark target data for audio models
    """
    device_json_list = get_perf_target_rows(model_spec.model_name, device_str)

    if not device_json_list:
        logger.warning(
            f"No performance targets found for audio model {model_spec.model_name} on {device_str}"
        )

    return device_json_list


def get_cnn_benchmark_targets(model_spec, device_str, logger):
    """Get CNN-specific benchmark targets.

    Args:
        model_spec: Model specification
        device_str: Device string
        logger: Logger instance

    Returns:
        Benchmark target data for CNN models
    """
    device_json_list = get_perf_target_rows(model_spec.model_name, device_str)

    if not device_json_list:
        logger.warning(
            f"No performance targets found for CNN model {model_spec.model_name} on {device_str}"
        )

    return device_json_list


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM reports")
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on",
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for report output",
        required=True,
    )
    ret_args = parser.parse_args()
    return ret_args


def flatten_target_checks(rows):
    flat_rows = []
    for row in rows:
        # Start with all the top-level keys except "target_checks"
        flat = {k: v for k, v in row.items() if k != "target_checks"}
        # For each target (e.g. "reference", "other"), and each metric inside it,
        # create a new key "<target>_<metric>"
        for target_name, checks in row.get("target_checks", {}).items():
            for metric, value in checks.items():
                flat[f"{target_name}_{metric}"] = value
        flat_rows.append(flat)
    return flat_rows


def benchmark_release_markdown(release_raw, target_checks=None):
    # Define display columns mapping
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Interactivity (tok/s/user)"),
        ("tput", "Output Tput (tok/s)"),
    ]
    check_cols = []
    if target_checks:
        # NOTE: set column order via tuple
        check_cols = [
            (
                f"{k}_{metric}",
                " ".join(
                    w.upper() if w.lower() == "ttft" else w.capitalize()
                    for w in f"{k}_{metric}".split("_")
                )
                + (
                    ""  # no unit for any "_check" column
                    if metric.endswith("_check") or metric.endswith("_ratio")
                    else " (ms)"  # TTFT always in milliseconds
                    if metric.startswith("ttft")
                    else " (tok/s)"  # any Tput* in transactions/second
                    if metric.startswith("tput")
                    else ""
                ),
            )
            for k in target_checks.keys()
            # NOTE: comment out columns to hide them from display
            for metric in (
                "ttft_check",
                "tput_user_check",
                # "tput_check",
                "ttft",
                # "ttft_ratio",
                "tput_user",
                # "tput_user_ratio",
                # "tput",
                # "tput_ratio",
            )
        ]
        check_cols.sort(key=lambda col: not col[0].endswith("_check"))

    display_cols += check_cols
    NOT_MEASURED_STR = "N/A"
    cols_to_round = [_col[0] for _col in check_cols]
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if isinstance(value, ReportCheckTypes):
                row_dict[display_header] = ReportCheckTypes.to_display_string(value)
            elif col_name in cols_to_round and isinstance(value, float):
                row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def benchmark_vlm_release_markdown(release_raw, target_checks=None):
    """Build markdown table for VLM benchmark results (isl, osl, image dimensions, etc.)."""
    # Display columns for VLM benchmarks only
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Interactivity (tok/s/user)"),
        ("tput", "Output Tput (tok/s)"),
    ]
    check_cols = []
    if target_checks:
        # NOTE: set column order via tuple
        check_cols = [
            (
                f"{k}_{metric}",
                " ".join(
                    w.upper() if w.lower() == "ttft" else w.capitalize()
                    for w in f"{k}_{metric}".split("_")
                )
                + (
                    ""  # no unit for any "_check" column
                    if metric.endswith("_check") or metric.endswith("_ratio")
                    else " (ms)"  # TTFT always in milliseconds
                    if metric.startswith("ttft")
                    else " (tok/s)"  # any Tput* in transactions/second
                    if metric.startswith("tput")
                    else ""
                ),
            )
            for k in target_checks.keys()
            # NOTE: comment out columns to hide them from display
            for metric in (
                "ttft_check",
                "tput_user_check",
                # "tput_check",
                "ttft",
                # "ttft_ratio",
                "tput_user",
                # "tput_user_ratio",
                # "tput",
                # "tput_ratio",
            )
        ]
        check_cols.sort(key=lambda col: not col[0].endswith("_check"))

    display_cols += check_cols
    NOT_MEASURED_STR = "N/A"
    cols_to_round = [_col[0] for _col in check_cols]
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if isinstance(value, ReportCheckTypes):
                row_dict[display_header] = ReportCheckTypes.to_display_string(value)
            elif col_name in cols_to_round and isinstance(value, float):
                row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_release_markdown(release_raw, is_vlm_benchmark=False):
    """Generate markdown table for AIPerf benchmarks with detailed metrics.

    This follows NVIDIA's genai-perf style output with mean, median, and p99 percentiles
    for each key metric category.

    Args:
        release_raw: Raw benchmark data
        is_vlm_benchmark: If True, table is for VLM results (includes image dimension columns).
    """
    # Define display columns mapping - NVIDIA style with detailed percentiles
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
    ]

    # Add image-dimension columns for VLM benchmarks only
    if is_vlm_benchmark:
        display_cols.extend(
            [
                ("image_height", "Image Height"),
                ("image_width", "Image Width"),
                ("images_per_prompt", "Images per Prompt"),
            ]
        )

    display_cols.extend(
        [
            ("num_requests", "N"),
            # TTFT metrics
            ("mean_ttft_ms", "TTFT Avg (ms)"),
            ("median_ttft_ms", "TTFT P50 (ms)"),
            ("p99_ttft_ms", "TTFT P99 (ms)"),
            # TPOT metrics (Time Per Output Token)
            ("mean_tpot_ms", "TPOT Avg (ms)"),
            ("median_tpot_ms", "TPOT P50 (ms)"),
            ("p99_tpot_ms", "TPOT P99 (ms)"),
            # E2EL metrics (End-to-End Latency)
            ("mean_e2el_ms", "E2EL Avg (ms)"),
            ("median_e2el_ms", "E2EL P50 (ms)"),
            ("p99_e2el_ms", "E2EL P99 (ms)"),
            # Throughput
            ("output_token_throughput", "Output Tok/s"),
            ("total_token_throughput", "Total Tok/s"),
            ("request_throughput", "Req/s"),
        ]
    )

    NOT_MEASURED_STR = "N/A"
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                # Format floats with appropriate precision
                if col_name in ("request_throughput",):
                    row_dict[display_header] = f"{value:.4f}"
                elif col_name in ("output_token_throughput", "total_token_throughput"):
                    row_dict[display_header] = f"{value:.2f}"
                else:
                    row_dict[display_header] = f"{value:.1f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_throughput_markdown(release_raw):
    """Generate markdown table for benchmarks with derived throughput metrics.

    This follows the genai-perf comparison style with Interactivity, Output Tput, and Input Tput
    columns for easy comparison between vLLM, AIPerf, and genai-perf benchmarks.
    """
    # Define display columns - genai-perf comparison style with Source column
    display_cols = [
        ("source", "Source"),
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
        ("num_requests", "N"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("tput_user", "User Tput (tok/s)"),
        ("tput_decode", "Output Tput (tok/s)"),
        ("tput_prefill", "Input Tput (tok/s)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    NOT_MEASURED_STR = "N/A"
    display_dicts = []
    for row in release_raw:
        # Calculate derived throughput metrics
        tpot = row.get("mean_tpot_ms", 0)
        ttft = row.get("mean_ttft_ms", 0)
        isl = row.get("isl", 0)
        concurrency = row.get("concurrency", 1)

        tput_user = 1000.0 / tpot if tpot > 0 else 0
        tput_decode = tput_user * concurrency
        tput_prefill = (isl * concurrency) / (ttft / 1000.0) if ttft > 0 else 0

        # Add derived metrics to row
        row_with_derived = dict(row)
        row_with_derived["tput_user"] = tput_user
        row_with_derived["tput_decode"] = tput_decode
        row_with_derived["tput_prefill"] = tput_prefill

        row_dict = {}
        for col_name, display_header in display_cols:
            value = row_with_derived.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                # Format floats with appropriate precision
                if col_name == "request_throughput":
                    row_dict[display_header] = f"{value:.3f}"
                elif col_name in ("tput_user", "tput_decode", "tput_prefill"):
                    row_dict[display_header] = f"{value:.1f}"
                elif col_name in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms"):
                    row_dict[display_header] = f"{value:.1f}"
                else:
                    row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_throughput_markdown_with_images(release_raw):
    """Generate markdown table for image benchmarks with image parameters.
    Similar to aiperf_throughput_markdown but includes image dimensions.
    """
    # Define display columns for image benchmarks
    display_cols = [
        ("source", "Source"),
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
        ("num_requests", "N"),
        ("images", "Images"),
        ("image_width", "Width"),
        ("image_height", "Height"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("tput_user", "Interactivity (tok/s/user)"),
        ("tput_decode", "Output Tput (tok/s)"),
        ("tput_prefill", "Input Tput (tok/s)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    NOT_MEASURED_STR = "N/A"
    display_dicts = []
    for row in release_raw:
        # Calculate derived throughput metrics
        tpot = row.get("mean_tpot_ms", 0)
        ttft = row.get("mean_ttft_ms", 0)
        isl = row.get("isl", 0)
        concurrency = row.get("concurrency", 1)

        tput_user = 1000.0 / tpot if tpot > 0 else 0
        tput_decode = tput_user * concurrency
        tput_prefill = (isl * concurrency) / (ttft / 1000.0) if ttft > 0 else 0

        # Add derived metrics to row
        row_with_derived = dict(row)
        row_with_derived["tput_user"] = tput_user
        row_with_derived["tput_decode"] = tput_decode
        row_with_derived["tput_prefill"] = tput_prefill

        row_dict = {}
        for col_name, display_header in display_cols:
            value = row_with_derived.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                # Format floats with appropriate precision
                if col_name == "request_throughput":
                    row_dict[display_header] = f"{value:.3f}"
                elif col_name in ("tput_user", "tput_decode", "tput_prefill"):
                    row_dict[display_header] = f"{value:.1f}"
                elif col_name in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms"):
                    row_dict[display_header] = f"{value:.1f}"
                else:
                    row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_benchmark_generate_report(
    args, server_mode, model_spec, report_id, metadata={}
):
    """Generate benchmark report specifically for AIPerf results.

    AIPerf provides more detailed metrics than vLLM's benchmark_serving.py,
    including mean, median, and p99 percentiles for TTFT, TPOT, and E2EL.
    This function creates a separate report in NVIDIA's genai-perf style.
    Table 2 (Comparison) combines both vLLM and AIPerf results for easy comparison.
    """
    # All benchmark tools now use the same output directory
    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"

    # Look for aiperf benchmark files
    aiperf_pattern = f"aiperf_benchmark_{model_spec.model_id}_*.json"
    aiperf_files = glob(f"{benchmarks_output_dir}/{aiperf_pattern}")

    # Also look for vLLM benchmark files for comparison table
    vllm_pattern = f"benchmark_{model_spec.model_id}_*.json"
    vllm_files = glob(f"{benchmarks_output_dir}/{vllm_pattern}")

    # Look for GenAI-Perf benchmark files
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")

    output_dir = Path(args.output_path) / "benchmarks_aiperf"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("AIPerf Benchmark Summary")
    logger.info(f"Found {len(aiperf_files)} AIPerf benchmark files")
    logger.info(f"Found {len(vllm_files)} vLLM benchmark files for comparison")
    logger.info(f"Found {len(genai_files)} GenAI-Perf benchmark files for comparison")

    if not aiperf_files and not vllm_files and not genai_files:
        logger.info("No benchmark files found. Skipping AIPerf report.")
        return "", [], None, None

    # Helper function to keep only the latest file for each (isl, osl, concurrency, task_type) config
    def deduplicate_by_config(files):
        """Keep only the latest file for each unique benchmark configuration.

        Files are sorted by name (which includes timestamp) in reverse order,
        so we keep the first occurrence of each config.

        Config key includes:
        - isl, osl, concurrency, num_requests (base params)
        - images, height, width (for image benchmarks - treated as separate configs)
        """
        config_to_file = {}
        # Sort in reverse order so latest files come first
        for filepath in sorted(files, reverse=True):
            filename = Path(filepath).name
            # Extract base config from filename
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, con, n = map(int, match.groups())

                # Check if this is an image benchmark (has images-X in filename)
                img_match = re.search(
                    r"images-(\d+)_height-(\d+)_width-(\d+)", filename
                )
                if img_match:
                    images, height, width = map(int, img_match.groups())
                    config_key = (isl, osl, con, n, images, height, width)
                else:
                    # Text-only benchmark
                    config_key = (isl, osl, con, n, 0, 0, 0)

                # Only keep the first (latest) file for each config
                if config_key not in config_to_file:
                    config_to_file[config_key] = filepath
            else:
                # If no match, include the file anyway
                config_to_file[filepath] = filepath
        return list(config_to_file.values())

    # Deduplicate files to keep only latest run for each config
    vllm_files = deduplicate_by_config(vllm_files)
    aiperf_files = deduplicate_by_config(aiperf_files)
    genai_files = deduplicate_by_config(genai_files)

    logger.info(
        f"After deduplication: {len(vllm_files)} vLLM, {len(aiperf_files)} AIPerf, {len(genai_files)} GenAI-Perf files"
    )

    # Separate text-only and VLM benchmarks
    vllm_text_only_files = [f for f in vllm_files if "images" not in Path(f).name]
    vllm_vlm_files = [f for f in vllm_files if "images" in Path(f).name]
    aiperf_text_only_files = [f for f in aiperf_files if "images" not in Path(f).name]
    aiperf_vlm_files = [f for f in aiperf_files if "images" in Path(f).name]
    genai_text_only_files = [f for f in genai_files if "images" not in Path(f).name]
    genai_vlm_files = [f for f in genai_files if "images" in Path(f).name]

    logger.info(
        f"Text benchmarks: {len(vllm_text_only_files)} vLLM, {len(aiperf_text_only_files)} AIPerf, {len(genai_text_only_files)} GenAI-Perf"
    )
    logger.info(
        f"VLM benchmarks: {len(vllm_vlm_files)} vLLM, {len(aiperf_vlm_files)} AIPerf, {len(genai_vlm_files)} GenAI-Perf"
    )

    # Process text-only vLLM benchmarks
    vllm_text_results = []
    for filepath in sorted(vllm_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            # Pattern: benchmark_*_isl-{isl}_osl-{osl}_maxcon-{con}_n-{n}*.json
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "vLLM",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "vllm",
            }
            vllm_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing vLLM file {filepath}: {e}")
            continue

    # Process text-only AIPerf files
    aiperf_text_results = []
    for filepath in sorted(aiperf_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            # Pattern: aiperf_benchmark_*_isl-{isl}_osl-{osl}_maxcon-{con}_n-{n}.json
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "aiperf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "aiperf",
            }
            aiperf_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing AIPerf file {filepath}: {e}")
            continue

    # Process VLM vLLM benchmarks
    vllm_vlm_results = []
    for filepath in sorted(vllm_vlm_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            # Calculate throughput metrics
            mean_tpot_ms = data.get("mean_tpot_ms", 0)
            if mean_tpot_ms and mean_tpot_ms > 0:
                mean_tps = 1000.0 / mean_tpot_ms
            else:
                mean_tps = 0

            actual_max_con = min(concurrency, num_requests)
            tps_decode_throughput = mean_tps * actual_max_con if mean_tps else 0

            result = {
                "source": "vLLM",
                "task_type": "vlm",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "max_con": concurrency,
                "num_requests": num_requests,
                "images": images,
                "image_height": height,
                "image_width": width,
                "images_per_prompt": images,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput (calculated)
                "mean_tps": mean_tps,
                "tps_decode_throughput": tps_decode_throughput,
                "output_token_throughput": data.get("output_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": data.get("backend", "vllm"),
            }
            vllm_vlm_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing vLLM VLM file {filepath}: {e}")
            continue

    # Process VLM AIPerf files
    aiperf_vlm_results = []
    for filepath in sorted(aiperf_vlm_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "aiperf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images_per_prompt": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "aiperf",
            }
            aiperf_vlm_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing AIPerf VLM file {filepath}: {e}")
            continue

    # Process GenAI-Perf text files
    genai_text_results = []
    for filepath in sorted(genai_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            # Pattern: genai_benchmark_*_isl-{isl}_osl-{osl}_maxcon-{con}_n-{n}.json
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf file {filepath}: {e}")
            continue

    # Process GenAI-Perf VLM files
    genai_vlm_results = []
    for filepath in sorted(genai_vlm_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_vlm_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf VLM file {filepath}: {e}")
            continue

    if (
        not aiperf_text_results
        and not vllm_text_results
        and not genai_text_results
        and not aiperf_vlm_results
        and not vllm_vlm_results
        and not genai_vlm_results
    ):
        return "", [], None, None

    # Sort text benchmarks by ISL, OSL, concurrency
    vllm_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))
    aiperf_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))
    genai_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))

    # Sort VLM benchmarks by ISL, OSL, concurrency, image size
    vllm_vlm_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["image_height"],
            x["image_width"],
        )
    )
    aiperf_vlm_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["image_height"],
            x["image_width"],
        )
    )
    genai_vlm_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["image_height"],
            x["image_width"],
        )
    )

    # Build the complete report
    release_str = ""

    # Only include section if there are results to display
    if aiperf_text_results or aiperf_vlm_results:
        release_str = f"### Benchmark Performance Results for {model_spec.model_name} on {args.device}\n\n"

        # TEXT BENCHMARKS SECTION
        if aiperf_text_results:
            release_str += "#### AIPerf Text Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf)\n\n"

            # Only show AIPerf-specific detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(aiperf_text_results)
            release_str += nvidia_markdown_str
            release_str += "\n\n"

        # VLM BENCHMARKS SECTION
        if aiperf_vlm_results:
            release_str += "#### AIPerf VLM Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf)\n\n"

            # Only show AIPerf-specific detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(
                aiperf_vlm_results, is_vlm_benchmark=True
            )
            release_str += nvidia_markdown_str
            release_str += "\n\n"

        # Metric definitions
        release_str += "**Metric Definitions:**\n"
        release_str += "> - **ISL**: Input Sequence Length (tokens)\n"
        release_str += "> - **OSL**: Output Sequence Length (tokens)\n"
        release_str += "> - **Concur**: Concurrent requests (batch size)\n"
        release_str += "> - **N**: Total number of requests\n"
        release_str += "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)\n"
        release_str += "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **Output Tok/s**: Output token throughput\n"
        release_str += (
            "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
        )
        release_str += "> - **Req/s**: Request throughput\n"

    # Save markdown report
    disp_md_path = output_dir / f"aiperf_benchmark_display_{report_id}.md"
    with open(disp_md_path, "w", encoding="utf-8") as f:
        f.write(release_str)
    logger.info(f"AIPerf report saved to: {disp_md_path}")

    # Save CSV data for text benchmarks
    text_data_file_path = (
        output_dir / "data" / f"aiperf_benchmark_text_stats_{report_id}.csv"
    )
    text_data_file_path.parent.mkdir(parents=True, exist_ok=True)

    if aiperf_text_results:
        headers = list(aiperf_text_results[0].keys())
        with open(text_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in aiperf_text_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"AIPerf text benchmark data saved to: {text_data_file_path}")

    # Save CSV data for VLM benchmarks
    image_data_file_path = (
        output_dir / "data" / f"aiperf_benchmark_vlm_stats_{report_id}.csv"
    )
    if aiperf_vlm_results:
        headers = list(aiperf_vlm_results[0].keys())
        with open(image_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in aiperf_vlm_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"AIPerf VLM benchmark data saved to: {image_data_file_path}")

    # Return combined results for both text and VLM
    all_aiperf_results = aiperf_text_results + aiperf_vlm_results
    return release_str, all_aiperf_results, disp_md_path, text_data_file_path


def genai_perf_benchmark_generate_report(
    args, server_mode, model_spec, report_id, metadata={}
):
    """Generate benchmark report specifically for GenAI-Perf results.

    GenAI-Perf provides detailed metrics similar to AIPerf,
    including mean, median, and p99 percentiles for TTFT, TPOT, and E2EL.
    This function creates a separate detailed report following the same format as AIPerf.
    """
    # All benchmark tools now use the same output directory
    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"

    # Look for genai-perf benchmark files
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")

    output_dir = Path(args.output_path) / "benchmarks_genai_perf"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("GenAI-Perf Benchmark Summary")
    logger.info(f"Found {len(genai_files)} GenAI-Perf benchmark files")

    if not genai_files:
        logger.info("No GenAI-Perf benchmark files found. Skipping GenAI-Perf report.")
        return "", [], None, None

    # Helper function to keep only the latest file for each config
    def deduplicate_by_config(files):
        """Keep only the latest file for each unique benchmark configuration."""
        config_to_file = {}
        # Sort in reverse order so latest files come first
        for filepath in sorted(files, reverse=True):
            filename = Path(filepath).name
            # Extract base config from filename
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, maxcon, n = match.groups()
                # For image benchmarks, also include image dimensions
                image_match = re.search(
                    r"images-(\d+)_height-(\d+)_width-(\d+)", filename
                )
                if image_match:
                    images, height, width = image_match.groups()
                    config_key = (isl, osl, maxcon, n, images, height, width)
                else:
                    config_key = (isl, osl, maxcon, n)

                # Only keep the first (latest) file for each config
                if config_key not in config_to_file:
                    config_to_file[config_key] = filepath
            else:
                # If no match, include the file anyway
                config_to_file[filepath] = filepath
        return list(config_to_file.values())

    genai_files = deduplicate_by_config(genai_files)
    logger.info(f"After deduplication: {len(genai_files)} GenAI-Perf benchmark files")

    # Separate text-only and VLM benchmarks
    genai_text_only_files = [f for f in genai_files if "images" not in Path(f).name]
    genai_vlm_files = [f for f in genai_files if "images" in Path(f).name]

    logger.info(
        f"GenAI-Perf Text benchmarks: {len(genai_text_only_files)}, VLM benchmarks: {len(genai_vlm_files)}"
    )

    # Process text-only GenAI-Perf benchmarks
    genai_text_results = []
    for filepath in sorted(genai_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf text file {filepath}: {e}")
            continue

    # Process VLM GenAI-Perf files
    genai_vlm_results = []
    for filepath in sorted(genai_vlm_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images_per_prompt": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_vlm_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf VLM file {filepath}: {e}")
            continue

    if not genai_text_results and not genai_vlm_results:
        logger.info("No GenAI-Perf results to process.")
        return "", [], None, None

    # Sort text benchmarks by ISL, OSL, concurrency
    genai_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))

    # Sort VLM benchmarks by ISL, OSL, concurrency, image dimensions
    genai_vlm_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["images_per_prompt"],
            x["image_height"],
            x["image_width"],
        )
    )

    # Build the complete report
    release_str = ""

    # Only include section if there are results to display
    if genai_text_results or genai_vlm_results:
        release_str = f"### GenAI-Perf Benchmark Performance Results for {model_spec.model_name} on {args.device}\n\n"

        # TEXT BENCHMARKS SECTION
        if genai_text_results:
            release_str += "#### GenAI-Perf Text Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer)\n\n"

            # Show GenAI-Perf detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(genai_text_results)
            release_str += nvidia_markdown_str
            release_str += "\n*Note: GenAI-Perf does not natively support total token throughput metrics.*\n\n"

        # VLM BENCHMARKS SECTION
        if genai_vlm_results:
            release_str += "#### GenAI-Perf VLM Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer)\n\n"

            # Show GenAI-Perf detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(
                genai_vlm_results, is_vlm_benchmark=True
            )
            release_str += nvidia_markdown_str
            release_str += "\n*Note: GenAI-Perf does not natively support total token throughput metrics.*\n\n"

        # Metric definitions
        release_str += "**Metric Definitions:**\n"
        release_str += "> - **ISL**: Input Sequence Length (tokens)\n"
        release_str += "> - **OSL**: Output Sequence Length (tokens)\n"
        release_str += "> - **Concur**: Concurrent requests (batch size)\n"
        release_str += "> - **N**: Total number of requests\n"
        release_str += "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)\n"
        release_str += "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **Output Tok/s**: Output token throughput\n"
        release_str += (
            "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
        )
        release_str += "> - **Req/s**: Request throughput\n"

    # Save markdown report
    disp_md_path = output_dir / f"genai_perf_benchmark_display_{report_id}.md"
    with open(disp_md_path, "w", encoding="utf-8") as f:
        f.write(release_str)
    logger.info(f"GenAI-Perf report saved to: {disp_md_path}")

    # Save CSV data for text benchmarks
    text_data_file_path = (
        output_dir / "data" / f"genai_perf_benchmark_text_stats_{report_id}.csv"
    )
    text_data_file_path.parent.mkdir(parents=True, exist_ok=True)

    if genai_text_results:
        headers = list(genai_text_results[0].keys())
        with open(text_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in genai_text_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"GenAI-Perf text benchmark data saved to: {text_data_file_path}")

    # Save CSV data for VLM benchmarks
    image_data_file_path = (
        output_dir / "data" / f"genai_perf_benchmark_vlm_stats_{report_id}.csv"
    )
    if genai_vlm_results:
        headers = list(genai_vlm_results[0].keys())
        with open(image_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in genai_vlm_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"GenAI-Perf VLM benchmark data saved to: {image_data_file_path}")

    # Return combined results for both text and VLM
    all_genai_results = genai_text_results + genai_vlm_results
    return release_str, all_genai_results, disp_md_path, text_data_file_path


def benchmark_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    # Look for vLLM, genai-perf, and AIPerf benchmark files (all stack together)
    # All benchmark tools now use the same unified output directory
    vllm_pattern = f"benchmark_{model_spec.model_id}_*.json"
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"
    aiperf_pattern = f"aiperf_benchmark_{model_spec.model_id}_*.json"

    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"

    vllm_files = glob(f"{benchmarks_output_dir}/{vllm_pattern}")
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")
    aiperf_files = glob(f"{benchmarks_output_dir}/{aiperf_pattern}")

    logger.info(
        f"Found {len(vllm_files)} vLLM, {len(genai_files)} genai-perf, and {len(aiperf_files)} AIPerf benchmark files before deduplication"
    )

    # Deduplicate files - keep only latest run for each config
    def deduplicate_by_config(files):
        """Keep only the latest file for each unique benchmark configuration."""
        config_to_file = {}
        # Sort in reverse order so latest files come first
        for filepath in sorted(files, reverse=True):
            filename = Path(filepath).name
            # Extract base config from filename
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, con, n = map(int, match.groups())

                # Check if this is an image benchmark (has images-X in filename)
                img_match = re.search(
                    r"images-(\d+)_height-(\d+)_width-(\d+)", filename
                )
                if img_match:
                    images, height, width = map(int, img_match.groups())
                    config_key = (isl, osl, con, n, images, height, width)
                else:
                    # Text-only benchmark
                    config_key = (isl, osl, con, n, 0, 0, 0)

                # Only keep the first (latest) file for each config
                if config_key not in config_to_file:
                    config_to_file[config_key] = filepath
            else:
                # If no match, include the file anyway
                config_to_file[filepath] = filepath
        return list(config_to_file.values())

    vllm_files = deduplicate_by_config(vllm_files)
    genai_files = deduplicate_by_config(genai_files)
    aiperf_files = deduplicate_by_config(aiperf_files)

    logger.info(
        f"After deduplication: {len(vllm_files)} vLLM, {len(genai_files)} genai-perf, {len(aiperf_files)} AIPerf benchmark files"
    )
    output_dir = Path(args.output_path) / "benchmarks"

    if not vllm_files and not genai_files and not aiperf_files:
        logger.info("No benchmark files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            None,
            None,
        )

    # Process each tool separately to generate individual tables
    # Order: vLLM -> AIPerf -> GenAI-Perf (for both text and image)
    all_tool_results = []

    # Process all tools and collect results by type (text/image/audio/tts/embedding/cnn)
    text_sections = []
    image_sections = []
    audio_sections = []
    tts_sections = []
    embedding_sections = []
    cnn_sections = []
    video_sections = []

    # Process vLLM benchmarks
    if vllm_files:
        _, vllm_release_raw, _, _ = generate_benchmark_summary_report(
            vllm_files, output_dir, report_id, metadata, model_spec=model_spec
        )
        all_tool_results.extend(vllm_release_raw)

        # Separate text, vlm, audio, tts, embedding and cnn for vLLM
        vllm_text = [r for r in vllm_release_raw if r.get("task_type") == "text"]
        vllm_vlm = [r for r in vllm_release_raw if r.get("task_type") == "vlm"]
        vllm_audio = [r for r in vllm_release_raw if r.get("task_type") == "audio"]
        vllm_tts = [r for r in vllm_release_raw if r.get("task_type") == "tts"]
        vllm_embedding = [
            r for r in vllm_release_raw if r.get("task_type") == "embedding"
        ]
        vllm_cnn = [r for r in vllm_release_raw if r.get("task_type") == "cnn"]
        vllm_image = [r for r in vllm_release_raw if r.get("task_type") == "image"]
        vllm_video = [r for r in vllm_release_raw if r.get("task_type") == "video"]

        if vllm_text:
            vllm_text_display = [create_display_dict(r) for r in vllm_text]
            vllm_text_md = get_markdown_table(vllm_text_display)
            text_sections.append(
                f"#### vLLM Text-to-Text Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_text_md}"
            )

        if vllm_vlm:
            vllm_vlm_display = [create_vlm_display_dict(r) for r in vllm_vlm]
            vllm_vlm_md = get_markdown_table(vllm_vlm_display)
            image_sections.append(
                f"#### vLLM Vision-Language Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_vlm_md}"
            )

        if vllm_audio:
            vllm_audio_display = [
                create_audio_display_dict(r, model_spec) for r in vllm_audio
            ]
            vllm_audio_md = get_markdown_table(vllm_audio_display)
            audio_sections.append(
                f"#### vLLM Audio Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_audio_md}"
            )

        if vllm_tts:
            vllm_tts_display = [create_tts_display_dict(r) for r in vllm_tts]
            vllm_tts_md = get_markdown_table(vllm_tts_display)
            tts_sections.append(
                f"#### vLLM Text-to-Speech Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_tts_md}"
            )

        if vllm_embedding:
            vllm_embedding_display = [
                create_embedding_display_dict(r) for r in vllm_embedding
            ]
            vllm_embedding_md = get_markdown_table(vllm_embedding_display)
            embedding_sections.append(
                f"#### vLLM Embedding Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_embedding_md}"
            )

        if vllm_cnn:
            vllm_cnn_display = [
                create_image_generation_display_dict(r) for r in vllm_cnn
            ]
            vllm_cnn_md = get_markdown_table(vllm_cnn_display)
            cnn_sections.append(
                f"#### CNN Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_cnn_md}"
            )

        if vllm_image:
            vllm_image_display = [
                create_image_generation_display_dict(r) for r in vllm_image
            ]
            vllm_image_md = get_markdown_table(vllm_image_display)
            image_sections.append(
                f"#### vLLM Image Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_image_md}"
            )

        if vllm_video:
            vllm_video_display = [create_video_display_dict(r) for r in vllm_video]
            vllm_video_md = get_markdown_table(vllm_video_display)
            video_sections.append(
                f"#### vLLM Video Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_video_md}"
            )

    # Process AIPerf benchmarks
    if aiperf_files:
        _, aiperf_release_raw, _, _ = generate_benchmark_summary_report(
            aiperf_files, output_dir, report_id, metadata, model_spec=model_spec
        )
        all_tool_results.extend(aiperf_release_raw)

        # Separate text and vlm for AIPerf
        aiperf_text = [r for r in aiperf_release_raw if r.get("task_type") == "text"]
        aiperf_vlm = [r for r in aiperf_release_raw if r.get("task_type") == "vlm"]

        if aiperf_text:
            aiperf_text_display = [create_display_dict(r) for r in aiperf_text]
            aiperf_text_md = get_markdown_table(aiperf_text_display)
            text_sections.append(
                f"#### AIPerf Text-to-Text Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_text_md}"
            )

        if aiperf_vlm:
            aiperf_vlm_display = [create_vlm_display_dict(r) for r in aiperf_vlm]
            aiperf_vlm_md = get_markdown_table(aiperf_vlm_display)
            image_sections.append(
                f"#### AIPerf VLM Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_vlm_md}"
            )

    # Process GenAI-Perf benchmarks
    if genai_files:
        _, genai_release_raw, _, _ = generate_benchmark_summary_report(
            genai_files, output_dir, report_id, metadata, model_spec=model_spec
        )
        all_tool_results.extend(genai_release_raw)

        # Separate text and vlm for GenAI-Perf
        genai_text = [r for r in genai_release_raw if r.get("task_type") == "text"]
        genai_vlm = [r for r in genai_release_raw if r.get("task_type") == "vlm"]

        if genai_text:
            genai_text_display = [create_display_dict(r) for r in genai_text]
            genai_text_md = get_markdown_table(genai_text_display)
            text_sections.append(
                f"#### GenAI-Perf Text-to-Text Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_text_md}"
            )

        if genai_vlm:
            genai_vlm_display = [create_vlm_display_dict(r) for r in genai_vlm]
            genai_vlm_md = get_markdown_table(genai_vlm_display)
            image_sections.append(
                f"#### GenAI-Perf VLM Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_vlm_md}"
            )

    # Combine sections: text, image, audio, embedding, then cnn (matching original order)
    markdown_sections = (
        text_sections
        + image_sections
        + audio_sections
        + tts_sections
        + embedding_sections
        + cnn_sections
        + video_sections
    )

    # Combine all sections
    release_str = ""
    if markdown_sections:
        release_str = (
            f"### Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n"
            + "\n\n".join(markdown_sections)
        )

    # Save combined CSV for all tools
    stats_file_path = output_dir / "data" / f"benchmark_stats_{report_id}.csv"
    stats_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(all_tool_results, stats_file_path)

    # Save display markdown
    disp_md_path = output_dir / f"benchmark_display_{report_id}.md"
    save_markdown_table(release_str, disp_md_path)

    release_raw = all_tool_results
    return release_str, release_raw, disp_md_path, stats_file_path


def extract_eval_json_data(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    configs = data.get("configs", {})

    first_key = list(results.keys())[0]

    # extract first results' metrics
    first_results = results[first_key]
    extracted_metrics = {
        k: v
        for k, v in first_results.items()
        if "alias" not in k and "_stderr" not in k
    }
    extracted = [{first_key: extracted_metrics}]

    config = configs.get(first_key, {})
    task_name = config.get("task", first_key)

    # assert that all configs have the same dataset path
    dataset_path = list(configs.values())[0]["dataset_path"]  # first_dataset_path
    for config in configs.values():
        config_dataset_path = config.get("dataset_path")
        assert dataset_path == config_dataset_path

    assert task_name == first_key, f"Task name mismatch: {task_name} != {first_key}"

    meta_data = {"task_name": task_name, "dataset_path": dataset_path}

    return extracted, meta_data


def extract_eval_results(files):
    files = sorted(files, key=lambda f: Path(f).stat().st_mtime, reverse=True)
    results = {}
    meta_data = {}
    for json_file in files:
        res, meta = extract_eval_json_data(Path(json_file))
        _ = meta.pop("task_name", None)
        for task_dict in res:
            for specific_task_name, metrics in task_dict.items():
                if specific_task_name not in results:
                    results[specific_task_name] = metrics
                    meta_data[specific_task_name] = meta

    return results, meta_data


def evals_release_report_data(args, results, meta_data, model_spec):
    eval_config = EVAL_CONFIGS[model_spec.model_name]

    report_rows = []

    for task in eval_config.tasks:
        if not task.score:
            logger.info(
                f"Skipping report for task:= {task.task_name}, no eval score is defined."
            )
            continue

        target_keys = []
        # Check for exact match (e.g. "meta_gpqa")
        if task.task_name in results:
            target_keys.append(task.task_name)
        else:
            # Check for subtasks (e.g. config says "longbench", results have "longbench_2wikimqa")
            prefix = f"{task.task_name}_"
            subtasks = [k for k in results if k.startswith(prefix)]
            target_keys.extend(sorted(subtasks))
        if target_keys:
            for t_key in target_keys:
                logger.info(f"eval processing task_name: {t_key}")

                # do NOT extract results[t_key] here.
                # The score_func expects the ROOT results dict so it can do results[task_name].

                kwargs = task.score.score_func_kwargs
                # Update task_name so the score function looks up the specific subtask (e.g. longbench_2wikimqa)
                kwargs["task_name"] = t_key
                configured_keys = kwargs.get("result_keys", [])
                actual_data = results.get(t_key, {})

                key_found = any(k in actual_data for k in configured_keys)

                if not key_found:
                    valid_candidates = [
                        k
                        for k, v in actual_data.items()
                        if isinstance(v, (int, float))
                        and "stderr" not in k
                        and "alias" not in k
                    ]

                    if valid_candidates:
                        logger.info(
                            f"  Metric mismatch for {t_key}. Auto-detected replacement: {valid_candidates[0]}"
                        )
                        kwargs["result_keys"] = [valid_candidates[0]]
                try:
                    score = task.score.score_func(
                        results, task_name=t_key, kwargs=kwargs
                    )
                except Exception as e:
                    logger.warning(f"  Could not calculate score for {t_key}: {e}")
                    score = 0.0
                if kwargs.get("unit") == "WER":
                    score = 100 - score

                if task.score.published_score:
                    assert task.score.published_score > 0, "Published score is not > 0"
                    ratio_to_published = score / task.score.published_score
                else:
                    ratio_to_published = "N/A"

                if task.score.gpu_reference_score:
                    assert task.score.gpu_reference_score > 0, (
                        "Reference score is not > 0"
                    )
                    ratio_to_reference = score / task.score.gpu_reference_score
                    accuracy_check = ReportCheckTypes.from_result(
                        ratio_to_reference >= (1.0 - task.score.tolerance)
                    )
                else:
                    ratio_to_reference = "N/A"
                    if task.score.published_score:
                        accuracy_check = ReportCheckTypes.from_result(
                            ratio_to_published >= (1.0 - task.score.tolerance)
                        )
                    else:
                        accuracy_check = ReportCheckTypes.NA

                report_rows.append(
                    {
                        "model": model_spec.model_name,
                        "device": args.device,
                        "task_name": t_key,
                        "accuracy_check": accuracy_check,
                        "score": score,
                        "ratio_to_reference": ratio_to_reference,
                        "gpu_reference_score": task.score.gpu_reference_score,
                        "gpu_reference_score_ref": task.score.gpu_reference_score_ref,
                        "ratio_to_published": ratio_to_published,
                        "published_score": task.score.published_score,
                        "published_score_ref": task.score.published_score_ref,
                        "metadata": meta_data.get(t_key),
                    }
                )
        else:
            score = "N/A"
            ratio_to_published = "N/A"
            ratio_to_reference = "N/A"
            accuracy_check = ReportCheckTypes.NA

            report_rows.append(
                {
                    "model": model_spec.model_name,
                    "device": args.device,
                    "task_name": task.task_name,
                    "accuracy_check": accuracy_check,
                    "score": score,
                    "ratio_to_reference": ratio_to_reference,
                    "gpu_reference_score": task.score.gpu_reference_score,
                    "gpu_reference_score_ref": task.score.gpu_reference_score_ref,
                    "ratio_to_published": ratio_to_published,
                    "published_score": task.score.published_score,
                    "published_score_ref": task.score.published_score_ref,
                    "metadata": meta_data.get(task.task_name),
                }
            )

    return report_rows


def generate_evals_release_markdown(report_rows):
    # Step 1: Convert all values to strings with proper formatting
    def format_value(key, value, row):
        if key == "published_score":
            # Format published_score as a hyperlink to published_score_ref
            score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            ref_val = row.get("published_score_ref", "")
            return f"[{score_val}]({ref_val})" if ref_val else score_val
        elif key == "gpu_reference_score":
            # Format gpu_reference_score as a hyperlink to gpu_reference_score_ref
            score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            ref_val = row.get("gpu_reference_score_ref", "")
            return f"[{score_val}]({ref_val})" if ref_val else score_val
        elif key == "accuracy_check":
            return ReportCheckTypes.to_display_string(value)
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    formatted_rows = [
        {k: format_value(k, v, row) for k, v in row.items()} for row in report_rows
    ]

    # Remove published_score_ref column from display
    remove_keys = ["published_score_ref", "metadata", "gpu_reference_score_ref"]
    headers = [h for h in formatted_rows[0].keys() if h not in remove_keys]

    # Step 2: Compute max width per column
    column_widths = {
        header: max(len(header), max(len(row[header]) for row in formatted_rows))
        for header in headers
    }

    # Step 3: Build table rows
    def format_row(row):
        return (
            "| " + " | ".join(f"{row[h]:<{column_widths[h]}}" for h in headers) + " |"
        )

    # Step 4: Build header and divider rows
    header_row = "| " + " | ".join(f"{h:<{column_widths[h]}}" for h in headers) + " |"
    divider_row = "|-" + "-|-".join("-" * column_widths[h] for h in headers) + "-|"

    row_strs = [format_row(row) for row in formatted_rows]

    explain_str = "\n\nNote: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score."

    markdown_str = (
        header_row + "\n" + divider_row + "\n" + "\n".join(row_strs) + explain_str
    )
    return markdown_str


def separate_files_by_format(files):
    """Separate eval files into dict-format and list-format.

    Detects JSON structure to differentiate between:
    - Dict format: {"results": {...}, "configs": {...}} (lmms-eval)
    - List format: [{...}] (image_client)

    Args:
        files: List of file paths to eval JSON files

    Returns:
        Tuple of (dict_format_files, list_format_files)
    """
    dict_format_files = []
    list_format_files = []

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                list_format_files.append(filepath)
            elif isinstance(data, dict):
                dict_format_files.append(filepath)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read or parse file {filepath}: {e}")

    return dict_format_files, list_format_files


def process_list_format_eval_files(list_files):
    """Process list-format JSON files from image_client.

    Extracts metrics from CNN image generation eval results.
    List format is: [{metric1: value1, metric2: value2, ...}]

    Args:
        list_files: List of file paths with list-format JSON

    Returns:
        Tuple of (results_dict, meta_data_dict) in the same format as extract_eval_results()
    """
    list_files = sorted(list_files, key=lambda f: Path(f).stat().st_mtime, reverse=True)
    results = {}
    meta_data = {}

    for filepath in list_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # data is a list of dicts, typically with one element from image_client
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"List format file {filepath} is empty or invalid")
                continue

            # Extract the first dict from the list (image_client typically writes one)
            eval_data = data[0]

            # Extract task name if available
            task_name = eval_data.get("task_name", "image_generation")

            if task_name in results:
                continue

            results[task_name] = eval_data

            meta_data[task_name] = {
                "task_name": task_name,
                "dataset_path": eval_data.get("dataset_path", "N/A"),
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not process list format file {filepath}: {e}")

    return results, meta_data


def evals_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    eval_run_id = f"{model_spec.model_id}"
    output_dir = Path(args.output_path) / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get file pattern based on model type
    if model_spec.model_type == ModelType.AUDIO:
        file_name_pattern = generate_audio_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.CNN:
        file_name_pattern = generate_cnn_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.IMAGE:
        file_name_pattern = generate_image_generation_report_data(
            model_spec, eval_run_id
        )
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.EMBEDDING:
        file_name_pattern = generate_embedding_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
        file_name_pattern = generate_tts_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.VIDEO:
        file_name_pattern = generate_video_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
        file_name_pattern = generate_tts_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    else:
        # LLM models use results_*.json pattern
        file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)

    if "image" in model_spec.supported_modalities:
        image_file_name_pattern = f"eval_{eval_run_id}/*_results.json"
        image_file_path_pattern = f"{get_default_workflow_root_log_dir()}/evals_output/{image_file_name_pattern}"
        image_files = glob(image_file_path_pattern)
        files.extend(image_files)
        image_file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/*results.json"
        image_file_path_pattern = f"{get_default_workflow_root_log_dir()}/evals_output/{image_file_name_pattern}"
        logger.info(f"Image File Pattern: {image_file_path_pattern}")
        image_files = glob(image_file_path_pattern)
        logger.info(f"Image Files: {image_files}")
        files.extend(image_files)
    files = list(dict.fromkeys(files))
    logger.info("Evaluations Summary")
    logger.info(f"Processing: {len(files)} files")
    if (
        model_spec.model_type.name == ModelType.CNN.name
        or model_spec.model_type.name == ModelType.IMAGE.name
        or model_spec.model_type.name == ModelType.EMBEDDING.name
        or model_spec.model_type.name == ModelType.VIDEO.name
        or model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name
    ):
        # TODO rewrite this
        data_fpath = data_dir / f"eval_data_{report_id}.json"

        # Combine files into one JSON
        combined_data = {}
        for i, file_path in enumerate(files):
            with open(file_path, "r") as f:
                file_data = json.load(f)
            combined_data = file_data

        # Write combined data to data_fpath
        with open(data_fpath, "w") as f:
            json.dump(combined_data, f, indent=4)

        release_str = (
            f"### Accuracy Evaluations for {model_spec.model_name} on {args.device}"
        )
        summary_fpath = output_dir / f"summary_{report_id}.md"
        with summary_fpath.open("w", encoding="utf-8") as f:
            f.write("MD summary to do")

        return release_str, combined_data, summary_fpath, data_fpath

    dict_format_files, list_format_files = separate_files_by_format(files)

    results = {}
    meta_data = {}

    if dict_format_files:
        dict_results, dict_meta_data = extract_eval_results(dict_format_files)
        results.update(dict_results)
        meta_data.update(dict_meta_data)
    if list_format_files:
        list_results, list_meta_data = process_list_format_eval_files(list_format_files)
        results.update(list_results)
        meta_data.update(list_meta_data)

    if not results:
        logger.warning("No evaluation files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            None,
            None,
        )
    # generate release report
    report_rows = evals_release_report_data(args, results, meta_data, model_spec)

    # store results
    markdown_str = generate_evals_release_markdown(report_rows)

    release_str = f"### Accuracy Evaluations for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    # generate summary report
    summary_fpath = output_dir / f"summary_{report_id}.md"
    summary_markdown_str = generate_evals_markdown_table(results, meta_data)
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(summary_markdown_str)

    # store raw data
    release_raw = report_rows
    data_fpath = data_dir / f"eval_data_{report_id}.json"

    with data_fpath.open("w", encoding="utf-8") as f:
        json.dump(release_raw, f, indent=4)

    disp_md_path = summary_fpath
    data_file_path = data_fpath
    return release_str, release_raw, disp_md_path, data_file_path


def generate_tests_report(args, server_mode, model_spec, report_id, metadata={}):
    # glob on all test reports - each test category might produce its own report
    file_name_pattern = f"test_{model_spec.model_id}_*/*"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/tests_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"summary_{report_id}.md"

    logger.info("Tests Summary")
    logger.info(f"Processing: {len(files)} files")
    if not files:
        logger.info("No tests report files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            None,
            None,
        )
    # When multiple test runs exist, use only the most recent result
    files = max(files, key=lambda f: Path(f).stat().st_mtime)
    logger.info(f"Selected most recent test report: {files}")

    # generate vLLM parameter coverage report
    markdown_str = generate_vllm_parameter_report(
        files, output_path, report_id, metadata, model_spec=model_spec
    )

    # Look for parameter_report.json in tests_output directory
    release_raw = None
    test_dir_pattern = f"test_{model_spec.model_id}_*"
    test_dir_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/tests_output/{test_dir_pattern}"
    )
    test_dirs = sorted(
        glob(test_dir_path_pattern),
        key=lambda d: Path(d).stat().st_mtime,
        reverse=True,
    )
    for test_dir in test_dirs:
        parameter_report_path = Path(test_dir) / "parameter_report.json"
        if parameter_report_path.exists():
            try:
                with open(parameter_report_path, "r", encoding="utf-8") as f:
                    release_raw = json.load(f)
                logger.info(f"Loaded parameter report from: {parameter_report_path}")
                break
            except Exception as e:
                logger.warning(
                    f"Could not read parameter report {parameter_report_path}: {e}"
                )

    if release_raw is None:
        logger.info("No parameter_report.json found in tests_output directory.")
        release_raw = [
            {
                "model": getattr(args, "model", "unknown_model"),
                "device": getattr(args, "device", "unknown_device"),
            }
        ]

    release_str = f"### Test Results for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    # Write markdown report to file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(release_str)
    logger.info(f"Tests report saved to: {output_path}")

    # Save raw data to data directory
    data_fpath = data_dir / f"tests_data_{report_id}.json"
    with data_fpath.open("w", encoding="utf-8") as f:
        json.dump(release_raw, f, indent=4)
    logger.info(f"Tests data saved to: {data_fpath}")

    return release_str, release_raw, output_path, data_fpath


def generate_evals_markdown_table(results, meta_data) -> str:
    rows = []
    for task_name, metrics in results.items():
        for metric_name, metric_value in metrics.items():
            if metric_name and metric_name != " ":
                if not isinstance(
                    metric_value, float
                ):  # some metrics in image evals are not floats
                    continue
                rows.append((task_name, metric_name, f"{metric_value:.4f}"))

    if not rows:
        return "No evaluation results to display."
    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = f"|{'-' * (col_widths[0] + 2)}|{'-' * (col_widths[1] + 2)}|{'-' * (col_widths[2] + 2)}|"
    markdown = header + "\n" + separator + "\n"

    for task_name, metric_name, metric_value in rows:
        markdown += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"

    return markdown


def generate_stress_tests_markdown_table(release_raw, model_config):
    """Generate markdown table for test results with mean values only (original format)."""

    # Define display columns: ISL, OSL, Concurrency, Num Prompts
    # Then mean values for TTFT, TPOT, ITL, E2EL
    # Then throughput metrics
    display_cols = [
        # Configuration columns
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("num_prompts", "Num Prompts"),
        # Mean metrics only (original format)
        ("ttft", "TTFT (ms)"),
        ("tpot", "TPOT (ms)"),
        ("itl", "ITL (ms)"),
        ("e2el", "E2EL (ms)"),
        # Throughput metrics at the end
        ("tput_user", "Interactivity (tok/s/user)"),
        ("tput", "Output Tput (tok/s)"),
    ]

    NOT_MEASURED_STR = "N/A"

    # Define decimal formatting standards
    decimal_places_map = {
        "ISL": 0,
        "OSL": 0,
        "Concurrency": 0,
        "Num Prompts": 0,
        "TTFT (ms)": 1,
        "TPOT (ms)": 1,
        "ITL (ms)": 1,
        "E2EL (ms)": 1,
        "Interactivity (tok/s/user)": 2,
        "Output Tput (tok/s)": 1,
    }

    display_dicts = []

    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            if col_name == "isl":
                value = row.get(
                    "input_sequence_length", row.get("isl", NOT_MEASURED_STR)
                )
            elif col_name == "osl":
                value = row.get(
                    "output_sequence_length", row.get("osl", NOT_MEASURED_STR)
                )
            elif col_name == "max_concurrency":
                value = row.get("max_con", NOT_MEASURED_STR)
            elif col_name == "num_prompts":
                value = row.get("num_prompts", NOT_MEASURED_STR)
            elif col_name == "ttft":
                value = row.get("mean_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "tpot":
                value = row.get("mean_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "itl":
                value = row.get("mean_itl_ms", NOT_MEASURED_STR)
            elif col_name == "e2el":
                value = row.get("mean_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "tput_user":
                value = row.get("mean_tps", NOT_MEASURED_STR)
            elif col_name == "tput":
                value = row.get("tps_decode_throughput", NOT_MEASURED_STR)
            else:
                value = row.get(col_name, NOT_MEASURED_STR)

            # Format numeric values with consistent decimal places for proper alignment
            if value == NOT_MEASURED_STR or value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, (int, float)) and not (
                isinstance(value, float) and (value != value)
            ):  # Check for NaN
                decimal_places = decimal_places_map.get(display_header, 2)
                if decimal_places == 0:
                    # Format as integer
                    row_dict[display_header] = str(int(value))
                else:
                    # Format as float with specified decimal places
                    row_dict[display_header] = f"{float(value):.{decimal_places}f}"
            else:
                # Handle string numbers or other formats
                try:
                    numeric_value = float(value)
                    decimal_places = decimal_places_map.get(display_header, 2)
                    if decimal_places == 0:
                        row_dict[display_header] = str(int(numeric_value))
                    else:
                        row_dict[display_header] = f"{numeric_value:.{decimal_places}f}"
                except (ValueError, TypeError):
                    row_dict[display_header] = str(value)

        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def generate_stress_tests_markdown_table_detailed(release_raw, model_config):
    """Generate detailed markdown table with percentile statistics for test results."""

    # Define display columns in requested order:
    # ISL, OSL, Concurrency, Num Prompts
    # Then for each metric (ttft, tpot, itl, e2el): mean, p05, p25, p50, p95, p99
    # Then throughput metrics
    display_cols = [
        # Configuration columns
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("num_prompts", "Num Prompts"),
        # TTFT metrics: mean, p05, p25, p50, p95, p99
        ("ttft", "TTFT (ms)"),
        ("p5_ttft", "P5 TTFT (ms)"),
        ("p25_ttft", "P25 TTFT (ms)"),
        ("p50_ttft", "P50 TTFT (ms)"),
        ("p95_ttft", "P95 TTFT (ms)"),
        ("p99_ttft", "P99 TTFT (ms)"),
        # TPOT metrics: mean, p05, p25, p50, p95, p99
        ("tpot", "TPOT (ms)"),
        ("p5_tpot", "P5 TPOT (ms)"),
        ("p25_tpot", "P25 TPOT (ms)"),
        ("p50_tpot", "P50 TPOT (ms)"),
        ("p95_tpot", "P95 TPOT (ms)"),
        ("p99_tpot", "P99 TPOT (ms)"),
        # ITL metrics: mean, p05, p25, p50, p95, p99
        ("itl", "ITL (ms)"),
        ("p5_itl", "P5 ITL (ms)"),
        ("p25_itl", "P25 ITL (ms)"),
        ("p50_itl", "P50 ITL (ms)"),
        ("p95_itl", "P95 ITL (ms)"),
        ("p99_itl", "P99 ITL (ms)"),
        # E2EL metrics: mean, p05, p25, p50, p95, p99
        ("e2el", "E2EL (ms)"),
        ("p5_e2el", "P5 E2EL (ms)"),
        ("p25_e2el", "P25 E2EL (ms)"),
        ("p50_e2el", "P50 E2EL (ms)"),
        ("p95_e2el", "P95 E2EL (ms)"),
        ("p99_e2el", "P99 E2EL (ms)"),
        # Throughput metrics at the end
        ("tput_user", "Interactivity (tok/s/user)"),
        ("tput", "Output Tput (tok/s)"),
    ]

    NOT_MEASURED_STR = "N/A"

    # Define decimal formatting standards based on benchmarking standards
    decimal_places_map = {
        "ISL": 0,
        "OSL": 0,
        "Concurrency": 0,
        "Num Prompts": 0,
        # TTFT
        "TTFT (ms)": 1,
        "P5 TTFT (ms)": 1,
        "P25 TTFT (ms)": 1,
        "P50 TTFT (ms)": 1,
        "P95 TTFT (ms)": 1,
        "P99 TTFT (ms)": 1,
        # TPOT
        "TPOT (ms)": 1,
        "P5 TPOT (ms)": 1,
        "P25 TPOT (ms)": 1,
        "P50 TPOT (ms)": 1,
        "P95 TPOT (ms)": 1,
        "P99 TPOT (ms)": 1,
        # ITL
        "ITL (ms)": 1,
        "P5 ITL (ms)": 1,
        "P25 ITL (ms)": 1,
        "P50 ITL (ms)": 1,
        "P95 ITL (ms)": 1,
        "P99 ITL (ms)": 1,
        # E2EL
        "E2EL (ms)": 1,
        "P5 E2EL (ms)": 1,
        "P25 E2EL (ms)": 1,
        "P50 E2EL (ms)": 1,
        "P95 E2EL (ms)": 1,
        "P99 E2EL (ms)": 1,
        # Throughput
        "Interactivity (tok/s/user)": 2,
        "Output Tput (tok/s)": 1,
    }

    display_dicts = []

    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            if col_name == "isl":
                value = row.get(
                    "input_sequence_length", row.get("isl", NOT_MEASURED_STR)
                )
            elif col_name == "osl":
                value = row.get(
                    "output_sequence_length", row.get("osl", NOT_MEASURED_STR)
                )
            elif col_name == "max_concurrency":
                value = row.get("max_con", NOT_MEASURED_STR)
            elif col_name == "num_prompts":
                value = row.get("num_prompts", NOT_MEASURED_STR)

            # TTFT metrics
            elif col_name == "ttft":
                value = row.get("mean_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p5_ttft":
                value = row.get("p5_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p25_ttft":
                value = row.get("p25_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p50_ttft":
                value = row.get("p50_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p95_ttft":
                value = row.get("p95_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p99_ttft":
                value = row.get("p99_ttft_ms", NOT_MEASURED_STR)

            # TPOT metrics
            elif col_name == "tpot":
                value = row.get("mean_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p5_tpot":
                value = row.get("p5_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p25_tpot":
                value = row.get("p25_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p50_tpot":
                value = row.get("p50_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p95_tpot":
                value = row.get("p95_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p99_tpot":
                value = row.get("p99_tpot_ms", NOT_MEASURED_STR)

            # ITL metrics
            elif col_name == "itl":
                value = row.get("mean_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p5_itl":
                value = row.get("p5_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p25_itl":
                value = row.get("p25_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p50_itl":
                value = row.get("p50_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p95_itl":
                value = row.get("p95_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p99_itl":
                value = row.get("p99_itl_ms", NOT_MEASURED_STR)

            # E2EL metrics
            elif col_name == "e2el":
                value = row.get("mean_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p5_e2el":
                value = row.get("p5_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p25_e2el":
                value = row.get("p25_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p50_e2el":
                value = row.get("p50_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p95_e2el":
                value = row.get("p95_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p99_e2el":
                value = row.get("p99_e2el_ms", NOT_MEASURED_STR)

            # Throughput metrics
            elif col_name == "tput_user":
                value = row.get("mean_tps", NOT_MEASURED_STR)
            elif col_name == "tput":
                value = row.get("tps_decode_throughput", NOT_MEASURED_STR)

            else:
                value = row.get(col_name, NOT_MEASURED_STR)

            # Format numeric values with consistent decimal places for proper alignment
            if value == NOT_MEASURED_STR or value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, (int, float)) and not (
                isinstance(value, float) and (value != value)
            ):  # Check for NaN
                decimal_places = decimal_places_map.get(display_header, 2)
                if decimal_places == 0:
                    # Format as integer
                    row_dict[display_header] = str(int(value))
                else:
                    # Format as float with specified decimal places
                    row_dict[display_header] = f"{float(value):.{decimal_places}f}"
            else:
                # Handle string numbers or other formats
                try:
                    numeric_value = float(value)
                    decimal_places = decimal_places_map.get(display_header, 2)
                    if decimal_places == 0:
                        row_dict[display_header] = str(int(numeric_value))
                    else:
                        row_dict[display_header] = f"{numeric_value:.{decimal_places}f}"
                except (ValueError, TypeError):
                    row_dict[display_header] = str(value)

        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def stress_test_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    """Generate stress test report using stress_tests-specific summary report module."""
    file_name_pattern = f"stress_test_{model_spec.model_id}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/stress_tests_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "stress_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stress Tests Summary")
    logger.info(f"Processing: {len(files)} files")
    if not files:
        logger.info("No stress test files found. Skipping.")
        return "", None, None, None

    # Use the stress_tests-specific generate_report function
    release_str, release_raw, disp_md_path, stats_file_path = (
        stress_test_generate_report_helper(files, output_dir, report_id, metadata)
    )

    # Generate stress test-specific release report
    # Build stress test performance report
    stress_test_release_str = (
        f"### Stress Test Results for {model_spec.model_name} on {args.device}\n\n"
    )

    if release_raw:
        # Check if percentile report is requested
        percentile_report = getattr(args, "percentile_report", False)

        # Create stress test-specific markdown table (detailed or simple format)
        if percentile_report:
            logger.info("Generating detailed percentile report for stress tests")
            stress_test_markdown = generate_stress_tests_markdown_table_detailed(
                release_raw, model_spec
            )
        else:
            logger.info(
                "Generating simplified report for stress tests (use --percentile-report for detailed statistics)"
            )
            stress_test_markdown = generate_stress_tests_markdown_table(
                release_raw, model_spec
            )

        stress_test_release_str += stress_test_markdown
    else:
        stress_test_release_str += (
            "No stress test results found for this model and device combination.\n"
        )

    # Save stress test-specific summary
    summary_fpath = output_dir / f"stress_test_summary_{report_id}.md"
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(stress_test_release_str)

    # Save raw data
    data_fpath = data_dir / f"stress_test_data_{report_id}.json"
    with data_fpath.open("w", encoding="utf-8") as f:
        json.dump(release_raw, f, indent=4, default=str)

    return stress_test_release_str, release_raw, summary_fpath, data_fpath


def benchmarks_release_data_format(
    model_spec, device_str, benchmark_summary_data, runtime_config=None
):
    """Convert the benchmark release data to the desired format"""
    reformated_benchmarks_release_data = []

    benchmark_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model": model_spec.model_name,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "backend": model_spec.model_type.name.lower(),
        "device": device_str,
        "num_requests": benchmark_summary_data.get("num_requests", 1),
        "num_inference_steps": benchmark_summary_data.get("num_inference_steps", 0),
        "ttft": benchmark_summary_data.get("mean_ttft_ms", 0) / 1000,
        "inference_steps_per_second": benchmark_summary_data.get(
            "inference_steps_per_second", 0
        ),
        "filename": benchmark_summary_data.get("filename", ""),
        "task_type": model_spec.model_type.name.lower(),
    }

    if (
        model_spec.model_type.name == ModelType.CNN.name
        or model_spec.model_type.name == ModelType.IMAGE.name
        or model_spec.model_type.name == ModelType.VIDEO.name
    ):
        benchmark_summary["tput_user"] = benchmark_summary_data.get("tput_user", 0)

    if model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name:
        benchmark_summary["ttft_p90"] = (
            benchmark_summary_data.get("p90_ttft_ms", 0) / 1000
        )
        benchmark_summary["ttft_p95"] = (
            benchmark_summary_data.get("p95_ttft_ms", 0) / 1000
        )
        benchmark_summary["rtr"] = benchmark_summary_data.get("rtr", 0)

    # Add Whisper-specific fields only for Whisper models
    if "whisper" in model_spec.hf_model_repo.lower():
        # Create a simple object that mimics what the utility functions expect
        class ModelSpecWrapper:
            def __init__(self, model_spec):
                self.model_spec = model_spec

        wrapper = ModelSpecWrapper(model_spec)
        streaming_enabled = is_streaming_enabled_for_whisper(wrapper)
        preprocessing_enabled = is_preprocessing_enabled_for_whisper(wrapper)

        benchmark_summary["streaming_enabled"] = streaming_enabled
        benchmark_summary["preprocessing_enabled"] = preprocessing_enabled

    reformated_benchmarks_release_data.append(benchmark_summary)
    return reformated_benchmarks_release_data


def benchmarks_release_data_format_embedding(
    model_spec, device_str, benchmark_summary_data
):
    """Convert the benchmark release data to the desired format for EMBEDDING models"""

    return [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model": model_spec.model_name,
            "model_name": model_spec.model_name,
            "model_id": model_spec.model_id,
            "backend": model_spec.model_type.name.lower(),
            "device": device_str,
            "num_requests": benchmark_summary_data.get("num_requests", 1),
            "ISL": benchmark_summary_data.get("input_sequence_length", 0),
            "concurrency": benchmark_summary_data.get("max_con", 0),
            "tput_user": benchmark_summary_data.get("mean_tps", 0),
            "tput_prefill": benchmark_summary_data.get("tps_prefill_throughput", 0),
            "e2el_ms": benchmark_summary_data.get("mean_e2el_ms", 0),
            "filename": benchmark_summary_data.get("filename", ""),
            "task_type": model_spec.model_type.name.lower(),
        }
    ]


def add_target_checks_cnn_image_video(
    targets, evals_release_data, benchmark_summary_data, metrics
):
    """Add target checks for CNN, IMAGE and VIDEO models based on evals and benchmark data."""
    logger.info("Adding target_checks to CNN, IMAGE and VIDEO benchmark release data")
    tput_user = evals_release_data[0].get("tput_user", 0) if evals_release_data else 0
    benchmark_summary_data["tput_user"] = tput_user

    # extract targets for functional, complete, target and calculate them
    target_tput_user = targets.tput_user
    complete_tput_user = target_tput_user / 2  # Complete target is 2x slower
    functional_tput_user = target_tput_user / 10  # Functional target is 10x slower

    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "ttft": metrics["functional_ttft"] / 1000,  # Convert ms to seconds
            "ttft_ratio": metrics["functional_ttft_ratio"],
            "ttft_check": metrics["functional_ttft_check"],
            "tput_check": 2 if tput_user > functional_tput_user else 3,
        },
        "complete": {
            "ttft": metrics["complete_ttft"] / 1000,  # Convert ms to seconds
            "ttft_ratio": metrics["complete_ttft_ratio"],
            "ttft_check": metrics["complete_ttft_check"],
            "tput_check": 2 if tput_user > complete_tput_user else 3,
        },
        "target": {
            "ttft": metrics["target_ttft"] / 1000,  # Convert ms to seconds
            "ttft_ratio": metrics["target_ttft_ratio"],
            "ttft_check": metrics["target_ttft_check"],
            "tput_check": 2 if tput_user > target_tput_user else 3,
        },
    }

    return target_checks


def add_target_checks_embedding(metrics):
    """Add target checks for EMBEDDING models based on evals and benchmark data."""
    logger.info("Adding target_checks to EMBEDDING benchmark release data")

    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "tput_user": metrics["functional_tput_user"],
            "tput_user_ratio": metrics["functional_tput_user_ratio"],
            "tput_user_check": metrics["functional_tput_user_check"],
            "tput_prefill": metrics["functional_tput_prefill"],
            "tput_prefill_ratio": metrics["functional_tput_prefill_ratio"],
            "tput_prefill_check": metrics["functional_tput_prefill_check"],
            "e2el_ms": metrics["functional_e2el_ms"],
            "e2el_ms_ratio": metrics["functional_e2el_ms_ratio"],
            "e2el_ms_check": metrics["functional_e2el_ms_check"],
        },
        "complete": {
            "tput_user": metrics["complete_tput_user"],
            "tput_user_ratio": metrics["complete_tput_user_ratio"],
            "tput_user_check": metrics["complete_tput_user_check"],
            "tput_prefill": metrics["complete_tput_prefill"],
            "tput_prefill_ratio": metrics["complete_tput_prefill_ratio"],
            "tput_prefill_check": metrics["complete_tput_prefill_check"],
            "e2el_ms": metrics["complete_e2el_ms"],
            "e2el_ms_ratio": metrics["complete_e2el_ms_ratio"],
            "e2el_ms_check": metrics["complete_e2el_ms_check"],
        },
        "target": {
            "tput_user": metrics["target_tput_user"],
            "tput_user_ratio": metrics["target_tput_user_ratio"],
            "tput_user_check": metrics["target_tput_user_check"],
            "tput_prefill": metrics["target_tput_prefill"],
            "tput_prefill_ratio": metrics["target_tput_prefill_ratio"],
            "tput_prefill_check": metrics["target_tput_prefill_check"],
            "e2el_ms": metrics["target_e2el_ms"],
            "e2el_ms_ratio": metrics["target_e2el_ms_ratio"],
            "e2el_ms_check": metrics["target_e2el_ms_check"],
        },
    }

    return target_checks


def add_target_checks_video(metrics):
    """Add target checks for VIDEO models based on evals and benchmark data."""
    logger.info("Adding target_checks to VIDEO benchmark release data")
    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "concurrency": metrics["functional_concurrency"],
            "concurrency_ratio": metrics["functional_concurrency_ratio"],
            "concurrency_check": metrics["functional_concurrency_check"],
        },
        "complete": {
            "concurrency": metrics["complete_concurrency"],
            "concurrency_ratio": metrics["complete_concurrency_ratio"],
            "concurrency_check": metrics["complete_concurrency_check"],
        },
        "target": {
            "concurrency": metrics["target_concurrency"],
            "concurrency_ratio": metrics["target_concurrency_ratio"],
            "concurrency_check": metrics["target_concurrency_check"],
        },
    }

    return target_checks


def calculate_target_metrics(metrics_config):
    """Calculate metrics for functional, complete, and target thresholds.

    Args:
        metrics_config: List of metric configurations. Each config is a dict with:
            - avg_metric: Average metric from benchmark results
            - target_metric: Target metric from performance reference
            - field_name: Name of the metric field
            - is_ascending_metric: If True, higher values are preffered (e.g., throughput).
            If False, lower values are preffered (e.g., latency, TTFT).

    Returns:
        Dict containing metrics for all target levels (functional, complete, target)
    """

    def get_metric_ratio_and_check(avg_metric, ref_metric, is_ascending_metric):
        if not ref_metric:
            return "Undefined", "Undefined"
        if not avg_metric:
            return 0.0, 1
        ratio = avg_metric / ref_metric
        if is_ascending_metric:
            check = 2 if ratio > 1.0 else 3
        else:
            check = 2 if ratio < 1.0 else 3
        return ratio, check

    # Define target level multipliers
    target_multipliers = {
        "functional": FUNCTIONAL_TARGET,  # 10x slower than target
        "complete": COMPLETE_TARGET,  # 2x slower than target
        "target": 1,  # actual target
    }

    metrics = {}
    for config in metrics_config:
        avg_metric = config["avg_metric"]
        target_metric = config["target_metric"]
        field_name = config["field_name"]
        is_ascending_metric = config.get("is_ascending_metric", False)

        # Skip if target_metric is None (e.g., for TTS when target_rtr is not set)
        if target_metric is None:
            logger.warning(
                f"Skipping metric calculation for {field_name}: target_metric is None"
            )
            continue

        for level, multiplier in target_multipliers.items():
            if is_ascending_metric:
                level_metric = target_metric / multiplier
            else:
                level_metric = target_metric * multiplier

            ratio, check = get_metric_ratio_and_check(
                avg_metric, level_metric, is_ascending_metric
            )

            metrics[f"{level}_{field_name}"] = level_metric
            metrics[f"{level}_{field_name}_ratio"] = ratio
            metrics[f"{level}_{field_name}_check"] = check

    return metrics


def add_target_checks_audio(metrics):
    logger.info("Adding target_checks to Audio benchmark release data")
    # tput_check is always 1 for now (no tput target)
    tput_check = 1
    target_checks = {
        "functional": {
            "ttft": metrics["functional_ttft"],
            "ttft_ratio": metrics["functional_ttft_ratio"],
            "ttft_check": metrics["functional_ttft_check"],
            "tput_check": tput_check,
        },
        "complete": {
            "ttft": metrics["complete_ttft"],
            "ttft_ratio": metrics["complete_ttft_ratio"],
            "ttft_check": metrics["complete_ttft_check"],
            "tput_check": tput_check,
        },
        "target": {
            "ttft": metrics["target_ttft"],
            "ttft_ratio": metrics["target_ttft_ratio"],
            "ttft_check": metrics["target_ttft_check"],
            "tput_check": tput_check,
        },
    }

    return target_checks


def add_target_checks_tts(metrics):
    logger.info("Adding target_checks to TTS benchmark release data")
    # tput_check is always 1 for now (no tput target)
    tput_check = 1
    target_checks = {
        "functional": {
            "ttft": metrics.get("functional_ttft"),
            "ttft_ratio": metrics.get("functional_ttft_ratio", "Undefined"),
            "ttft_check": metrics.get("functional_ttft_check", "Undefined"),
            "rtr_check": metrics.get("functional_rtr_check", 1),
            "tput_check": tput_check,
        },
        "complete": {
            "ttft": metrics.get("complete_ttft"),
            "ttft_ratio": metrics.get("complete_ttft_ratio", "Undefined"),
            "ttft_check": metrics.get("complete_ttft_check", "Undefined"),
            "rtr_check": metrics.get("complete_rtr_check", 1),
            "tput_check": tput_check,
        },
        "target": {
            "ttft": metrics.get("target_ttft"),
            "ttft_ratio": metrics.get("target_ttft_ratio", "Undefined"),
            "ttft_check": metrics.get("target_ttft_check", "Undefined"),
            "rtr_check": metrics.get("target_rtr_check", 1),
            "tput_check": tput_check,
        },
    }

    return target_checks


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    # runtime config loaded from JSON
    model = runtime_config.model
    device_str = runtime_config.device
    docker_server = runtime_config.docker_server

    workflow_config = WORKFLOW_REPORT_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    device = DeviceTypes.from_string(device_str)
    assert device == model_spec.device_type

    server_mode = "API"
    if docker_server:
        server_mode = "docker"

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_id = f"{model_spec.model_id}_{run_timestamp}"

    run_cmd = getattr(runtime_config, "original_run_command", None)
    if not run_cmd:
        # Compatibility fallback for older runtime JSON files that predate
        # original_run_command persistence.
        run_cmd_parts = [
            "python",
            "run.py",
            "--model",
            model,
            "--tt-device",
            device_str,
        ]
        if not model_spec.device_model_spec.default_impl:
            run_cmd_parts.extend(["--impl", model_spec.impl.impl_name])
        run_cmd_parts.extend(["--workflow", "release"])
        if docker_server:
            run_cmd_parts.append("--docker-server")
        if runtime_config.generate_report_schema:
            run_cmd_parts.append("--generate-report-schema")
        run_cmd = " ".join(run_cmd_parts)

    metadata = {
        "report_id": report_id,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "runtime_model_spec_json": args.runtime_model_spec_json,
        "model_repo": model_spec.hf_model_repo,
        "model_impl": model_spec.impl.impl_name,
        "inference_engine": model_spec.inference_engine,
        "device": device_str,
        "server_mode": server_mode,
        "release_version": model_spec.release_version,
        "tt_metal_commit": model_spec.tt_metal_commit,
        "vllm_commit": model_spec.vllm_commit,
        "run_command": run_cmd,
    }

    # Create a simple args object for the report generation functions
    class SimpleArgs:
        def __init__(
            self,
            output_path,
            model,
            device,
            runtime_model_spec_json,
            percentile_report=False,
        ):
            self.output_path = output_path
            self.model = model
            self.device = device
            self.runtime_model_spec_json = runtime_model_spec_json
            self.percentile_report = percentile_report

    percentile_report = runtime_config.percentile_report

    simple_args = SimpleArgs(
        args.output_path,
        model,
        device_str,
        args.runtime_model_spec_json,
        percentile_report=percentile_report,
    )

    # generate vLLM benchmarks report
    (
        benchmarks_release_str,
        benchmarks_release_data,
        _,
        benchmarks_data_file_path,
    ) = benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate AIPerf benchmarks report (separate detailed report)
    (
        aiperf_release_str,
        aiperf_release_data,
        _,
        aiperf_data_file_path,
    ) = aiperf_benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate GenAI-Perf benchmarks report (separate detailed report)
    (
        genai_perf_release_str,
        _,
        _,
        _,
    ) = genai_perf_benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate evals report
    evals_release_str, evals_release_data, _, _ = evals_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate tests report
    (
        tests_release_str,
        tests_release_data,
        _,
        _,
    ) = generate_tests_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )
    # generate stress test report
    (
        stress_tests_release_str,
        stress_tests_release_data,
        _,
        _,
    ) = stress_test_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate server tests report
    server_tests_release_str, _ = server_tests_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    logging.info("Release Summary\n\n")

    release_output_dir = Path(args.output_path) / "release"
    release_output_dir.mkdir(parents=True, exist_ok=True)
    release_data_dir = release_output_dir / "data"
    release_data_dir.mkdir(parents=True, exist_ok=True)
    release_file = release_output_dir / f"report_{report_id}.md"
    raw_file = release_data_dir / f"report_data_{report_id}.json"
    release_str = ""
    with raw_file.open("w", encoding="utf-8") as f:
        # Read detailed benchmark statistics from CSV if available
        benchmarks_detailed_data = None
        if benchmarks_data_file_path:
            try:
                with open(benchmarks_data_file_path, "r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    benchmarks_detailed_data = list(csv_reader)
            except Exception as e:
                logger.warning(f"Could not read benchmark CSV data: {e}")

        # Check for server tests JSON files
        server_tests_data = []

        # Use tests_release_data for parameter_support_tests
        parameter_support_tests_data = tests_release_data if tests_release_data else []

        # Read AIPerf benchmark data if available
        aiperf_detailed_data = None
        if aiperf_data_file_path:
            try:
                aiperf_detailed_data = _read_optional_csv_rows(
                    aiperf_data_file_path, "AIPerf"
                )
            except Exception as e:
                logger.warning(f"Could not read AIPerf CSV data: {e}")

        # Read server tests data if available
        server_tests_data = []
        server_tests_path = Path(project_root) / "test_reports"
        if server_tests_path.exists():
            server_tests_json_files = list(server_tests_path.glob("*.json"))
            if server_tests_json_files:
                logger.info(
                    f"Found {len(server_tests_json_files)} server test report(s)"
                )
                for json_file in server_tests_json_files:
                    try:
                        with open(json_file, "r", encoding="utf-8") as test_file:
                            test_data = json.load(test_file)
                            server_tests_data.append(test_data)
                    except Exception as e:
                        logger.warning(
                            f"Could not read server test file {json_file}: {e}"
                        )

        spec_tests_data = _normalize_spec_tests_payload(server_tests_data)

        # Build the final JSON output
        output_data = {
            "metadata": metadata,
            "benchmarks_summary": benchmarks_release_data,
            "aiperf_benchmarks": aiperf_release_data if aiperf_release_data else [],
            "evals": evals_release_data,
            "stress_tests": stress_tests_release_data,
            "benchmarks": benchmarks_detailed_data
            if benchmarks_detailed_data
            else [
                {
                    "model_id": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            "aiperf_benchmarks_detailed": aiperf_detailed_data
            if aiperf_detailed_data
            else [],
            "parameter_support_tests": parameter_support_tests_data
            if parameter_support_tests_data
            else {},
            "spec_tests": spec_tests_data,
        }

        # Add server_tests only if data exists
        if server_tests_data:
            output_data["server_tests"] = server_tests_data

        benchmark_target_evaluation = evaluate_benchmark_targets(output_data)
        output_data["benchmark_target_evaluation"] = benchmark_target_evaluation
        accepted, acceptance_blockers = acceptance_criteria_check(
            output_data, benchmark_target_evaluation
        )
        acceptance_summary_markdown = format_acceptance_summary_markdown(
            accepted,
            acceptance_blockers,
            benchmark_target_evaluation,
        )
        output_data["acceptance_criteria"] = accepted
        output_data["acceptance_blockers"] = acceptance_blockers
        output_data["acceptance_summary_markdown"] = acceptance_summary_markdown

        json.dump(output_data, f, indent=4)

    if runtime_config.generate_report_schema:
        schema_path = write_reports_schema(raw_file)
        logger.info(f"Generated report schema at: {schema_path}")

    validate_report_file(raw_file)
    release_str = build_release_report_markdown(raw_file)
    print(release_str)

    with release_file.open("w", encoding="utf-8") as f:
        f.write(release_str)

    main_return_code = 0
    return main_return_code


def server_tests_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    """Generate server tests report by reading all markdown files from test_reports directory.

    Args:
        args: Command line arguments
        server_mode: Server mode (API/docker)
        model_spec: Model specification
        report_id: Report identifier
        metadata: Additional metadata

    Returns:
        Tuple of (release_str, release_data) where:
            release_str: Markdown formatted string of all test reports
            release_data: List of test report data
    """
    output_dir = Path(args.output_path) / "server_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for markdown files in project_root/test_reports
    test_reports_path = Path(project_root) / "test_reports"

    logger.info("Server Tests Summary")

    if not test_reports_path.exists():
        logger.info(f"Test reports directory not found: {test_reports_path}")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
        )

    # Find all markdown files
    md_files = list(test_reports_path.glob("*.md"))

    logger.info(f"Processing: {len(md_files)} markdown file(s)")

    if not md_files:
        logger.info("No server test report markdown files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
        )

    # Read and combine all markdown files
    combined_markdown = []
    release_data = []

    for md_file in sorted(md_files):
        try:
            logger.info(f"Reading: {md_file.name}")
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                combined_markdown.append(f"#### {md_file.stem}\n\n{content}")

                # Try to extract JSON data if corresponding JSON file exists
                json_file = md_file.with_suffix(".json")
                if json_file.exists():
                    with open(json_file, "r", encoding="utf-8") as jf:
                        json_data = json.load(jf)
                        release_data.append(json_data)
        except Exception as e:
            logger.warning(f"Could not read file {md_file}: {e}")

    # Join all markdown content
    markdown_str = "\n\n---\n\n".join(combined_markdown)

    release_str = f"### Server Test Results for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    # Save combined report
    summary_fpath = output_dir / f"summary_{report_id}.md"
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(markdown_str)

    logger.info(f"Server tests summary saved to: {summary_fpath}")

    return release_str, release_data


def _normalize_spec_tests_payload(server_tests_data):
    reports = server_tests_data if isinstance(server_tests_data, list) else []
    results = []
    for report_index, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        for test_index, test_result in enumerate(report.get("tests", [])):
            if not isinstance(test_result, dict):
                continue
            results.append(
                {
                    **test_result,
                    "report_index": report_index,
                    "test_index": test_index,
                }
            )
    return {"reports": reports, "results": results}


if __name__ == "__main__":
    sys.exit(main())
