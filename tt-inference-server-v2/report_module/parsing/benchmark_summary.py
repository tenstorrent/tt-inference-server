# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Benchmark summary + target-check computation for non-text/VLM model types.

Text and VLM benchmarks are handled inside ``StandardReportStrategy`` via
per-reference target checks.  The model types here (IMAGE, CNN, VIDEO,
AUDIO, TEXT_TO_SPEECH, EMBEDDING) aggregate metrics across all benchmark
rows of a given tool and produce a single summary row with a tiered
``target_checks`` dict (``functional`` / ``complete`` / ``target``).

The logic mirrors the legacy implementation in
``workflows/run_reports.py`` but is scoped, typed, and independent of
file IO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from workflows.model_spec import ModelSpec
from workflows.utils import (
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from workflows.utils_report import PerformanceTargets, get_performance_targets
from workflows.workflow_types import ModelType, ReportCheckTypes

logger = logging.getLogger(__name__)

FUNCTIONAL_TARGET = 10
COMPLETE_TARGET = 2
TIER_MULTIPLIERS: Dict[str, float] = {
    "functional": FUNCTIONAL_TARGET,
    "complete": COMPLETE_TARGET,
    "target": 1,
}

_CNN_LIKE_TYPES = {ModelType.CNN.name, ModelType.IMAGE.name, ModelType.VIDEO.name}


@dataclass(frozen=True)
class MetricConfig:
    """One measured/target pair fed into ``calculate_target_metrics``."""

    avg_metric: float
    target_metric: Optional[float]
    field_name: str
    is_ascending_metric: bool = False


def calculate_target_metrics(
    metric_configs: List[MetricConfig],
) -> Dict[str, Any]:
    """Compute per-tier ratios and PASS/FAIL checks for each metric."""
    metrics: Dict[str, Any] = {}
    for config in metric_configs:
        if config.target_metric is None:
            logger.warning(
                f"Skipping metric calculation for {config.field_name}: "
                "target_metric is None"
            )
            continue
        _populate_tier_metrics(metrics, config)
    return metrics


def _populate_tier_metrics(
    metrics: Dict[str, Any], config: MetricConfig
) -> None:
    for tier, multiplier in TIER_MULTIPLIERS.items():
        tier_target = (
            config.target_metric / multiplier
            if config.is_ascending_metric
            else config.target_metric * multiplier
        )
        ratio, check = _metric_ratio_and_check(
            config.avg_metric, tier_target, config.is_ascending_metric
        )
        metrics[f"{tier}_{config.field_name}"] = tier_target
        metrics[f"{tier}_{config.field_name}_ratio"] = ratio
        metrics[f"{tier}_{config.field_name}_check"] = check


def _metric_ratio_and_check(
    avg_metric: float,
    ref_metric: Optional[float],
    is_ascending_metric: bool,
):
    if not ref_metric:
        return "Undefined", "Undefined"
    if not avg_metric:
        return 0.0, ReportCheckTypes.NA
    ratio = avg_metric / ref_metric
    if is_ascending_metric:
        passed = ratio > 1.0
    else:
        passed = ratio < 1.0
    return ratio, ReportCheckTypes.from_result(passed)


# ── Aggregation helpers ──────────────────────────────────────────────────


def _aggregate_rows_cnn_image_video_audio_tts(
    rows: List[Dict[str, Any]], model_spec: ModelSpec
) -> Dict[str, Any]:
    """Collapse multiple benchmark rows into one summary dict.

    Mirrors the old aggregation in ``workflows/run_reports.py``: averages
    ``mean_ttft_ms``/``rtr`` across all rows, keeps the latest values for
    count/filename fields (matches the overwrite-in-loop behaviour there).
    """
    if not rows:
        return {}

    total_ttft = 0.0
    total_rtr = 0.0
    is_tts = model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name
    summary: Dict[str, Any] = {}

    for row in rows:
        total_ttft += row.get("mean_ttft_ms", 0) or 0
        if is_tts:
            total_rtr += row.get("rtr", 0) or 0
        summary["num_requests"] = row.get("num_requests", 0)
        summary["num_inference_steps"] = row.get("num_inference_steps", 0)
        summary["inference_steps_per_second"] = row.get(
            "inference_steps_per_second", 0
        )
        summary["filename"] = row.get("filename", "")
        summary["mean_ttft_ms"] = row.get("mean_ttft_ms", 0)
        if is_tts:
            summary["p90_ttft_ms"] = row.get("p90_ttft_ms", 0)
            summary["p95_ttft_ms"] = row.get("p95_ttft_ms", 0)
            summary["rtr"] = row.get("rtr", 0)

    summary["avg_ttft_ms"] = total_ttft / len(rows)
    if is_tts:
        summary["avg_rtr"] = total_rtr / len(rows)
    return summary


# ── Target check builders (one per model type) ──────────────────────────


def _tput_check(measured: float, threshold: float) -> ReportCheckTypes:
    return ReportCheckTypes.from_result(measured > threshold)


_EMBEDDING_METRIC_KEYS = ("tput_user", "tput_prefill", "e2el_ms")


def _build_target_checks(
    metrics: Dict[str, Any],
    keys: List[str],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for tier in TIER_MULTIPLIERS:
        entry: Dict[str, Any] = {}
        for key in keys:
            if f"{tier}_{key}" not in metrics:
                continue
            entry[key] = metrics[f"{tier}_{key}"]
            entry[f"{key}_ratio"] = metrics[f"{tier}_{key}_ratio"]
            entry[f"{key}_check"] = metrics[f"{tier}_{key}_check"]
        if entry:
            checks[tier] = entry
    return checks


def _build_target_checks_cnn_image_video(
    targets: PerformanceTargets, tput_user: float, metrics: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    target_tput_user = targets.tput_user
    if target_tput_user is None:
        raise ValueError("CNN/IMAGE/VIDEO target requires tput_user")
    checks = _build_target_checks(metrics, ["ttft"])
    for tier, entry in checks.items():
        entry["ttft"] = entry["ttft"] / 1000
        entry["tput_check"] = _tput_check(tput_user, target_tput_user / TIER_MULTIPLIERS[tier])
    return checks


# ── Release-row formatters ──────────────────────────────────────────────


def _format_cnn_image_audio_tts_row(
    model_spec: ModelSpec, device_str: str, summary: Dict[str, Any]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model": model_spec.model_name,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "backend": model_spec.model_type.name.lower(),
        "device": device_str,
        "num_requests": summary.get("num_requests", 1),
        "num_inference_steps": summary.get("num_inference_steps", 0),
        "ttft": (summary.get("mean_ttft_ms", 0) or 0) / 1000,
        "inference_steps_per_second": summary.get("inference_steps_per_second", 0),
        "filename": summary.get("filename", ""),
        "task_type": model_spec.model_type.name.lower(),
    }

    if model_spec.model_type.name in _CNN_LIKE_TYPES:
        row["tput_user"] = summary.get("tput_user", 0)

    if model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name:
        row["ttft_p90"] = (summary.get("p90_ttft_ms", 0) or 0) / 1000
        row["ttft_p95"] = (summary.get("p95_ttft_ms", 0) or 0) / 1000
        row["rtr"] = summary.get("rtr", 0)

    if "whisper" in model_spec.hf_model_repo.lower():
        wrapper = _ModelSpecWrapper(model_spec)
        row["streaming_enabled"] = is_streaming_enabled_for_whisper(wrapper)
        row["preprocessing_enabled"] = is_preprocessing_enabled_for_whisper(wrapper)

    return row


def _format_embedding_row(
    model_spec: ModelSpec, device_str: str, first_row: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model": model_spec.model_name,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "backend": model_spec.model_type.name.lower(),
        "device": device_str,
        "num_requests": first_row.get("num_requests", 1),
        "ISL": first_row.get("input_sequence_length", 0),
        "concurrency": first_row.get("max_con", 0),
        "tput_user": first_row.get("mean_tps", 0),
        "tput_prefill": first_row.get("tps_prefill_throughput", 0),
        "e2el_ms": first_row.get("mean_e2el_ms", 0),
        "filename": first_row.get("filename", ""),
        "task_type": model_spec.model_type.name.lower(),
    }


class _ModelSpecWrapper:
    """Adapter so whisper helpers see ``.model_spec``."""

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec


# ── Public entry point ──────────────────────────────────────────────────


def build_summary_row(
    model_spec: ModelSpec,
    device_str: str,
    rows: List[Dict[str, Any]],
    evals_tput_user: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Produce a single benchmarks_summary row with ``target_checks``.

    Returns ``None`` when ``rows`` is empty.  The returned dict matches
    the shape that used to live in ``benchmarks_release_data[0]`` in the
    legacy pipeline.
    """
    if not rows:
        return None

    model_type = model_spec.model_type.name
    targets = get_performance_targets(
        model_spec.model_name, device_str, model_type=model_type
    )

    if model_type == ModelType.EMBEDDING.name:
        return _build_embedding_summary(model_spec, device_str, rows, targets)

    return _build_cnn_image_audio_tts_summary(
        model_spec, device_str, rows, targets, evals_tput_user
    )


def _build_cnn_image_audio_tts_summary(
    model_spec: ModelSpec,
    device_str: str,
    rows: List[Dict[str, Any]],
    targets: PerformanceTargets,
    evals_tput_user: Optional[float],
) -> Optional[Dict[str, Any]]:
    if targets.ttft_ms is None:
        logger.info(
            f"No ttft_ms target for {model_spec.model_name} on {device_str}; "
            "skipping benchmarks_summary."
        )
        return None

    summary = _aggregate_rows_cnn_image_video_audio_tts(rows, model_spec)
    model_type = model_spec.model_type.name

    metric_configs = [
        MetricConfig(
            avg_metric=summary.get("avg_ttft_ms", 0),
            target_metric=targets.ttft_ms,
            field_name="ttft",
            is_ascending_metric=False,
        )
    ]
    if model_type == ModelType.TEXT_TO_SPEECH.name and targets.rtr is not None:
        metric_configs.append(
            MetricConfig(
                avg_metric=summary.get("avg_rtr", 0),
                target_metric=targets.rtr,
                field_name="rtr",
                is_ascending_metric=True,
            )
        )

    metrics = calculate_target_metrics(metric_configs)

    tput_user = evals_tput_user if evals_tput_user is not None else 0
    if model_type in _CNN_LIKE_TYPES:
        summary["tput_user"] = tput_user
        target_checks = _build_target_checks_cnn_image_video(
            targets, tput_user, metrics
        )
    elif model_type == ModelType.AUDIO.name:
        target_checks = _build_target_checks(metrics, ["ttft"])
    elif model_type == ModelType.TEXT_TO_SPEECH.name:
        target_checks = _build_target_checks(metrics, ["ttft", "rtr"])
    else:
        logger.warning(f"Unsupported model type for summary: {model_type}")
        target_checks = _build_target_checks(metrics, ["ttft"])

    row = _format_cnn_image_audio_tts_row(model_spec, device_str, summary)
    row["target_checks"] = target_checks
    return row


def _build_embedding_summary(
    model_spec: ModelSpec,
    device_str: str,
    rows: List[Dict[str, Any]],
    targets: PerformanceTargets,
) -> Optional[Dict[str, Any]]:
    first = rows[0]
    metric_configs = [
        MetricConfig(
            avg_metric=first.get("mean_tps", 0),
            target_metric=targets.tput_user,
            field_name="tput_user",
            is_ascending_metric=True,
        ),
        MetricConfig(
            avg_metric=first.get("tps_prefill_throughput", 0),
            target_metric=targets.tput_prefill,
            field_name="tput_prefill",
            is_ascending_metric=True,
        ),
        MetricConfig(
            avg_metric=first.get("mean_e2el_ms", 0),
            target_metric=targets.e2el_ms,
            field_name="e2el_ms",
            is_ascending_metric=False,
        ),
    ]
    metrics = calculate_target_metrics(metric_configs)
    target_checks = _build_target_checks(metrics, list(_EMBEDDING_METRIC_KEYS))
    if not target_checks:
        logger.info(
            f"No embedding targets defined for {model_spec.model_name} "
            f"on {device_str}; skipping benchmarks_summary."
        )
        return None

    row = _format_embedding_row(model_spec, device_str, first)
    row["target_checks"] = target_checks
    return row
