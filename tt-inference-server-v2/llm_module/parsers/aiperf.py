# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ai-dynamo/aiperf ``JsonExportData`` shape."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from report_module.schema import Block

from .base import LLMResultParser

LATENCY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("request_latency", "request_latency"),
    ("time_to_first_token", "ttft"),
    ("time_to_second_token", "ttst"),
    ("inter_token_latency", "itl"),
    ("inter_chunk_latency", "icl"),
)
THROUGHPUT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("request_throughput", "request_throughput"),
    ("output_token_throughput", "output_token_throughput"),
    ("output_token_throughput_per_user", "output_token_throughput_per_user"),
    ("goodput", "goodput"),
)
SEQUENCE_LENGTH_METRICS: Tuple[Tuple[str, str], ...] = (
    ("input_sequence_length", "input_sequence_length"),
    ("output_sequence_length", "output_sequence_length"),
)
COUNT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("request_count", "request_count"),
    ("good_request_count", "good_request_count"),
    ("error_request_count", "error_request_count"),
    ("output_token_count", "output_token_count"),
    ("reasoning_token_count", "reasoning_token_count"),
    ("benchmark_duration", "benchmark_duration"),
    ("total_output_tokens", "total_output_tokens"),
    ("total_reasoning_tokens", "total_reasoning_tokens"),
    ("total_isl", "total_isl"),
    ("total_osl", "total_osl"),
)
ORDERED_STATS: Tuple[str, ...] = (
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "std",
)


class AIPerfParser(LLMResultParser):
    kind = "aiperf"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record: Dict[str, Any] = {
            "kind": self.kind,
            "model": _model_name(raw),
            "device": device,
            "timestamp": _timestamp(raw),
            "Run Configuration": _run_configuration(raw),
            "Latency Statistics": _metric_table(raw, LATENCY_METRICS),
            "Throughput": _metric_table(raw, THROUGHPUT_METRICS),
            "Sequence Lengths": _metric_table(raw, SEQUENCE_LENGTH_METRICS),
            "Counts & Totals": _metric_table(raw, COUNT_METRICS),
            "Telemetry": _telemetry(raw),
        }
        return self._wrap_record(record)


def _metric_table(
    raw: Mapping[str, Any],
    metrics: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    columns_present = _detect_present_stats(raw, metrics)
    rows: List[Dict[str, Any]] = []
    for source_key, label in metrics:
        result = raw.get(source_key)
        if not isinstance(result, Mapping):
            continue
        row: Dict[str, Any] = {"metric": label, "unit": result.get("unit", "")}
        for stat in columns_present:
            row[stat] = _round(result.get(stat), 4)
        rows.append(row)
    return rows


def _detect_present_stats(
    raw: Mapping[str, Any],
    metrics: Sequence[Tuple[str, str]],
) -> List[str]:
    seen: Dict[str, None] = {}
    for source_key, _label in metrics:
        result = raw.get(source_key)
        if not isinstance(result, Mapping):
            continue
        for stat in ORDERED_STATS:
            if result.get(stat) is not None:
                seen[stat] = None
    return [s for s in ORDERED_STATS if s in seen]


def _model_name(raw: Mapping[str, Any]) -> str:
    config = raw.get("input_config")
    if not isinstance(config, Mapping):
        return ""
    endpoint = config.get("endpoint")
    if isinstance(endpoint, Mapping):
        names = endpoint.get("model_names")
        if isinstance(names, list) and names:
            return str(names[0])
        model = endpoint.get("model")
        if model:
            return str(model)
    model = config.get("model")
    return str(model) if model else ""


def _timestamp(raw: Mapping[str, Any]) -> str:
    start = raw.get("start_time")
    if isinstance(start, str) and start:
        return _normalize_iso(start)
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_iso(text: str) -> str:
    cleaned = text.rstrip("Z").split(".")[0].replace("T", " ")
    try:
        parsed = dt.datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text


def _run_configuration(raw: Mapping[str, Any]) -> Dict[str, Any]:
    config = raw.get("input_config") or {}
    if not isinstance(config, Mapping):
        config = {}
    out: Dict[str, Any] = {
        "schema_version": raw.get("schema_version"),
        "aiperf_version": raw.get("aiperf_version"),
        "benchmark_id": raw.get("benchmark_id"),
        "model": _model_name(raw),
        "start_time": raw.get("start_time"),
        "end_time": raw.get("end_time"),
        "was_cancelled": raw.get("was_cancelled"),
    }
    endpoint = config.get("endpoint")
    if isinstance(endpoint, Mapping):
        out["endpoint_type"] = endpoint.get("type")
    loadgen = config.get("loadgen")
    if isinstance(loadgen, Mapping):
        for key in ("concurrency", "request_rate"):
            if key in loadgen:
                out[key] = loadgen[key]
    return out


def _telemetry(raw: Mapping[str, Any]) -> Any:
    telemetry = raw.get("telemetry_data") or {}
    if not isinstance(telemetry, Mapping):
        return {}
    gpu_metrics = telemetry.get("gpu_metrics")
    if isinstance(gpu_metrics, Mapping):
        return {str(name): dict(values) for name, values in gpu_metrics.items()}
    return {str(k): v for k, v in telemetry.items()}


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value
