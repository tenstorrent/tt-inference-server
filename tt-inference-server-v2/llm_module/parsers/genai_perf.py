# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the NVIDIA genai-perf ``*_genai_perf.json`` shape."""

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
)
THROUGHPUT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("request_throughput", "request_throughput"),
    ("output_token_throughput", "output_token_throughput"),
    ("output_token_throughput_per_user", "output_token_throughput_per_user"),
)
SEQUENCE_LENGTH_METRICS: Tuple[Tuple[str, str], ...] = (
    ("input_sequence_length", "input_sequence_length"),
    ("output_sequence_length", "output_sequence_length"),
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


class GenAIPerfParser(LLMResultParser):
    kind = "genai_perf"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record: Dict[str, Any] = {
            "kind": self.kind,
            "model": _model_name(raw),
            "device": device,
            "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Run Configuration": _run_configuration(raw),
            "Latency Statistics": _metric_table(raw, LATENCY_METRICS),
            "Throughput": _metric_table(raw, THROUGHPUT_METRICS),
            "Sequence Lengths": _metric_table(raw, SEQUENCE_LENGTH_METRICS),
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
    model = config.get("model") or config.get("formatted_model_name")
    if isinstance(model, list) and model:
        return str(model[0])
    return str(model) if model else ""


def _run_configuration(raw: Mapping[str, Any]) -> Dict[str, Any]:
    config = raw.get("input_config") or {}
    if not isinstance(config, Mapping):
        config = {}
    keys = (
        "model",
        "endpoint_type",
        "service_kind",
        "url",
        "concurrency",
        "request_rate",
        "num_prompts",
        "synthetic_input_tokens_mean",
        "output_tokens_mean",
        "tokenizer",
        "streaming",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if key in config:
            value = config[key]
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            out[key] = value
    return out


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value
