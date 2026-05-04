# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC

"""Adapter for the InferenceX ``benchmark_serving.py`` flat JSON shape."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping, Tuple

LATENCY_METRICS: Tuple[str, ...] = ("ttft", "tpot", "itl", "e2el")
LATENCY_STATS: Tuple[str, ...] = ("mean", "median", "std", "p90", "p95", "p99")
GOODPUT_KEY = "request_goodput:"


def to_report_record(
    raw: Mapping[str, Any],
    *,
    device: str = "",
) -> Dict[str, Any]:
    return {
        "kind": "inferencex",
        "model": str(raw.get("model_id", "") or ""),
        "device": device,
        "timestamp": _format_date(raw.get("date", "")),
        "Run Configuration": _run_configuration(raw),
        "Latency Statistics (ms)": _latency_stats(raw),
        "Throughput": _throughput(raw),
        "Request Totals": _request_totals(raw),
    }


def _run_configuration(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "backend": raw.get("backend"),
        "model": raw.get("model_id"),
        "tokenizer": raw.get("tokenizer_id"),
        "num_prompts": raw.get("num_prompts"),
        "max_concurrency": raw.get("max_concurrency"),
        "request_rate": raw.get("request_rate"),
        "burstiness": raw.get("burstiness"),
        "best_of": raw.get("best_of"),
        "duration_sec": _round(raw.get("duration"), 2),
    }


def _latency_stats(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    columns_present = _detect_present_stats(raw)
    rows: List[Dict[str, Any]] = []
    for metric in LATENCY_METRICS:
        if not any(f"{stat}_{metric}_ms" in raw for stat in LATENCY_STATS):
            continue
        row: Dict[str, Any] = {"metric": metric}
        for stat in columns_present:
            row[stat] = _round(raw.get(f"{stat}_{metric}_ms"), 4)
        rows.append(row)
    return rows


def _detect_present_stats(raw: Mapping[str, Any]) -> List[str]:
    seen: List[str] = []
    for stat in LATENCY_STATS:
        for metric in LATENCY_METRICS:
            if f"{stat}_{metric}_ms" in raw:
                seen.append(stat)
                break
    return seen


def _throughput(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "requests_per_second": _round(raw.get("request_throughput"), 4),
        "request_goodput": _round(raw.get(GOODPUT_KEY), 4),
        "output_tokens_per_second": _round(raw.get("output_throughput"), 2),
        "total_tokens_per_second": _round(raw.get("total_token_throughput"), 2),
    }


def _request_totals(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "completed": raw.get("completed"),
        "total_input_tokens": raw.get("total_input_tokens"),
        "total_output_tokens": raw.get("total_output_tokens"),
    }


def _format_date(date_str: Any) -> str:
    if not date_str:
        return ""
    text = str(date_str)
    for fmt in ("%Y%m%d-%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return text


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value
