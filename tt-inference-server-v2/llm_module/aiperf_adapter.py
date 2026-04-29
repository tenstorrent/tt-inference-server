# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping

LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")
LATENCY_STATS = ("mean", "median", "p99", "std")


def to_report_record(
    raw: Mapping[str, Any],
    *,
    device: str = "",
) -> Dict[str, Any]:
    """Build the single aiperf report record from a raw aiperf summary."""
    return {
        "kind": "aiperf",
        "model": raw.get("model_id", ""),
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
        "model_id": raw.get("model_id"),
        "tokenizer_id": raw.get("tokenizer_id"),
        "num_prompts": raw.get("num_prompts"),
        "max_concurrency": raw.get("max_concurrency"),
    }


def _latency_stats(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in LATENCY_METRICS:
        row: Dict[str, Any] = {"metric": metric}
        for stat in LATENCY_STATS:
            row[stat] = _round(raw.get(f"{stat}_{metric}_ms"), 2)
        rows.append(row)
    return rows


def _throughput(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "output_tokens_per_second": _round(raw.get("output_token_throughput"), 2),
        "requests_per_second": _round(raw.get("request_throughput"), 4),
    }


def _request_totals(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "completed": raw.get("completed"),
        "total_input_tokens": raw.get("total_input_tokens"),
        "total_output_tokens": raw.get("total_output_tokens"),
    }


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def _format_date(date_str: str) -> str:
    """Reformat aiperf's ``"20251210-155450"`` → ``"2025-12-10 15:54:50"``."""
    if not date_str:
        return ""
    try:
        parsed = dt.datetime.strptime(date_str, "%Y%m%d-%H%M%S")
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(date_str)
