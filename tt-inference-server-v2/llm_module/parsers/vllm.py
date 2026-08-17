# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ``vllm bench serve`` flat JSON shape (``--save-result``)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping, Tuple

from report_module.schema import Block

from .base import LLMResultParser

LATENCY_METRICS: Tuple[str, ...] = ("ttft", "tpot", "itl", "e2el")
LATENCY_STATS: Tuple[str, ...] = (
    "mean",
    "median",
    "std",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
)


class VLLMBenchParser(LLMResultParser):
    kind = "vllm"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record: Dict[str, Any] = {
            "kind": self.kind,
            "model": str(raw.get("model_id", "") or ""),
            "device": device,
            "timestamp": _format_date(raw.get("date", "")),
            "Run Configuration": _run_configuration(raw),
            "Latency Statistics (ms)": _latency_stats(raw),
            "Throughput": _throughput(raw),
            "Request Totals": _request_totals(raw),
        }
        return self._wrap_record(record)


def _run_configuration(raw: Mapping[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "backend": raw.get("backend"),
        "endpoint_type": raw.get("endpoint_type"),
        "label": raw.get("label"),
        "model": raw.get("model_id"),
        "tokenizer": raw.get("tokenizer_id"),
        "num_prompts": raw.get("num_prompts"),
        "max_concurrency": raw.get("max_concurrency"),
        "request_rate": raw.get("request_rate"),
        "burstiness": raw.get("burstiness"),
        "duration_sec": _round(raw.get("duration"), 2),
    }
    if raw.get("ramp_up_strategy") is not None:
        config["ramp_up_strategy"] = raw.get("ramp_up_strategy")
        config["ramp_up_start_rps"] = raw.get("ramp_up_start_rps")
        config["ramp_up_end_rps"] = raw.get("ramp_up_end_rps")
    return config


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
        "request_goodput": _round(raw.get("request_goodput"), 4),
        "output_tokens_per_second": _round(raw.get("output_throughput"), 2),
        "total_tokens_per_second": _round(raw.get("total_token_throughput"), 2),
        "max_output_tokens_per_second": _round(raw.get("max_output_tokens_per_s"), 2),
        "max_concurrent_requests": raw.get("max_concurrent_requests"),
    }


def _request_totals(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "completed": raw.get("completed"),
        "failed": raw.get("failed"),
        "total_input_tokens": raw.get("total_input_tokens"),
        "total_output_tokens": raw.get("total_output_tokens"),
    }


def _format_date(date_str: Any) -> str:
    if not date_str:
        return ""
    text = str(date_str)
    for fmt in (
        "%Y%m%d-%H%M%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
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
