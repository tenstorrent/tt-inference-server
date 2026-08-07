# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ``vllm bench serve`` flat JSON shape (``--save-result``)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser
from .base import round_metric as _round


class VLLMBenchParser(LLMResultParser):
    tool = "vllm"
    tool_label = "vLLM"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        completed = _num(raw.get("completed"))
        record: Dict[str, Any] = {
            "tool": self.tool,
            "model": str(raw.get("model_id", "") or ""),
            "device": device,
            "timestamp": _format_date(raw.get("date", "")),
            "concurrency": _num_int(raw.get("max_concurrency")),
            "num_requests": _num_int(raw.get("completed")),
            "input_sequence_length": _per_request_int(
                raw.get("total_input_tokens"), completed
            ),
            "output_sequence_length": _per_request(
                raw.get("total_output_tokens"), completed
            ),
            # Percentile keys beyond mean/median/p99 are only present when the run
            # passed `--percentile-metrics` and `--metric-percentiles`. When absent
            # `_round` yields None, never 0 — a zero here would score as a perfect
            # result in the grading rubric.
            "mean_ttft_ms": _round(raw.get("mean_ttft_ms"), 4),
            "p50_ttft": _round(raw.get("median_ttft_ms"), 4),
            "p90_ttft": _round(raw.get("p90_ttft_ms"), 4),
            "p95_ttft": _round(raw.get("p95_ttft_ms"), 4),
            "p99_ttft": _round(raw.get("p99_ttft_ms"), 4),
            "std_ttft_ms": _round(raw.get("std_ttft_ms"), 4),
            "mean_tpot_ms": _round(raw.get("mean_tpot_ms"), 4),
            "p50_tpot_ms": _round(raw.get("median_tpot_ms"), 4),
            "p90_tpot_ms": _round(raw.get("p90_tpot_ms"), 4),
            "p95_tpot_ms": _round(raw.get("p95_tpot_ms"), 4),
            "p99_tpot_ms": _round(raw.get("p99_tpot_ms"), 4),
            "std_tpot_ms": _round(raw.get("std_tpot_ms"), 4),
            "mean_e2el_ms": _round(raw.get("mean_e2el_ms"), 4),
            "p50_e2el_ms": _round(raw.get("median_e2el_ms"), 4),
            "p90_e2el_ms": _round(raw.get("p90_e2el_ms"), 4),
            "p95_e2el_ms": _round(raw.get("p95_e2el_ms"), 4),
            "p99_e2el_ms": _round(raw.get("p99_e2el_ms"), 4),
            "std_e2el_ms": _round(raw.get("std_e2el_ms"), 4),
            "tps_decode_throughput": _round(raw.get("output_throughput"), 4),
            "request_throughput": _round(raw.get("request_throughput"), 4),
            "error_request_count": _errors(raw.get("failed")),
        }
        return self._wrap_record(record)


def _num(value: Any) -> Optional[float]:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _num_int(value: Any) -> Optional[int]:
    v = _num(value)
    return int(v) if v is not None else None


def _per_request(total: Any, completed: Optional[float]) -> Optional[float]:
    t = _num(total)
    if t is None or not completed:
        return None
    return round(t / completed, 1)


def _per_request_int(total: Any, completed: Optional[float]) -> Optional[int]:
    value = _per_request(total, completed)
    return int(round(value)) if value is not None else None


def _errors(value: Any) -> Optional[int]:
    """Failed request count, preserving a measured zero.

    ``vllm bench serve`` reports ``failed: 0`` on a clean run. Zero must survive
    as 0: it is the outcome acceptance has to confirm (RFP G.2.6 requires zero
    failed requests), and collapsing it to None would make "no requests failed"
    indistinguishable from "nobody looked".
    """
    v = _num(value)
    return int(v) if v is not None else None


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
