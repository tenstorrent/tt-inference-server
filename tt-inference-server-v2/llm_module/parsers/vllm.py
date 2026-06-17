# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ``vllm bench serve`` flat JSON shape (``--save-result``)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser


class VLLMBenchParser(LLMResultParser):
    kind = "vllm"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        completed = _num(raw.get("completed"))
        record: Dict[str, Any] = {
            "kind": self.kind,
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
            "mean_ttft_ms": _round(raw.get("mean_ttft_ms"), 4),
            "p50_ttft": _round(raw.get("median_ttft_ms"), 4),
            "p99_ttft": _round(raw.get("p99_ttft_ms"), 4),
            "mean_tpot_ms": _round(raw.get("mean_tpot_ms"), 4),
            "mean_e2el_ms": _round(raw.get("mean_e2el_ms"), 4),
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
    v = _num(value)
    return int(v) if v else None


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
