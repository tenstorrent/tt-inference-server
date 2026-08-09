# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ``vllm bench serve`` flat JSON shape (``--save-result``)."""

from __future__ import annotations

import datetime as dt
import math
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
            "mean_ttft_ms": _round(raw.get("mean_ttft_ms"), 4),
            "p50_ttft": _round(raw.get("median_ttft_ms"), 4),
            "p99_ttft": _round(raw.get("p99_ttft_ms"), 4),
            "mean_tpot_ms": _round(raw.get("mean_tpot_ms"), 4),
            "mean_e2el_ms": _round(raw.get("mean_e2el_ms"), 4),
            "tps_decode_throughput": _round(raw.get("output_throughput"), 4),
            "request_throughput": _round(raw.get("request_throughput"), 4),
            "error_request_count": _errors(raw.get("failed")),
        }
        output_block_size = _num_int(raw.get("tt_output_block_size")) or 1
        if output_block_size > 1:
            blocks_per_request = _blocks_per_request(
                raw,
                completed=completed,
                output_block_size=output_block_size,
            )
            request_throughput = _num(raw.get("request_throughput"))
            mean_e2el_ms = _num(raw.get("mean_e2el_ms"))
            record.update(
                {
                    "metric_semantics": "block_granular",
                    "output_block_size": output_block_size,
                    "output_blocks_per_request": _round(blocks_per_request, 4),
                    "output_blocks_per_second": _round(
                        request_throughput * blocks_per_request
                        if request_throughput is not None and blocks_per_request
                        else None,
                        4,
                    ),
                    # vLLM's token TPOT divides inter-event time by every token ID
                    # delivered in that event. A 256-ID block can therefore report
                    # a near-zero TPOT even though producing it took tens of
                    # seconds. Derive latency from request E2EL and the number of
                    # scheduling blocks that request emitted instead.
                    "mean_block_latency_ms": _round(
                        mean_e2el_ms / blocks_per_request
                        if mean_e2el_ms is not None and blocks_per_request
                        else None,
                        4,
                    ),
                    "primary_throughput_metric": "output_blocks_per_second",
                    "primary_latency_metric": "mean_block_latency_ms",
                }
            )
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


def _blocks_per_request(
    raw: Mapping[str, Any],
    *,
    completed: float | None,
    output_block_size: int,
) -> float | None:
    output_lens = raw.get("output_lens")
    if isinstance(output_lens, list) and output_lens:
        valid_lens = [_num(value) for value in output_lens]
        if all(value is not None and value >= 0 for value in valid_lens):
            return sum(
                math.ceil(value / output_block_size) if value else 0
                for value in valid_lens
            ) / len(valid_lens)

    output_tokens = _per_request(raw.get("total_output_tokens"), completed)
    if output_tokens is None or output_tokens <= 0:
        return None
    return float(math.ceil(output_tokens / output_block_size))


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
