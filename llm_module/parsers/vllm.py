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
            "tps_input_throughput": _input_throughput(raw),
            "tps_output_throughput": _round(raw.get("output_throughput"), 4),
            "tps_total_throughput": _total_throughput(raw),
            "request_throughput": _round(raw.get("request_throughput"), 4),
            "goodput_pct": _goodput_pct(raw),
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
                        if request_throughput is not None
                        and blocks_per_request is not None
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
                        if mean_e2el_ms is not None
                        and blocks_per_request is not None
                        and blocks_per_request > 0
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


def _input_throughput(raw: Mapping[str, Any]) -> Optional[float]:
    """Input (prefill) tokens per second over the benchmark window.

    vLLM reports no input throughput of its own, but its two throughputs
    share one duration, so their difference is exactly the input rate.
    """
    total = _num(raw.get("total_token_throughput"))
    output = _num(raw.get("output_throughput"))
    if total is not None and output is not None:
        return round(total - output, 4)
    duration = _num(raw.get("duration"))
    tokens = _num(raw.get("total_input_tokens"))
    if tokens is not None and duration:
        return round(tokens / duration, 4)
    return None


def _total_throughput(raw: Mapping[str, Any]) -> Optional[float]:
    """Input + output tokens per second over the benchmark window.

    Prefer vLLM's own ``total_token_throughput``; when it is missing,
    recompute it the same way vLLM does from the token totals.
    """
    total = _num(raw.get("total_token_throughput"))
    if total is not None:
        return round(total, 4)
    duration = _num(raw.get("duration"))
    input_tokens = _num(raw.get("total_input_tokens"))
    output_tokens = _num(raw.get("total_output_tokens"))
    if input_tokens is not None and output_tokens is not None and duration:
        return round((input_tokens + output_tokens) / duration, 4)
    return None


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
    if completed is None or completed < 0 or not completed.is_integer():
        return None
    completed_count = int(completed)
    if completed_count == 0:
        return 0.0

    output_lens = raw.get("output_lens")
    if isinstance(output_lens, list) and output_lens:
        valid_lens = [_num(value) for value in output_lens]
        if all(value is not None and value >= 0 for value in valid_lens):
            # vLLM detailed results include a zero-length entry for each failed
            # request, while request_throughput and mean_e2el_ms use completed
            # requests only. Remove exactly the excess zero entries so every
            # derived metric uses the same statistical population.
            excess = len(valid_lens) - completed_count
            if excess < 0:
                return None
            completed_lens = []
            for value in valid_lens:
                if excess and value == 0:
                    excess -= 1
                    continue
                completed_lens.append(value)
            if excess or len(completed_lens) != completed_count:
                return None
            return (
                sum(
                    math.ceil(value / output_block_size) if value else 0
                    for value in completed_lens
                )
                / completed_count
            )

    # Totals cannot recover mean(ceil(per-request tokens / block size)); using
    # ceil(mean tokens / block size) produces biased block throughput/latency.
    return None


def _errors(value: Any) -> Optional[int]:
    v = _num(value)
    return int(v) if v else None


def _goodput_pct(raw: Mapping[str, Any]) -> Optional[float]:
    """Share of completed requests that met the run's ``--goodput`` SLOs (%).

    vLLM only reports ``request_goodput`` (good requests/sec) when the run
    passed ``--goodput``. It shares its duration with ``request_throughput``
    (completed requests/sec), so the ratio is the fraction of requests
    meeting every SLO — the percentage requirements documents express as
    ``goodputPct`` / ``request_goodput``.
    """
    goodput = _num(raw.get("request_goodput"))
    throughput = _num(raw.get("request_throughput"))
    if goodput is None or not throughput:
        return None
    return round(100.0 * goodput / throughput, 2)


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
