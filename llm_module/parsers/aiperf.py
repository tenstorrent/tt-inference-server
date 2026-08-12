# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ai-dynamo/aiperf ``JsonExportData`` shape."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser, decode_throughput
from .base import metric_stat as _stat
from .base import metric_stat_int as _stat_int


class AIPerfParser(LLMResultParser):
    tool = "aiperf"
    tool_label = "AIPerf"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record: Dict[str, Any] = {
            "tool": self.tool,
            "model": _model_name(raw),
            "device": device,
            "timestamp": _timestamp(raw),
            "concurrency": _concurrency(raw),
            "num_requests": _stat_int(raw, "request_count"),
            "input_sequence_length": _stat_int(raw, "input_sequence_length"),
            "output_sequence_length": _stat(raw, "output_sequence_length"),
            "mean_ttft_ms": _stat_any(raw, "time_to_first_token", "avg", "mean"),
            "p50_ttft": _stat_any(raw, "time_to_first_token", "p50", "median"),
            "p90_ttft": _stat(raw, "time_to_first_token", "p90"),
            "p95_ttft": _stat(raw, "time_to_first_token", "p95"),
            "p99_ttft": _stat(raw, "time_to_first_token", "p99"),
            "std_ttft_ms": _stat(raw, "time_to_first_token", "std"),
            "mean_tpot_ms": _stat_any(raw, "inter_token_latency", "avg", "mean"),
            "p50_tpot_ms": _stat_any(raw, "inter_token_latency", "p50", "median"),
            "p90_tpot_ms": _stat(raw, "inter_token_latency", "p90"),
            "p95_tpot_ms": _stat(raw, "inter_token_latency", "p95"),
            "p99_tpot_ms": _stat(raw, "inter_token_latency", "p99"),
            "std_tpot_ms": _stat(raw, "inter_token_latency", "std"),
            "mean_e2el_ms": _stat_any(raw, "request_latency", "avg", "mean"),
            "p50_e2el_ms": _stat_any(raw, "request_latency", "p50", "median"),
            "p90_e2el_ms": _stat(raw, "request_latency", "p90"),
            "p95_e2el_ms": _stat(raw, "request_latency", "p95"),
            "p99_e2el_ms": _stat(raw, "request_latency", "p99"),
            "std_e2el_ms": _stat(raw, "request_latency", "std"),
            "tput_user": _stat(raw, "output_token_throughput_per_user"),
            # Graded against the `tput` target, which is defined as decode
            # interactivity x concurrency -- so it is derived the same way rather
            # than taken from AIPerf's wall-clock output_token_throughput. See
            # llm_module.parsers.base.decode_throughput.
            "tps_decode_throughput": None,  # set below, needs concurrency
            "tps_output_throughput": _stat(raw, "output_token_throughput"),
            "request_throughput": _stat(raw, "request_throughput"),
            "error_request_count": _errors(raw),
        }
        record["tps_decode_throughput"] = decode_throughput(
            record["tput_user"], record["mean_tpot_ms"], record["concurrency"]
        )
        return self._wrap_record(record)


def _stat_any(raw: Mapping[str, Any], key: str, *stats: str) -> Any:
    """First present stat from a metric block, trying each name in turn.

    AIPerf labels the same statistic differently across versions — ``avg`` or
    ``mean``, ``p50`` or ``median``. ``llm_module/drivers/aiperf_prefix_cache``
    already handles both when reading real AIPerf output; this keeps the
    standard benchmark parser consistent with it rather than silently
    returning None on an export that uses the other spelling.
    """
    metric = raw.get(key)
    if not isinstance(metric, Mapping):
        return None
    for stat in stats:
        if stat in metric:
            return _stat(raw, key, stat)
    return None


def _concurrency(raw: Mapping[str, Any]) -> Optional[int]:
    config = raw.get("input_config")
    loadgen = config.get("loadgen") if isinstance(config, Mapping) else None
    if isinstance(loadgen, Mapping):
        value = loadgen.get("concurrency")
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _errors(raw: Mapping[str, Any]) -> Optional[int]:
    """Failed request count, preserving a measured zero.

    Zero must survive as 0. It is the *good* outcome and the one acceptance has
    to confirm — RFP G.2.6 requires zero failed requests — so collapsing it to
    None would make "no requests failed" indistinguishable from "nobody looked",
    and an unverifiable requirement reads as a satisfied one.

    AIPerf makes that harder than it sounds: it omits ``error_request_count``
    from the export entirely when nothing failed, and only emits it (alongside
    ``error_isl`` / ``total_error_isl``) once at least one request has failed.
    Reading absence as "nobody looked" therefore inverted the meaning of AIPerf's
    healthiest possible output, and blocked the acceptance gate on every clean
    point.

    ``error_summary`` is the signal to use instead. AIPerf always writes it — an
    empty list on a clean run, one entry per distinct failure otherwise — so it
    is an affirmative "the tool looked", which is exactly what the gate needs to
    distinguish a confirmed zero from silence. Verified against v0.5.0 with the
    simulator's ``--failure-injection-rate``: at 0 the export carries
    ``error_summary: []`` and no count; at 50 it carries ``error_request_count:
    5`` and five summarised failures.

    An export with neither key still yields None and still blocks.
    """
    value = raw.get("error_request_count")
    if isinstance(value, Mapping):
        value = value.get("avg")
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        return int(value)

    summary = raw.get("error_summary")
    if isinstance(summary, list):
        # Normally the aggregate above is present whenever the summary is
        # non-empty; sum it rather than assume, so a partial export cannot
        # report zero failures when it listed some.
        total = 0
        for entry in summary:
            count = entry.get("count") if isinstance(entry, Mapping) else None
            total += int(count) if isinstance(count, (int, float)) else 1
        return total
    return None


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
