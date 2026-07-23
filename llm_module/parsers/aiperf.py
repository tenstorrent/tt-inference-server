# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the ai-dynamo/aiperf ``JsonExportData`` shape."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser
from .base import metric_stat as _stat
from .base import metric_stat_int as _stat_int


class AIPerfParser(LLMResultParser):
    kind = "aiperf"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record: Dict[str, Any] = {
            "kind": self.kind,
            "model": _model_name(raw),
            "device": device,
            "timestamp": _timestamp(raw),
            "concurrency": _concurrency(raw),
            "num_requests": _stat_int(raw, "request_count"),
            "input_sequence_length": _stat_int(raw, "input_sequence_length"),
            "output_sequence_length": _stat(raw, "output_sequence_length"),
            "mean_ttft_ms": _stat(raw, "time_to_first_token", "avg"),
            "p50_ttft": _stat(raw, "time_to_first_token", "p50"),
            "p99_ttft": _stat(raw, "time_to_first_token", "p99"),
            "mean_tpot_ms": _stat(raw, "inter_token_latency", "avg"),
            "mean_e2el_ms": _stat(raw, "request_latency", "avg"),
            "tput_user": _stat(raw, "output_token_throughput_per_user"),
            "tps_decode_throughput": _stat(raw, "output_token_throughput"),
            "request_throughput": _stat(raw, "request_throughput"),
            "error_request_count": _errors(raw),
        }
        return self._wrap_record(record)


def _concurrency(raw: Mapping[str, Any]) -> Optional[int]:
    config = raw.get("input_config")
    loadgen = config.get("loadgen") if isinstance(config, Mapping) else None
    if isinstance(loadgen, Mapping):
        value = loadgen.get("concurrency")
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _errors(raw: Mapping[str, Any]) -> Optional[int]:
    value = raw.get("error_request_count")
    if isinstance(value, Mapping):
        value = value.get("avg")
    if isinstance(value, (int, float)) and value:
        return int(value)
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
