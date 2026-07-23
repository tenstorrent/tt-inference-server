# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the NVIDIA genai-perf ``*_genai_perf.json`` shape."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser
from .base import metric_stat as _stat
from .base import metric_stat_int as _stat_int


class GenAIPerfParser(LLMResultParser):
    kind = "genai_perf"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record: Dict[str, Any] = {
            "kind": self.kind,
            "model": _model_name(raw),
            "device": device,
            "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
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
        }
        return self._wrap_record(record)


def _concurrency(raw: Mapping[str, Any]) -> Optional[int]:
    config = raw.get("input_config")
    if not isinstance(config, Mapping):
        return None
    # Newer genai-perf nests it under perf_analyzer.stimulus.concurrency;
    # older dumps put it at input_config.concurrency.
    value = config.get("concurrency")
    if value is None:
        pa = config.get("perf_analyzer")
        stimulus = pa.get("stimulus") if isinstance(pa, Mapping) else None
        if isinstance(stimulus, Mapping):
            value = stimulus.get("concurrency")
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _model_name(raw: Mapping[str, Any]) -> str:
    config = raw.get("input_config")
    if not isinstance(config, Mapping):
        return ""
    model = config.get("model") or config.get("formatted_model_name")
    if isinstance(model, list) and model:
        return str(model[0])
    return str(model) if model else ""
