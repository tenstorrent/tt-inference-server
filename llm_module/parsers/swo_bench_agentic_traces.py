# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the SwarmOne swo-bench agentic trace-replay driver output.

Consumes the combined payload from
:class:`llm_module.drivers.swo_bench_agentic_traces.SwoBenchAgenticTracesDriver`
and emits one :class:`report_module.schema.Block`. It shares the ``agentic_traces``
kind with the InferenceX/AIPerf parser so the report generator collapses both
harnesses' rows into a single section; the ``backend`` / ``trace_source`` fields
distinguish them.

No dedicated renderer is registered: the payload is a flat record, which is what
the generic renderer expects.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

from report_module.schema import Block

from .base import LLMResultParser


class SwoBenchAgenticTracesParser(LLMResultParser):
    kind = "agentic_traces"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record = dict(raw)
        record.setdefault("kind", self.kind)
        record["model"] = str(raw.get("model_id") or raw.get("model") or "")
        record["device"] = device
        record["timestamp"] = _normalize_timestamp(raw.get("date"))

        # error_rate_pct is a percentage; carry a fraction alongside it so it can
        # be compared to a failed-request threshold without unit confusion.
        error_rate_pct = raw.get("error_rate_pct")
        if isinstance(error_rate_pct, (int, float)):
            record["error_rate"] = float(error_rate_pct) / 100.0

        # Surface run provenance as first-class fields: a replay number is only
        # reproducible alongside the client version, session, and mode.
        metadata = raw.get("metadata") or {}
        if isinstance(metadata, Mapping):
            for key in ("model_id", "mode"):
                if key in metadata and key not in record:
                    record[key] = metadata[key]

        return self._wrap_record(record)


def _normalize_timestamp(raw_date: Any) -> str:
    """Return ``YYYY-MM-DD HH:MM:SS`` (best effort, never raises)."""
    if isinstance(raw_date, str) and raw_date:
        for fmt in ("%Y%m%d-%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return dt.datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        return raw_date
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


__all__ = ["SwoBenchAgenticTracesParser"]
