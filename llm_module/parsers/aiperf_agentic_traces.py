# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the agentic trace-replay driver output.

Consumes the combined payload from
:class:`llm_module.drivers.aiperf_agentic_traces.AIPerfAgenticTracesDriver` and
emits one :class:`report_module.schema.Block` per run, all sharing kind
``agentic_traces`` so the report generator collapses them into one section.

The record stays flat; :mod:`report_module.agentic_traces_renderer` splits it
into metric, health, and configuration tables at render time. No acceptance
criteria are wired, but the scenario's own ``submission_valid`` verdict is
surfaced as ``submission_status`` -- the driver already fails runs it rejects, so
this is the audit trail rather than the gate.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

from report_module.schema import Block

from .base import LLMResultParser


class AIPerfAgenticTracesParser(LLMResultParser):
    kind = "agentic_traces"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        record = dict(raw)
        record.setdefault("kind", self.kind)
        record["model"] = str(raw.get("model_id") or raw.get("model") or "")
        record["device"] = device
        record["timestamp"] = _normalize_timestamp(raw.get("date"))

        # The fork exports request_error_rate as a percentage; carry a fraction
        # alongside it so it can be compared to failed_request_threshold, which
        # is expressed as a ratio (0.10), without unit confusion at the call site.
        error_rate_pct = raw.get("error_rate_pct")
        if isinstance(error_rate_pct, (int, float)):
            record["error_rate"] = float(error_rate_pct) / 100.0

        # A run the scenario rejected must not read like a passing row.
        if raw.get("submission_valid") is False:
            record["submission_status"] = "INVALID: " + ", ".join(
                raw.get("submission_invalid_reasons") or ["unspecified"]
            )
        elif raw.get("submission_valid") is True:
            record["submission_status"] = "valid"

        # Collapse the error breakdown to one readable cell; the full list stays
        # in the raw payload on disk.
        errors = raw.get("error_summary")
        if isinstance(errors, list) and errors:
            record["error_summary"] = ", ".join(
                f"{e.get('count')}x{e.get('type')}"
                for e in errors
                if isinstance(e, Mapping)
            )

        metadata = raw.get("metadata") or {}
        if isinstance(metadata, Mapping):
            # Surface the pinned client revision as a first-class field: an
            # agentic-trace number is only reproducible alongside it.
            for key in ("inferencex_git_ref", "model_id", "mode"):
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


__all__ = ["AIPerfAgenticTracesParser"]
