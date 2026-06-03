# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parser for AIPerf prefix-cache driver output.

Consumes the combined payload produced by
:class:`llm_module.drivers.aiperf_prefix_cache.AIPerfPrefixCacheDriver`
and emits exactly one :class:`report_module.schema.Block` per run.

All prefix-cache Blocks share kind ``aiperf_prefix_cache``; the report
generator's ``_collapse_same_heading_blocks`` then merges every per-run
Block into one section, and the registered renderer
(``report_module.renderers.render_aiperf_prefix_cache``) splits the
collapsed records into three Markdown tables (Synthetic, Trace-Driven,
Uplift) -- mirroring the v1 report layout.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

from report_module.schema import Block

from .base import LLMResultParser


class AIPerfPrefixCacheParser(LLMResultParser):
    kind = "aiperf_prefix_cache"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        """Wrap one driver payload as a Block.

        ``raw`` is the combined dict produced by the driver
        (``_build_payload``): aiperf summary + cache metrics + run
        provenance + optional ``trace_analysis``. We surface a couple of
        derived fields (display-friendly hit-rate percent, theoretical
        hit-rate percent) so the renderer can stay declarative.
        """
        record = dict(raw)
        record.setdefault("kind", self.kind)
        record["model"] = str(raw.get("model_id") or raw.get("model") or "")
        record["device"] = device
        record["timestamp"] = _normalize_timestamp(raw.get("date"))

        # Derived display fields (None-safe). Keep the source fields
        # untouched so CSV / JSON consumers see the canonical values.
        hit_rate = raw.get("prefix_cache_hit_rate")
        if isinstance(hit_rate, (int, float)):
            record["prefix_cache_hit_rate_pct"] = hit_rate * 100.0

        analysis = raw.get("trace_analysis") or {}
        theo = analysis.get("cache_hit_rate") if isinstance(analysis, Mapping) else None
        if isinstance(theo, (int, float)):
            record["trace_theoretical_hit_rate"] = theo
            record["trace_theoretical_hit_rate_pct"] = theo * 100.0

        metadata = raw.get("metadata") or {}
        if isinstance(metadata, Mapping):
            for key in ("trace_name", "synthesis_variant"):
                if key not in record and key in metadata:
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
