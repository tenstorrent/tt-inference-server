# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parser for AIPerf spec-decode driver output.

Consumes the combined payload produced by
:class:`llm_module.drivers.aiperf_spec_decode.AIPerfSpecDecodeDriver`
and emits exactly one :class:`report_module.schema.Block` per run.

All spec-decode Blocks share kind ``aiperf_spec_decode``; the report
generator's ``_collapse_same_heading_blocks`` then merges every per-run
Block into one section, and the registered renderer
(``report_module.spec_decode_renderer.render_aiperf_spec_decode``)
renders the collapsed records as the per-run table -- mirroring the v1
report layout.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

from report_module.schema import Block

from .base import LLMResultParser


class AIPerfSpecDecodeParser(LLMResultParser):
    kind = "aiperf_spec_decode"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        """Wrap one driver payload as a Block.

        ``raw`` is the combined dict produced by the driver
        (``_build_payload``): aiperf summary (vllm-bench field names) +
        run provenance + optional nested ``spec_decode_metrics``. We
        lift acceptance metrics to top-level display fields so the
        renderer can stay declarative; the nested block is kept
        untouched for CSV / JSON consumers.
        """
        record = dict(raw)
        record.setdefault("kind", self.kind)
        record["model"] = str(raw.get("model_id") or raw.get("model") or "")
        record["device"] = device
        record["timestamp"] = _normalize_timestamp(raw.get("date"))

        spec_metrics = raw.get("spec_decode_metrics")
        if isinstance(spec_metrics, Mapping):
            for key in ("acceptance_rate", "mean_accepted_length"):
                value = spec_metrics.get(key)
                if key not in record and isinstance(value, (int, float)):
                    record[key] = value

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
