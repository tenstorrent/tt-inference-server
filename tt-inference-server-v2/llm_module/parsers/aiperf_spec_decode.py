# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parser for AIPerf spec-decode driver output.

Consumes the combined payload produced by
:class:`llm_module.drivers.aiperf_spec_decode.AIPerfSpecDecodeDriver`
and emits exactly one :class:`report_module.schema.Block` per run.

All spec-decode Blocks share kind ``aiperf_spec_decode``; the report
generator's ``_collapse_same_heading_blocks`` then merges every per-run
Block into one section, which the generic table renderer
(``report_module.renderers.render_generic_table``) renders as the
per-run sweep table -- mirroring the v1 report layout. This parser owns
the column selection and ordering by emitting a narrow, ordered record;
display headers and precision come from ``report_module.display``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

from report_module.schema import Block

from .base import LLMResultParser

SECTION_TITLE = "Speculative Decoding Benchmark Results"

DISPLAY_FIELDS = (
    "public_dataset",
    "output_len",
    "max_concurrency",
    "completed",
    "acceptance_rate",
    "mean_accepted_length",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "mean_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "output_throughput",
    "total_token_throughput",
)


class AIPerfSpecDecodeParser(LLMResultParser):
    kind = "aiperf_spec_decode"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        """Wrap one driver payload as a narrow, ordered Block.

        ``raw`` is the combined dict produced by the driver
        (``_build_payload``): aiperf summary (vllm-bench field names) +
        run provenance + optional nested ``spec_decode_metrics``. We
        select just the display fields (lifting the acceptance metrics
        out of the nested block) and emit them in render order, so the
        generic renderer needs no spec-decode-specific knowledge. The
        raw payload remains available in the run's on-disk artefacts.
        """
        spec_metrics = raw.get("spec_decode_metrics")
        spec_metrics = spec_metrics if isinstance(spec_metrics, Mapping) else {}

        def _value(field: str) -> Any:
            if field in raw:
                return raw.get(field)
            return spec_metrics.get(field)

        record = {
            "kind": self.kind,
            "model": str(raw.get("model_id") or raw.get("model") or ""),
            "device": device,
            "timestamp": _normalize_timestamp(raw.get("date")),
        }
        for field in DISPLAY_FIELDS:
            record[field] = _value(field)

        return self._wrap_record(record, title=SECTION_TITLE)


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
