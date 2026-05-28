# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for the spec-decode aiperf raw result.

Builds on :class:`AIPerfParser` for the standard latency / throughput /
sequence-length / count tables, then layers a ``Spec Decode`` section
that carries acceptance-rate and per-position counters scraped from the
vLLM ``/metrics`` endpoint by the driver. The runner sets ``phase``
(``baseline`` or ``spec``) on the returned Block so the report-pairing
helper can match baseline ↔ spec entries.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .aiperf import (
    COUNT_METRICS,
    LATENCY_METRICS,
    SEQUENCE_LENGTH_METRICS,
    THROUGHPUT_METRICS,
    AIPerfParser,
    _metric_table,
    _model_name,
    _run_configuration,
    _telemetry,
    _timestamp,
)
from .base import LLMResultParser


class AIPerfSpecDecodeParser(LLMResultParser):
    kind = "aiperf_spec_decode"

    def parse(  # type: ignore[override]
        self,
        raw: Mapping[str, Any],
        *,
        device: str = "",
        phase: Optional[str] = None,
    ) -> Block:
        spec_metrics: Dict[str, Any] = dict(raw.get("spec_decode_metrics") or {})
        run_spec: Dict[str, Any] = dict(raw.get("spec_decode_run_spec") or {})

        record: Dict[str, Any] = {
            "kind": self.kind,
            "model": _model_name(raw),
            "device": device,
            "timestamp": _timestamp(raw),
            "phase": phase,
            "public_dataset": run_spec.get("public_dataset"),
            "max_concurrency": run_spec.get("max_concurrency"),
            "output_len": run_spec.get("output_len"),
            "num_prompts": run_spec.get("num_prompts"),
            "slug": run_spec.get("slug"),
            "Run Configuration": _run_configuration(raw),
            "Latency Statistics": _metric_table(raw, LATENCY_METRICS),
            "Throughput": _metric_table(raw, THROUGHPUT_METRICS),
            "Sequence Lengths": _metric_table(raw, SEQUENCE_LENGTH_METRICS),
            "Counts & Totals": _metric_table(raw, COUNT_METRICS),
            "Spec Decode": {
                "acceptance_rate": spec_metrics.get("acceptance_rate"),
                "mean_accepted_length": spec_metrics.get("mean_accepted_length"),
                "accepted_tokens": spec_metrics.get("accepted_tokens"),
                "draft_tokens": spec_metrics.get("draft_tokens"),
                "num_drafts": spec_metrics.get("num_drafts"),
                "accepted_per_pos": spec_metrics.get("accepted_per_pos"),
            },
            "Telemetry": _telemetry(raw),
        }
        # Reuse AIPerfParser's wrap so block.targets gets model/device/timestamp.
        return AIPerfParser()._wrap_record(record)


__all__ = ["AIPerfSpecDecodeParser"]
