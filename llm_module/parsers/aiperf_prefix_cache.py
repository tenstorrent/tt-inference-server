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
from typing import Any, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser

# Customer SLA targets (used for the per-run PASS/FAIL verdicts). Kept here
# so the renderer and tests share one source of truth. Latency targets are
# milliseconds; output speed is tokens/s/user; hit-rate is a fraction.
SLA_TTFT_P50_MAX_MS = 4_000.0
SLA_TTFT_P90_MAX_MS = 10_000.0
SLA_TTFT_P99_MAX_MS = 35_000.0
SLA_OUTPUT_SPEED_MIN_TPS_PER_USER = 45.0
SLA_HIT_RATE_MIN = 0.90


def _le(measured: Any, target: float) -> Optional[bool]:
    """Return measured <= target, or None when measured is missing/0."""
    if not isinstance(measured, (int, float)) or measured <= 0:
        return None
    return float(measured) <= target


def _ge(measured: Any, target: float) -> Optional[bool]:
    """Return measured >= target, or None when measured is missing/0."""
    if not isinstance(measured, (int, float)) or measured <= 0:
        return None
    return float(measured) >= target


def _compute_sla_checks(raw: Mapping[str, Any]) -> dict:
    """Compute customer SLA PASS/FAIL booleans from one run's metrics.

    Each check is ``True``/``False``, or ``None`` when the underlying
    metric was not captured (e.g. hit-rate when the worker ``/metrics``
    endpoint is unreachable). ``sla_pass`` is the strict overall verdict:
    ``True`` only when every check passed, ``False`` if any check failed,
    and ``None`` when any check is missing (incomplete data).
    """
    checks = {
        "sla_ttft_p50_pass": _le(raw.get("median_ttft_ms"), SLA_TTFT_P50_MAX_MS),
        "sla_ttft_p90_pass": _le(raw.get("p90_ttft_ms"), SLA_TTFT_P90_MAX_MS),
        "sla_ttft_p99_pass": _le(raw.get("p99_ttft_ms"), SLA_TTFT_P99_MAX_MS),
        "sla_output_speed_pass": _ge(
            raw.get("output_token_throughput_per_user"),
            SLA_OUTPUT_SPEED_MIN_TPS_PER_USER,
        ),
        "sla_hit_rate_pass": _ge(raw.get("prefix_cache_hit_rate"), SLA_HIT_RATE_MIN),
    }
    # Disaggregated deployments report prefill and decode as separate caches.
    # Both verdicts are computed and surfaced, but only DECODE gates the
    # overall SLA. Prefill is informational: by design it disproportionately
    # handles cache MISSES, so its hit-rate is structurally low and a
    # >=90% gate on it would spuriously fail healthy deployments.
    for role in ("prefill", "decode"):
        role_rate = raw.get(f"prefix_cache_hit_rate_{role}")
        if role_rate is not None:
            checks[f"sla_hit_rate_{role}_pass"] = _ge(role_rate, SLA_HIT_RATE_MIN)

    # Pick the checks that gate the overall verdict
    gating = dict(checks)
    gating.pop("sla_hit_rate_prefill_pass", None)
    if "sla_hit_rate_decode_pass" in gating:
        gating.pop("sla_hit_rate_pass", None)
    values = list(gating.values())
    if any(v is False for v in values):
        overall: Optional[bool] = False
    elif any(v is None for v in values):
        overall = None
    else:
        overall = True
    checks["sla_pass"] = overall
    return checks


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

        # Per-role display percents (disaggregated deployments). Absent for
        # aggregated runs, so the renderer's role columns stay hidden there.
        for role in ("prefill", "decode"):
            role_rate = raw.get(f"prefix_cache_hit_rate_{role}")
            if isinstance(role_rate, (int, float)):
                record[f"prefix_cache_hit_rate_{role}_pct"] = role_rate * 100.0

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

        # Customer SLA PASS/FAIL verdicts (None-safe; missing metrics ->
        # None). Surfaced as record fields so the renderer can build the
        # "SLA Compliance" table declaratively.
        record.update(_compute_sla_checks(raw))

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
