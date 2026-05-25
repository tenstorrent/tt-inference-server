# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Markdown renderer for ``aiperf_prefix_cache`` Blocks.

Mirrors the v1 ``benchmarking/prefix_cache_report.py`` report layout:
three sub-tables (Synthetic per-run, Trace-driven per-run, Uplift vs
zero-prefix baseline) plus the metric-definitions footer. The renderer
is fed the collapsed records from every ``aiperf_prefix_cache`` Block
emitted in the sweep -- the report generator's
``_collapse_same_heading_blocks`` merges per-run Blocks (same model +
device -> same Block id) before invoking the renderer once.

Registered with :func:`report_module.renderers.register` at import time
(see the bottom of this module).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module.markdown_table import build_markdown_table
from report_module.renderers import _extract_records, _resolve_model_device, register
from report_module.schema import Block

logger = logging.getLogger(__name__)

NA = "N/A"

# Columns surfaced for synthetic-scenario rows.
DISPLAY_COLUMNS: List[Tuple[str, str]] = [
    ("scenario", "Scenario"),
    ("label", "Label"),
    ("concurrency", "Concur"),
    ("arrival_pattern", "Arrival"),
    ("isl_mean", "ISL mean"),
    ("isl_stddev", "ISL stddev"),
    ("osl_mean", "OSL mean"),
    ("request_count", "N Req"),
    ("prefix_cache_hit_rate_pct", "Cache Hit %"),
    ("mean_ttft_ms", "TTFT Avg (ms)"),
    ("median_ttft_ms", "TTFT P50 (ms)"),
    ("p95_ttft_ms", "TTFT P95 (ms)"),
    ("p99_ttft_ms", "TTFT P99 (ms)"),
    ("mean_tpot_ms", "TPOT Avg (ms)"),
    ("p95_tpot_ms", "TPOT P95 (ms)"),
    ("p99_tpot_ms", "TPOT P99 (ms)"),
    ("mean_itl_ms", "ITL Avg (ms)"),
    ("p95_itl_ms", "ITL P95 (ms)"),
    ("p99_itl_ms", "ITL P99 (ms)"),
    ("mean_e2el_ms", "E2EL Avg (ms)"),
    ("p95_e2el_ms", "E2EL P95 (ms)"),
    ("p99_e2el_ms", "E2EL P99 (ms)"),
    ("output_token_throughput", "Output Tok/s"),
    ("request_throughput", "Req/s"),
]

# Trace-driven rows share the latency columns but swap the ISL/OSL/arrival
# columns for the synthesis parameters and the trace's analyse-derived
# theoretical hit rate.
TRACE_DISPLAY_COLUMNS: List[Tuple[str, str]] = [
    ("scenario", "Scenario"),
    ("label", "Label"),
    ("concurrency", "Concur"),
    ("trace_name", "Trace"),
    ("synthesis_variant", "Synth"),
    ("synthesis_speedup_ratio", "Speedup"),
    ("synthesis_prefix_len_multiplier", "Prefix Len ×"),
    ("synthesis_prefix_root_multiplier", "Prefix Roots"),
    ("synthesis_prompt_len_multiplier", "Prompt Len ×"),
    ("trace_theoretical_hit_rate_pct", "Trace Theo. Hit %"),
    ("prefix_cache_hit_rate_pct", "Measured Hit %"),
    ("mean_ttft_ms", "TTFT Avg (ms)"),
    ("p95_ttft_ms", "TTFT P95 (ms)"),
    ("p99_ttft_ms", "TTFT P99 (ms)"),
    ("mean_tpot_ms", "TPOT Avg (ms)"),
    ("p95_tpot_ms", "TPOT P95 (ms)"),
    ("mean_e2el_ms", "E2EL Avg (ms)"),
    ("p95_e2el_ms", "E2EL P95 (ms)"),
    ("p99_e2el_ms", "E2EL P99 (ms)"),
    ("output_token_throughput", "Output Tok/s"),
    ("request_throughput", "Req/s"),
]

UPLIFT_COLUMNS: List[Tuple[str, str]] = [
    ("scenario", "Scenario"),
    ("label", "Label"),
    ("concurrency", "Concur"),
    ("arrival_pattern", "Arrival"),
    ("isl_mean", "ISL mean"),
    ("cache_hit_rate_pct", "Cache Hit %"),
    ("baseline_ttft_ms", "Baseline TTFT (ms)"),
    ("treatment_ttft_ms", "TTFT (ms)"),
    ("ttft_uplift_pct", "TTFT Δ% vs base"),
    ("baseline_tpot_ms", "Baseline TPOT (ms)"),
    ("treatment_tpot_ms", "TPOT (ms)"),
    ("tpot_uplift_pct", "TPOT Δ% vs base"),
    ("baseline_e2el_ms", "Baseline E2EL (ms)"),
    ("treatment_e2el_ms", "E2EL (ms)"),
    ("e2el_uplift_pct", "E2EL Δ% vs base"),
]


def _format_number(col: str, value: Any) -> str:
    if value is None or value == "":
        return NA
    if isinstance(value, bool):
        return str(value)
    if not isinstance(value, (int, float)):
        return str(value)
    if col == "prefix_cache_hit_rate_pct":
        return f"{value:.1f}"
    if col == "request_throughput":
        return f"{value:.3f}"
    if col == "output_token_throughput":
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _row_to_display(
    row: Mapping[str, Any], columns: Sequence[Tuple[str, str]]
) -> Dict[str, str]:
    return {
        display_header: _format_number(col_name, row.get(col_name))
        for col_name, display_header in columns
    }


def _build_baseline_index(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[Any, Any, Any], Mapping[str, Any]]:
    index: Dict[Tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("scenario") != "baseline":
            continue
        key = (row.get("concurrency"), row.get("arrival_pattern"), row.get("isl_mean"))
        index[key] = row
    return index


def _delta_pct(
    treatment: Optional[float], baseline: Optional[float]
) -> Optional[float]:
    if treatment is None or baseline is None or baseline == 0:
        return None
    return (treatment - baseline) / baseline * 100.0


def _build_uplift_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baselines = _build_baseline_index(rows)
    if not baselines:
        return []
    uplift_rows: List[Dict[str, Any]] = []
    for row in rows:
        scenario = row.get("scenario")
        if scenario == "baseline":
            continue
        key = (row.get("concurrency"), row.get("arrival_pattern"), row.get("isl_mean"))
        base = baselines.get(key)
        if base is None:
            continue
        t_ttft = row.get("mean_ttft_ms")
        t_tpot = row.get("mean_tpot_ms")
        t_e2el = row.get("mean_e2el_ms")
        b_ttft = base.get("mean_ttft_ms")
        b_tpot = base.get("mean_tpot_ms")
        b_e2el = base.get("mean_e2el_ms")
        hit_rate = row.get("prefix_cache_hit_rate")
        uplift_rows.append(
            {
                "scenario": scenario,
                "label": row.get("label"),
                "concurrency": row.get("concurrency"),
                "arrival_pattern": row.get("arrival_pattern"),
                "isl_mean": row.get("isl_mean"),
                "cache_hit_rate_pct": (
                    hit_rate * 100.0 if isinstance(hit_rate, (int, float)) else None
                ),
                "baseline_ttft_ms": b_ttft,
                "treatment_ttft_ms": t_ttft,
                "ttft_uplift_pct": _delta_pct(t_ttft, b_ttft),
                "baseline_tpot_ms": b_tpot,
                "treatment_tpot_ms": t_tpot,
                "tpot_uplift_pct": _delta_pct(t_tpot, b_tpot),
                "baseline_e2el_ms": b_e2el,
                "treatment_e2el_ms": t_e2el,
                "e2el_uplift_pct": _delta_pct(t_e2el, b_e2el),
            }
        )
    return uplift_rows


def _format_uplift_row(row: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col, header in UPLIFT_COLUMNS:
        value = row.get(col)
        if col == "cache_hit_rate_pct":
            out[header] = f"{value:.1f}" if isinstance(value, (int, float)) else NA
        elif col.endswith("_uplift_pct"):
            out[header] = f"{value:+.1f}" if isinstance(value, (int, float)) else NA
        elif col.endswith("_ms") or col == "treatment_ttft_ms":
            out[header] = _format_number("mean_ttft_ms", value)
        else:
            out[header] = NA if value is None else str(value)
    return out


def _sort_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Keep baseline first then sort by scenario / isl / concurrency / arrival."""
    return sorted(
        rows,
        key=lambda r: (
            0 if r.get("scenario") == "baseline" else 1,
            str(r.get("scenario") or ""),
            int(r.get("isl_mean") or 0),
            int(r.get("concurrency") or 0),
            str(r.get("arrival_pattern") or ""),
            str(r.get("label") or ""),
        ),
    )


def render_aiperf_prefix_cache(block: Block, metadata: Mapping[str, Any]) -> str:
    """Render every prefix-cache run into the 3-table report section."""
    records = _extract_records(block)
    if not records:
        return ""
    # Drop the merge-helper bookkeeping that ``_collapse_same_heading_blocks``
    # leaves on each record (model/device duplicates).
    rows = _sort_rows([dict(r) for r in records])
    model, device = _resolve_model_device(block, metadata, rows)

    synthetic_rows = [r for r in rows if r.get("scenario") != "mooncake_trace"]
    trace_rows = [r for r in rows if r.get("scenario") == "mooncake_trace"]

    parts: List[str] = []
    suffix = " on ".join([p for p in (model, device) if p])
    heading_suffix = f" for {suffix}" if suffix else ""
    parts.append(f"### Prefix-Cache Benchmark Results{heading_suffix}")
    parts.append(
        "**Benchmarking Tool:** "
        "[AIPerf](https://github.com/ai-dynamo/aiperf) with the "
        "`--prefix-cache` scenario set. Cache hit-rate is derived from the "
        "vLLM Prometheus counters `vllm:prefix_cache_hits_total` / "
        "`vllm:prefix_cache_queries_total` scraped during each run via "
        "AIPerf's `--server-metrics`."
    )

    if synthetic_rows:
        synth_table = build_markdown_table(
            [_row_to_display(r, DISPLAY_COLUMNS) for r in synthetic_rows]
        )
        if synth_table:
            parts.append(
                f"#### Synthetic Scenarios — Per-run Percentiles\n\n{synth_table}"
            )

    if trace_rows:
        trace_table = build_markdown_table(
            [_row_to_display(r, TRACE_DISPLAY_COLUMNS) for r in trace_rows]
        )
        if trace_table:
            parts.append(
                "#### Trace-Driven (`mooncake_trace`) — Per-run Percentiles\n\n"
                "Each run replays a [mooncake](https://github.com/ai-dynamo/"
                "aiperf/blob/main/docs/tutorials/prefix-synthesis.md) JSONL "
                "trace via `aiperf profile --custom-dataset-type "
                "mooncake_trace`. **Synth** variants apply the "
                "`--synthesis-*` multipliers. **Trace Theo. Hit %** is the "
                "upper bound from `aiperf analyze-trace`; **Measured Hit %** "
                "is the actual vLLM hit-rate observed during the run.\n\n"
                f"{trace_table}"
            )

    if synthetic_rows:
        uplift_rows = _build_uplift_rows(synthetic_rows)
        if uplift_rows:
            uplift_table = build_markdown_table(
                [_format_uplift_row(r) for r in uplift_rows]
            )
            if uplift_table:
                parts.append(
                    "#### Uplift vs Zero-Prefix Baseline (mean metrics)\n\n"
                    "Each treatment row is paired with the `baseline` row "
                    "sharing the same `Concur`, `Arrival`, and `ISL mean`. "
                    "Negative TTFT/TPOT/E2EL deltas indicate latency "
                    "improvements from prefix-cache reuse.\n\n"
                    f"{uplift_table}"
                )

    parts.append(
        "**Metric definitions:**\n"
        "> - **Cache Hit %**: `(hits_delta / queries_delta) * 100` from the "
        "vLLM Prometheus counters across the benchmark window.\n"
        "> - **TTFT / TPOT / ITL / E2EL P50/P95/P99**: AIPerf percentiles "
        "from `profile_export_aiperf.json`.\n"
        "> - **Scenarios**: `shared_system` (100% shared prefix), "
        "`prefix_pool` (tunable reuse via N prefix prompts), `multi_turn` "
        "(organic reuse via conversation history), `mooncake_trace` "
        "(trace-driven via AIPerf prefix-synthesis), `baseline` "
        "(zero-prefix control).\n"
        "> - **Trace Theo. Hit %**: theoretical (infinite-cache) hit rate "
        "reported by `aiperf analyze-trace` before the benchmark runs."
    )

    return "\n\n".join(parts)


# Register at import time so any code path that imports report_module
# picks the renderer up.
register("aiperf_prefix_cache")(render_aiperf_prefix_cache)
