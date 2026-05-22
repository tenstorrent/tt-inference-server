# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Report generator for AIPerf prefix-caching benchmark runs.

Files written by `benchmarking/run_aiperf_benchmarks.py::save_prefix_cache_result`
match the glob pattern::

    workflow_logs/benchmarks_output/aiperf_prefix_cache_<model_id>_<ts>_<scenario>_<label>.json

This module discovers those files, builds a Markdown table that pairs each
reuse scenario with the matching ``baseline`` row, dumps a CSV with the full
metric set, and returns the artifact paths to ``workflows/run_reports.py`` so
the prefix-cache section ends up in the unified release report.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarking.summary_report import get_markdown_table

logger = logging.getLogger(__name__)

NA = "N/A"

# Columns surfaced in the per-run Markdown table for the synthetic
# scenarios (shared_system / prefix_pool / multi_turn / baseline).
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
    # Parser writes `median_ttft_ms` (consistent with the rest of the
    # benchmark suite, e.g. workflows/run_reports.py and
    # benchmarking/run_guidellm_benchmarks.py). The previous lookup of
    # `p50_ttft_ms` was a typo that silently rendered N/A every time.
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

# Trace-driven scenarios share the latency columns but swap the
# ISL/OSL/arrival columns for the synthesis parameters and the trace's
# analyse-derived theoretical hit rate.
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


def _load_prefix_cache_files(model_id: str, benchmarks_output_dir: str) -> List[Dict[str, Any]]:
    pattern = f"{benchmarks_output_dir}/aiperf_prefix_cache_{model_id}_*.json"
    files = glob(pattern)
    files.sort()
    rows: List[Dict[str, Any]] = []
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not parse {filepath}: {e}")
            continue
        if data.get("task_type") != "prefix_cache":
            continue
        data["__filepath"] = filepath
        rows.append(data)
    return rows


def _dedupe_latest(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the latest result per (scenario, label, concurrency, arrival_pattern, isl_mean).

    Filenames embed a timestamp, so we sort ascending then keep the last
    occurrence of each config key.
    """
    keyed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r.get("__filepath", "")):
        key = (
            row.get("scenario"),
            row.get("label"),
            row.get("concurrency"),
            row.get("arrival_pattern"),
            row.get("isl_mean"),
        )
        keyed[key] = row
    return list(keyed.values())


def _format_number(col: str, value: Any) -> str:
    if value is None or value == "":
        return NA
    if isinstance(value, bool):
        return str(value)
    if not isinstance(value, (int, float)):
        return str(value)
    if col == "prefix_cache_hit_rate_pct":
        return f"{value:.1f}"
    if col in ("request_throughput",):
        return f"{value:.3f}"
    if col in ("output_token_throughput",):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _enrich_for_display(row: Dict[str, Any]) -> Dict[str, Any]:
    hit_rate = row.get("prefix_cache_hit_rate")
    if isinstance(hit_rate, (int, float)):
        row["prefix_cache_hit_rate_pct"] = hit_rate * 100.0

    # Surface trace-driven metadata at the top level for the trace table.
    metadata = row.get("metadata") or {}
    if metadata:
        for key in ("trace_name", "synthesis_variant"):
            if key not in row and key in metadata:
                row[key] = metadata[key]

    # Pull the analyze-trace theoretical hit rate up so the report can show
    # measured-vs-theoretical at a glance.
    analysis = row.get("trace_analysis") or {}
    theo = analysis.get("cache_hit_rate")
    if isinstance(theo, (int, float)):
        row["trace_theoretical_hit_rate"] = theo
        row["trace_theoretical_hit_rate_pct"] = theo * 100.0
    return row


def _build_baseline_index(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[Any, Any, Any], Dict[str, Any]]:
    """Map (concurrency, arrival_pattern, isl_mean) -> baseline row for pairing."""
    index: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for row in rows:
        if row.get("scenario") != "baseline":
            continue
        key = (row.get("concurrency"), row.get("arrival_pattern"), row.get("isl_mean"))
        index[key] = row
    return index


def _row_to_display_dict(
    row: Dict[str, Any],
    columns: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, str]:
    cols = columns if columns is not None else DISPLAY_COLUMNS
    return {
        display_header: _format_number(col_name, row.get(col_name))
        for col_name, display_header in cols
    }


def _delta_pct(treatment: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if treatment is None or baseline is None:
        return None
    if baseline == 0:
        return None
    return (treatment - baseline) / baseline * 100.0


def _build_uplift_table(rows: List[Dict[str, Any]]) -> str:
    """Render an uplift table comparing each reuse scenario against its baseline."""
    baselines = _build_baseline_index(rows)
    if not baselines:
        return ""

    uplift_cols: List[Tuple[str, str]] = [
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

    display_dicts: List[Dict[str, str]] = []
    for row in rows:
        scenario = row.get("scenario")
        if scenario == "baseline":
            continue
        key = (row.get("concurrency"), row.get("arrival_pattern"), row.get("isl_mean"))
        base = baselines.get(key)
        if base is None:
            continue
        treatment_ttft = row.get("mean_ttft_ms")
        treatment_tpot = row.get("mean_tpot_ms")
        treatment_e2el = row.get("mean_e2el_ms")
        baseline_ttft = base.get("mean_ttft_ms")
        baseline_tpot = base.get("mean_tpot_ms")
        baseline_e2el = base.get("mean_e2el_ms")
        hit_rate = row.get("prefix_cache_hit_rate")
        formatted = {
            "scenario": scenario,
            "label": row.get("label", NA),
            "concurrency": row.get("concurrency", NA),
            "arrival_pattern": row.get("arrival_pattern", NA),
            "isl_mean": row.get("isl_mean", NA),
            "cache_hit_rate_pct": (
                f"{hit_rate * 100:.1f}" if isinstance(hit_rate, (int, float)) else NA
            ),
            "baseline_ttft_ms": _format_number("mean_ttft_ms", baseline_ttft),
            "treatment_ttft_ms": _format_number("mean_ttft_ms", treatment_ttft),
            "ttft_uplift_pct": (
                f"{_delta_pct(treatment_ttft, baseline_ttft):+.1f}"
                if _delta_pct(treatment_ttft, baseline_ttft) is not None
                else NA
            ),
            "baseline_tpot_ms": _format_number("mean_tpot_ms", baseline_tpot),
            "treatment_tpot_ms": _format_number("mean_tpot_ms", treatment_tpot),
            "tpot_uplift_pct": (
                f"{_delta_pct(treatment_tpot, baseline_tpot):+.1f}"
                if _delta_pct(treatment_tpot, baseline_tpot) is not None
                else NA
            ),
            "baseline_e2el_ms": _format_number("mean_e2el_ms", baseline_e2el),
            "treatment_e2el_ms": _format_number("mean_e2el_ms", treatment_e2el),
            "e2el_uplift_pct": (
                f"{_delta_pct(treatment_e2el, baseline_e2el):+.1f}"
                if _delta_pct(treatment_e2el, baseline_e2el) is not None
                else NA
            ),
        }
        display_dicts.append({h: str(formatted.get(c, NA)) for c, h in uplift_cols})
    return get_markdown_table(display_dicts)


def _write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Pick a stable header ordering: scalar fields first, metadata last.
    header_fields: List[str] = []
    seen = set()
    preferred = [
        "scenario",
        "label",
        "concurrency",
        "arrival_pattern",
        "arrival_smoothness",
        "request_rate",
        "isl_mean",
        "isl_stddev",
        "osl_mean",
        "osl_stddev",
        "request_count",
        "shared_system_prompt_length",
        "num_prefix_prompts",
        "prefix_prompt_length",
        "conversation_num",
        "conversation_turn_mean",
        "trace_input_file",
        "custom_dataset_type",
        "fixed_schedule",
        "block_size",
        "synthesis_speedup_ratio",
        "synthesis_prefix_len_multiplier",
        "synthesis_prefix_root_multiplier",
        "synthesis_prompt_len_multiplier",
        "synthesis_max_isl",
        "synthesis_max_osl",
        "trace_theoretical_hit_rate",
        "prefix_cache_hit_rate",
        "prefix_cache_hits_delta",
        "prefix_cache_queries_delta",
        "prefix_cache_hits_final",
        "prefix_cache_queries_final",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
        "output_token_throughput",
        "request_throughput",
    ]
    for f in preferred:
        header_fields.append(f)
        seen.add(f)
    for row in rows:
        for k in row.keys():
            if k not in seen and not k.startswith("__"):
                header_fields.append(k)
                seen.add(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header_fields)
        for row in rows:
            writer.writerow([_csv_value(row.get(h, "")) for h in header_fields])
    logger.info(f"Prefix-cache CSV saved to: {csv_path}")


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def generate_prefix_cache_report(
    model_id: str,
    model_name: str,
    device: str,
    benchmarks_output_dir: str,
    output_dir: Path,
    report_id: str,
) -> Tuple[str, List[Dict[str, Any]], Optional[Path], Optional[Path]]:
    """Build the prefix-cache report Markdown + CSV for a single model.

    Returns ``(release_str, raw_rows, md_path, csv_path)``. When no
    prefix-cache benchmark files are found, returns empty values so the
    caller can skip the section.
    """
    raw_rows = _load_prefix_cache_files(model_id, benchmarks_output_dir)
    if not raw_rows:
        logger.info(
            f"No prefix-cache benchmark files found for {model_id} in "
            f"{benchmarks_output_dir}"
        )
        return "", [], None, None

    deduped = _dedupe_latest(raw_rows)
    deduped.sort(
        key=lambda r: (
            # baseline first for visual pairing
            0 if r.get("scenario") == "baseline" else 1,
            r.get("scenario") or "",
            r.get("isl_mean") or 0,
            r.get("concurrency") or 0,
            r.get("arrival_pattern") or "",
            r.get("label") or "",
        )
    )

    enriched = [_enrich_for_display(dict(r)) for r in deduped]

    # Split rows: synthetic-scenario rows vs trace-driven rows.
    synthetic_rows = [r for r in enriched if r.get("scenario") != "mooncake_trace"]
    trace_rows = [r for r in enriched if r.get("scenario") == "mooncake_trace"]

    # Per-run percentile tables.
    synthetic_table_md = (
        get_markdown_table([_row_to_display_dict(row) for row in synthetic_rows])
        if synthetic_rows
        else ""
    )
    trace_table_md = (
        get_markdown_table(
            [_row_to_display_dict(row, TRACE_DISPLAY_COLUMNS) for row in trace_rows]
        )
        if trace_rows
        else ""
    )

    # Baseline-vs-treatment uplift table (only meaningful for synthetic
    # scenarios where a matched zero-prefix baseline exists in the sweep).
    uplift_md = _build_uplift_table(deduped) if synthetic_rows else ""

    release_str = (
        f"### Prefix-Cache Benchmark Results for {model_name} on {device}\n\n"
        "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf) "
        "with the `--prefix-cache` scenario set.\n\n"
        "Cache hit-rate is derived from the vLLM Prometheus counters "
        "`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` "
        "scraped during each run via AIPerf's `--server-metrics`.\n\n"
    )
    if synthetic_table_md:
        release_str += (
            "#### Synthetic Scenarios — Per-run Percentiles\n\n"
            f"{synthetic_table_md}\n\n"
        )
    if trace_table_md:
        release_str += (
            "#### Trace-Driven (`mooncake_trace`) — Per-run Percentiles\n\n"
            "Each run replays a [mooncake](https://github.com/ai-dynamo/aiperf"
            "/blob/main/docs/tutorials/prefix-synthesis.md) JSONL trace via "
            "`aiperf profile --custom-dataset-type mooncake_trace`. **Synth** "
            "variants apply the `--synthesis-*` multipliers to scale prefix "
            "lengths, prompt lengths, request rate and prefix-root diversity. "
            "**Trace Theo. Hit %** is the upper bound reported by "
            "`aiperf analyze-trace`; **Measured Hit %** is the actual vLLM "
            "prefix-cache hit-rate observed during the run.\n\n"
            f"{trace_table_md}\n\n"
        )
    if uplift_md:
        release_str += (
            "#### Uplift vs Zero-Prefix Baseline (mean metrics)\n\n"
            "Negative TTFT/TPOT/E2EL deltas indicate latency improvements "
            "from prefix-cache reuse.\n\n"
            f"{uplift_md}\n\n"
        )
    release_str += (
        "**Metric definitions:**\n"
        "> - **Cache Hit %**: `(hits_delta / queries_delta) * 100` from the "
        "vLLM Prometheus counters across the benchmark window.\n"
        "> - **TTFT / TPOT / ITL / E2EL P50/P95/P99**: AIPerf percentiles from "
        "`profile_export_aiperf.json`.\n"
        "> - **Scenarios**: `shared_system` (100% shared prefix), `prefix_pool` "
        "(tunable reuse via N prefix prompts), `multi_turn` (organic reuse via "
        "conversation history), `mooncake_trace` (trace-driven via AIPerf "
        "prefix-synthesis), `baseline` (zero-prefix control).\n"
        "> - **Trace Theo. Hit %**: theoretical (infinite-cache) hit rate "
        "reported by `aiperf analyze-trace` before the benchmark runs.\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"aiperf_prefix_cache_display_{report_id}.md"
    md_path.write_text(release_str, encoding="utf-8")
    logger.info(f"Prefix-cache report saved to: {md_path}")

    csv_path = output_dir / "data" / f"aiperf_prefix_cache_stats_{report_id}.csv"
    _write_csv(deduped, csv_path)

    return release_str, deduped, md_path, csv_path


# Convenience filename regex (used by downstream callers that want to filter
# the unified benchmark output dir without re-implementing the discovery).
PREFIX_CACHE_FILENAME_RE = re.compile(
    r"^aiperf_prefix_cache_(?P<model_id>.+?)_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_"
    r"(?P<scenario>shared_system|prefix_pool|multi_turn|baseline|mooncake_trace)_"
    r"(?P<label>.+)\.json$"
)
