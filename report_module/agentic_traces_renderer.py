# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Markdown renderer for ``agentic_traces`` Blocks.

The driver payload is one flat record per run that mixes four unrelated
concerns: latency, throughput/load, the run's health and validity signals, and
the config echo describing how the run was invoked. The generic renderer turns
that into a single very wide table where the config columns and the artifact
path precede the numbers anyone actually came for, so this renderer splits it
into four tables and drops the fields that only belong in the raw JSON.

The config table is transposed (one row per field) because it is the same
handful of values for every run and its values are long, so as columns they set
the width of the whole report.

Registered with :func:`report_module.renderers.register` at import time (see the
bottom of this module).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module.markdown_table import build_markdown_table
from report_module.renderers import _extract_records, _resolve_model_device, register
from report_module.schema import Block

logger = logging.getLogger(__name__)

NA = "N/A"

# The run identity carried by every table so rows can be lined up across them.
# The generated `label` is long and redundant with these three, so it is only
# shown in the configuration table.
IDENTITY_COLUMNS: List[Tuple[str, str]] = [
    ("trace_source", "Source"),
    ("concurrency", "Concur"),
    ("mode", "Mode"),
]

# Units live in the table caption rather than in every header, which buys back
# the width the added percentiles cost.
LATENCY_COLUMNS: List[Tuple[str, str]] = [
    ("mean_ttft_ms", "TTFT Avg"),
    ("median_ttft_ms", "TTFT P50"),
    ("p90_ttft_ms", "TTFT P90"),
    ("p99_ttft_ms", "TTFT P99"),
    ("std_ttft_ms", "TTFT Std"),
    ("mean_ttfot_ms", "TTFOT Avg"),
    ("mean_tpot_ms", "TPOT Avg"),
    ("p90_tpot_ms", "TPOT P90"),
    ("p99_tpot_ms", "TPOT P99"),
    ("mean_e2el_ms", "E2EL Avg"),
    ("median_e2el_ms", "E2EL P50"),
    ("p99_e2el_ms", "E2EL P99"),
    ("mean_effective_latency_ms", "CO-Adj E2EL Avg"),
    ("p99_effective_latency_ms", "CO-Adj E2EL P99"),
    ("p50_adj_ttft_ms", "Err-Adj TTFT P50"),
    ("p90_adj_ttft_ms", "Err-Adj TTFT P90"),
    ("p50_adj_e2el_ms", "Err-Adj E2EL P50"),
    ("p90_adj_e2el_ms", "Err-Adj E2EL P90"),
]

THROUGHPUT_COLUMNS: List[Tuple[str, str]] = [
    ("output_token_throughput_per_user", "Out Tok/s/User"),
    ("e2e_output_token_throughput_per_user", "E2E Tok/s/User"),
    ("output_token_throughput", "Output Tok/s"),
    ("total_token_throughput", "Total Tok/s"),
    ("request_throughput", "Req/s"),
    ("effective_prefill_throughput", "Prefill Tok/s"),
    ("active_prefill_throughput", "Prefill Tok/s (Active)"),
    ("effective_decode_throughput", "Decode Tok/s"),
    ("active_decode_throughput", "Decode Tok/s (Active)"),
    ("effective_concurrency", "Eff. Concur"),
    ("mean_tokens_in_flight", "Tok In Flight Avg"),
    ("max_tokens_in_flight", "Tok In Flight Max"),
    ("mean_isl", "ISL Mean"),
    ("mean_osl", "OSL Mean"),
    ("total_input_tokens", "Total In Tok"),
    ("total_output_tokens", "Total Out Tok"),
]

HEALTH_COLUMNS: List[Tuple[str, str]] = [
    ("submission_status", "Submission"),
    ("completed", "Reqs OK"),
    ("error_request_count", "Errors"),
    ("error_rate_pct", "Error %"),
    ("context_overflow_count", "Ctx Overflow"),
    ("osl_mismatch_count", "OSL Mismatch"),
    ("osl_mismatch_diff_pct", "OSL Diff %"),
    ("measured_prefix_cache_hit_pct", "Cache Hit %"),
    ("theoretical_prefix_cache_hit_pct", "Theo. Cache Hit %"),
    ("credit_drop_count", "Credit Drops"),
    ("connection_reuse_rate", "Conn Reuse"),
    ("branch_children_spawned", "Branches"),
    ("branch_children_errored", "Br. Errored"),
    ("branch_children_truncated", "Br. Truncated"),
    ("branch_children_delayed", "Br. Delayed"),
    ("branch_parents_failed_due_to_child_error", "Br. Parent Failed"),
    ("branch_joins_suppressed", "Br. Joins Dropped"),
    ("measured_benchmark_duration", "Measured (s)"),
    ("error_summary", "Top Errors"),
]

# Rendered transposed: these are the row labels, not headers.
CONFIG_FIELDS: List[Tuple[str, str]] = [
    ("label", "Run"),
    ("scenario", "Scenario"),
    ("public_dataset", "Dataset"),
    ("dataset_hf_name", "Dataset Repo"),
    ("dataset_num_entries", "Trace Pool"),
    ("benchmark_duration", "Profiling Window (s)"),
    ("warmup_requests_per_lane", "Warmup (req/lane)"),
    ("warmup_grace_period", "Warmup Grace (s)"),
    ("max_context_length", "Max Context"),
    ("failed_request_threshold", "Fail Threshold"),
    ("trajectory_start_min_ratio", "Trajectory Start Min"),
    ("trajectory_start_max_ratio", "Trajectory Start Max"),
    ("random_seed", "Random Seed"),
    ("inferencex_git_ref", "Client Ref"),
    ("aiperf_version", "AIPerf Version"),
    ("benchmark_id", "Benchmark ID"),
    ("run_started_at", "Started"),
    ("run_ended_at", "Ended"),
]

# Columns whose float formatting differs from the 1-decimal default.
_THROUGHPUT_KEYS = frozenset(
    {
        "output_token_throughput",
        "output_token_throughput_per_user",
        "e2e_output_token_throughput_per_user",
        "total_token_throughput",
        "effective_prefill_throughput",
        "active_prefill_throughput",
        "effective_decode_throughput",
        "active_decode_throughput",
    }
)
_CACHE_HIT_KEYS = frozenset(
    {
        "measured_prefix_cache_hit_pct",
        "theoretical_prefix_cache_hit_pct",
    }
)
_INT_KEYS = frozenset(
    {
        "completed",
        "error_request_count",
        "context_overflow_count",
        "osl_mismatch_count",
        "credit_drop_count",
        "branch_children_spawned",
        "branch_children_errored",
        "branch_children_truncated",
        "branch_children_delayed",
        "branch_parents_failed_due_to_child_error",
        "branch_joins_suppressed",
        "concurrency",
        "max_context_length",
        "random_seed",
        "dataset_num_entries",
        "benchmark_duration",
        "warmup_requests_per_lane",
        "warmup_grace_period",
        "measured_benchmark_duration",
        "mean_isl",
        "mean_osl",
        "total_input_tokens",
        "total_output_tokens",
        "mean_tokens_in_flight",
        "max_tokens_in_flight",
    }
)

# Kept even when empty: a missing verdict is itself the finding, so it has to
# read as N/A rather than vanish.
_ALWAYS_COLUMNS = frozenset({"submission_status"})

# Ratios and thresholds that the 1-decimal default would silently round to a
# different value (0.25 -> "0.2", a 0.05 threshold -> "0.1").
_RATIO_KEYS = frozenset(
    {
        "trajectory_start_min_ratio",
        "trajectory_start_max_ratio",
        "failed_request_threshold",
        "slice_duration",
        "effective_concurrency",
        "connection_reuse_rate",
    }
)


def _format_value(key: str, value: Any) -> str:
    """Format one cell, avoiding the scientific notation the generic table emits."""
    if value is None or value == "":
        return NA
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)):
        return str(value)
    if key in _INT_KEYS:
        return f"{round(value):,}"
    if key in _RATIO_KEYS:
        return f"{value:.2f}"
    if key == "request_throughput":
        return f"{value:.3f}"
    if key in ("error_rate_pct", "osl_mismatch_diff_pct"):
        return f"{value:.2f}"
    if key in _THROUGHPUT_KEYS or key in _CACHE_HIT_KEYS:
        return f"{value:,.2f}"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def _submission_cell(row: Mapping[str, Any]) -> str:
    """Render the scenario's validity verdict, defaulting to unknown.

    A run the scenario rejected must not read like a passing one, so the reason
    codes stay in the cell rather than being collapsed to a bare FAIL.
    """
    valid = row.get("submission_valid")
    if valid is False:
        reasons = row.get("submission_invalid_reasons")
        detail = ", ".join(reasons) if reasons else "unspecified"
        return f"❌ INVALID ({detail})"
    if valid is True:
        return "✅ valid"
    status = row.get("submission_status")
    return str(status) if status else NA


def _cell(row: Mapping[str, Any], key: str) -> str:
    if key == "submission_status":
        return _submission_cell(row)
    return _format_value(key, row.get(key))


def _table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]]
) -> str:
    """Render one metric table, identity columns first.

    Columns the client did not emit for any run are dropped rather than filled
    with N/A: the error-adjusted percentiles are absent exactly when a run had
    no errors, and carrying them as empty columns costs width for nothing.
    """
    present = [
        (key, header)
        for key, header in columns
        if key in _ALWAYS_COLUMNS or any(_cell(row, key) != NA for row in rows)
    ]
    if not present:
        return ""
    full = list(IDENTITY_COLUMNS) + present
    return build_markdown_table(
        [{header: _cell(row, key) for key, header in full} for row in rows]
    )


def _vertical_table(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[Tuple[str, str]]
) -> str:
    """Render one column per run, one row per field."""
    if not rows:
        return ""
    headers = [_run_header(row) for row in rows]
    table_rows: List[Dict[str, str]] = []
    for key, label in fields:
        cells = [_cell(row, key) for row in rows]
        if all(cell == NA for cell in cells):
            continue
        entry: Dict[str, str] = {"Field": label}
        for header, cell in zip(headers, cells):
            entry[header] = cell
        table_rows.append(entry)
    return build_markdown_table(table_rows)


def _run_header(row: Mapping[str, Any]) -> str:
    """Short identity for a run, used as the column header when transposed."""
    parts = [str(row.get("trace_source") or "run")]
    concurrency = row.get("concurrency")
    if concurrency:
        parts.append(f"c{_as_int(concurrency)}")
    mode = row.get("mode")
    if mode:
        parts.append(str(mode))
    return " ".join(parts)


def _sort_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("trace_source") or ""),
            _as_int(r.get("concurrency")),
            str(r.get("mode") or ""),
        ),
    )


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _resolve_client_ref(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for row in rows:
        ref = row.get("inferencex_git_ref")
        if isinstance(ref, str) and ref:
            return ref
    return None


def render_agentic_traces(block: Block, metadata: Mapping[str, Any]) -> str:
    """Render every agentic-trace run into the report section."""
    records = _extract_records(block)
    if not records:
        return ""
    rows = _sort_rows([dict(r) for r in records])
    model, device = _resolve_model_device(block, metadata, rows)

    suffix = " on ".join([p for p in (model, device) if p])
    heading_suffix = f" for {suffix}" if suffix else ""
    parts: List[str] = [f"### Agentic Trace Replay{heading_suffix}"]

    client_ref = _resolve_client_ref(rows)
    ref_note = f" pinned at `{client_ref}`" if client_ref else ""
    parts.append(
        "**Benchmarking Tool:** the AIPerf fork vendored in "
        "[InferenceX](https://github.com/SemiAnalysisAI/InferenceX)"
        f"{ref_note}, driving the `inferencex-agentx-mvp` scenario. Each run "
        "replays recorded multi-turn agentic coding traces with their original "
        "timing, including subagent fan-out, so the load shape is the trace's "
        "rather than a synthetic ISL/OSL grid. Numbers are only comparable "
        "across runs on the same client revision and dataset."
    )

    latency = _table(rows, LATENCY_COLUMNS)
    if latency:
        parts.append(f"#### Per-run Latency (ms)\n\n{latency}")

    throughput = _table(rows, THROUGHPUT_COLUMNS)
    if throughput:
        parts.append(f"#### Per-run Throughput & Load\n\n{throughput}")

    health = _table(rows, HEALTH_COLUMNS)
    if health:
        parts.append(
            "#### Run Health & Validity\n\n"
            "**Submission** is the scenario's own verdict, not ours: it folds "
            "static scenario-lock violations, a context-overflow rate above "
            "1%, and early cancellation into one flag. An invalid run is "
            "failed by the driver and is not a reportable number.\n\n"
            f"{health}"
        )

    config = _vertical_table(rows, CONFIG_FIELDS)
    if config:
        parts.append(
            "#### Run Configuration\n\n"
            "Echoed from "
            "`reference_config/agentic_traces/agentic_traces_config.py` for the "
            "ModelSpec, plus the client's own run identifiers, so a table above "
            f"can be traced back to what produced it.\n\n{config}"
        )

    parts.append(
        "**Metric definitions:**\n"
        "> - **TTFT / TPOT / E2EL**: AIPerf time-to-first-token, inter-token "
        "latency, and end-to-end request latency from "
        "`profile_export_aiperf.json`. Long-context agentic prefill makes TTFT "
        "the headline metric, so its spread is reported.\n"
        "> - **TTFOT**: time to the first token the user actually sees. On a "
        "reasoning model the gap to TTFT is time spent emitting reasoning "
        "tokens.\n"
        "> - **CO-Adj E2EL**: latency corrected for coordinated omission, so it "
        "includes the time a request spent waiting to be dispatched. It tracks "
        "E2EL until requests start queueing.\n"
        "> - **Err-Adj TTFT / E2EL**: percentiles with failed requests folded "
        "in, so a run that got fast by erroring out cannot look good. Omitted "
        "when a run had no errors.\n"
        "> - **Out Tok/s/User**: decode speed while a request is streaming. "
        "**E2E Tok/s/User** divides by whole-request wall clock instead, so it "
        "includes prefill and is the user-visible speed of an agentic turn; "
        "expect it to be several times lower.\n"
        "> - **Prefill / Decode Tok/s**: the phase split of serving throughput, "
        "averaged over the whole run. The **(Active)** variants only count the "
        "windows where that phase was actually working, so the ratio between "
        "them is the phase's duty cycle.\n"
        "> - **Eff. Concur**: concurrency actually in flight. Trace replay "
        "honours the recorded think time, so this sits below the requested "
        "**Concur** by design; a large shortfall means the intended load was "
        "never applied.\n"
        "> - **Tok In Flight**: KV footprint across in-flight requests. Compare "
        "the max against **Max Context**.\n"
        "> - **Reqs OK / Errors**: successful requests (the sample size behind "
        "every latency stat) versus failed ones. **Error %** is AIPerf's "
        "`request_error_rate`; the run aborts if it exceeds **Fail "
        "Threshold**.\n"
        "> - **Ctx Overflow**: requests whose trace exceeded **Max Context**. "
        "Non-zero means the replay was truncated and no longer matches what was "
        "recorded; above 1% the scenario invalidates the submission.\n"
        "> - **OSL Mismatch / OSL Diff %**: responses whose length did not "
        "match the trace's recorded output, and by how much. Non-zero means the "
        "replayed conversation diverged from what was recorded.\n"
        "> - **Cache Hit %**: measured from the serving engine's own counters "
        "over the profiling window, so the cache-priming warmup is excluded. "
        "**Theo. Cache Hit %** is the reuse inherent to the traces, i.e. the "
        "upper bound the engine was offered. A measured rate well below it means "
        "the cache was evicting reuse the workload had available. Omitted when "
        "the server exposes no such counters.\n"
        "> - **Credit Drops**: requests the load generator could not dispatch "
        "on the trace's schedule. Non-zero means the recorded timing was not "
        "reproduced, usually because the server was saturated.\n"
        "> - **Conn Reuse**: fraction of requests served on a reused "
        "connection. Below 1.0 means connections were being torn down, which is "
        "what shows up as `ServerDisconnectedError`.\n"
        "> - **Branches**: subagent trajectories spawned during replay. Any "
        "errored, truncated, or delayed child, any parent failed by its child, "
        "or any dropped join means the fan-out did not replay as recorded.\n"
        "> - **Measured (s)**: actual profiling wall clock, which should track "
        "**Profiling Window (s)**; a large shortfall means the run ended early."
    )

    return "\n\n".join(parts)


# Register at import time so any code path that imports report_module picks the
# renderer up.
register("agentic_traces")(render_agentic_traces)
