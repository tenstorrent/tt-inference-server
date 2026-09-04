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

Two harnesses share this kind -- the InferenceX AIPerf fork and SwarmOne's
swo-bench -- and a sweep can mix them. The tables are common ground (a column no
run emitted is dropped, so each source only pays width for its own metrics),
while the surrounding prose is selected per source: describing swo-bench numbers
as AIPerf's, or defining metrics the report does not contain, is worse than
silence.

Registered with :func:`report_module.renderers.register` at import time (see the
bottom of this module).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module.markdown_table import build_markdown_table
from report_module.renderers import _extract_records, _resolve_model_device, register
from report_module.schema import Block

logger = logging.getLogger(__name__)

NA = "N/A"

# ``TraceSource`` values, duplicated rather than imported: report_module renders
# whatever a Block carries and does not otherwise depend on reference_config.
INFERENCEX_SOURCE = "inferencex_agentx"
SWARMONE_SOURCE = "swarmone"

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
    ("p90_e2el_ms", "E2EL P90"),
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
    ("median_output_token_throughput_per_user", "Out Tok/s/User P50"),
    ("mean_e2e_norm_intvty", "E2E Norm Intvty Avg"),
    ("p75_e2e_norm_intvty", "E2E Norm Intvty P75"),
    ("p90_e2e_norm_intvty", "E2E Norm Intvty P90"),
    ("output_token_throughput", "Output Tok/s"),
    ("total_token_throughput", "Total Tok/s"),
    ("active_throughput_tok_per_s", "Total Tok/s (Active)"),
    ("request_throughput", "Req/s"),
    ("effective_prefill_throughput", "Prefill Tok/s"),
    ("active_prefill_throughput", "Prefill Tok/s (Active)"),
    # swo-bench measures prefill per request against the prompt actually sent,
    # so a cache hit shows up as a very high rate; it is not comparable to
    # AIPerf's run-averaged effective_prefill_throughput above.
    ("prefill_tok_per_sec_mean", "Prefill Tok/s/Req Avg"),
    ("prefill_tok_per_sec_p50", "Prefill Tok/s/Req P50"),
    ("prefill_tok_per_sec_p90", "Prefill Tok/s/Req P90"),
    ("effective_decode_throughput", "Decode Tok/s"),
    ("active_decode_throughput", "Decode Tok/s (Active)"),
    ("effective_concurrency", "Eff. Concur"),
    ("concurrency_mean", "Concur Avg"),
    ("concurrency_peak", "Concur Peak"),
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
    ("ready_starved_events", "Ready Starved"),
    ("pace_idle_ms", "Pace Idle (ms)"),
    ("tool_idle_ms", "Tool Idle (ms)"),
    ("measured_benchmark_duration", "Measured (s)"),
    ("error_summary", "Top Errors"),
]

# Rendered transposed: these are the row labels, not headers.
CONFIG_FIELDS: List[Tuple[str, str]] = [
    ("label", "Run"),
    ("scenario", "Scenario"),
    ("task", "Task"),
    ("public_dataset", "Dataset"),
    ("dataset_hf_name", "Dataset Repo"),
    ("dataset_num_entries", "Trace Pool"),
    ("benchmark_duration", "Profiling Window (s)"),
    ("warmup_requests_per_lane", "Warmup (req/lane)"),
    ("warmup_grace_period", "Warmup Grace (s)"),
    ("resident", "Resident Sessions"),
    ("cache_mode", "Cache Mode"),
    ("history_mode", "History Mode"),
    ("max_tokens", "Max Tokens"),
    ("max_tokens_mode", "Max Tokens Mode"),
    ("max_context_length", "Max Context"),
    ("failed_request_threshold", "Fail Threshold"),
    ("trajectory_start_min_ratio", "Trajectory Start Min"),
    ("trajectory_start_max_ratio", "Trajectory Start Max"),
    ("random_seed", "Random Seed"),
    ("inferencex_git_ref", "Client Ref"),
    ("aiperf_version", "AIPerf Version"),
    ("swo_bench_version", "swo-bench Version"),
    ("swo_session_id", "swo-bench Session"),
    ("benchmark_id", "Benchmark ID"),
    ("run_started_at", "Started"),
    ("run_ended_at", "Ended"),
]

# Columns whose float formatting differs from the 1-decimal default.
_THROUGHPUT_KEYS = frozenset(
    {
        "output_token_throughput",
        "output_token_throughput_per_user",
        "median_output_token_throughput_per_user",
        "mean_e2e_norm_intvty",
        "p75_e2e_norm_intvty",
        "p90_e2e_norm_intvty",
        "total_token_throughput",
        "active_throughput_tok_per_s",
        "effective_prefill_throughput",
        "active_prefill_throughput",
        "prefill_tok_per_sec_mean",
        "prefill_tok_per_sec_p50",
        "prefill_tok_per_sec_p90",
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
        "ready_starved_events",
        "pace_idle_ms",
        "tool_idle_ms",
        "resident",
        "max_tokens",
        "concurrency_peak",
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
        "concurrency_mean",
        "connection_reuse_rate",
    }
)


# Metric definitions, grouped by the harness that produces the metric. Emitting
# only the groups whose source is present keeps a swo-bench-only report from
# defining AIPerf metrics it never measured (and crediting AIPerf for the ones
# it did).
SHARED_DEFINITIONS: List[str] = [
    # TPOT is not here: only AIPerf reports a usable one, and naming it in a
    # swo-bench report would define a column that report does not contain.
    "**TTFT / E2EL**: time to first token and end-to-end request latency. "
    "Long-context agentic prefill makes TTFT the headline metric, so its "
    "spread is reported.",
    "**Out Tok/s/User**: decode speed while a request is streaming, i.e. how "
    "fast a single agent's turn produces text.",
    "**Reqs OK / Errors / Error %**: successful requests (the sample size "
    "behind every latency stat), failed ones, and the failure share.",
    "**Measured (s)**: actual wall clock the run occupied.",
]

INFERENCEX_DEFINITIONS: List[str] = [
    "**TPOT**: inter-token latency, i.e. the gap between streamed tokens once "
    "generation is under way.",
    "**TTFOT**: time to the first token the user actually sees. On a reasoning "
    "model the gap to TTFT is time spent emitting reasoning tokens.",
    "**CO-Adj E2EL**: latency corrected for coordinated omission, so it "
    "includes the time a request spent waiting to be dispatched. It tracks "
    "E2EL until requests start queueing.",
    "**Err-Adj TTFT / E2EL**: percentiles with failed requests folded in, so a "
    "run that got fast by erroring out cannot look good. Omitted when a run "
    "had no errors.",
    "**E2E Tok/s/User**: divides by whole-request wall clock rather than "
    "streaming time, so it includes prefill and is the user-visible speed of "
    "an agentic turn; expect it to be several times lower than "
    "**Out Tok/s/User**.",
    "**Prefill / Decode Tok/s**: the phase split of serving throughput, "
    "averaged over the whole run. The **(Active)** variants only count the "
    "windows where that phase was actually working, so the ratio between them "
    "is the phase's duty cycle.",
    "**Eff. Concur**: concurrency actually in flight. Trace replay honours the "
    "recorded think time, so this sits below the requested **Concur** by "
    "design; a large shortfall means the intended load was never applied.",
    "**Tok In Flight**: KV footprint across in-flight requests. Compare the "
    "max against **Max Context**.",
    "**Ctx Overflow**: requests whose trace exceeded **Max Context**. Non-zero "
    "means the replay was truncated and no longer matches what was recorded; "
    "above 1% the scenario invalidates the submission.",
    "**OSL Mismatch / OSL Diff %**: responses whose length did not match the "
    "trace's recorded output, and by how much. Non-zero means the replayed "
    "conversation diverged from what was recorded.",
    "**Cache Hit %**: measured from the serving engine's own counters over the "
    "profiling window, so the cache-priming warmup is excluded. **Theo. Cache "
    "Hit %** is the reuse inherent to the traces, i.e. the upper bound the "
    "engine was offered. A measured rate well below it means the cache was "
    "evicting reuse the workload had available. Omitted when the server "
    "exposes no such counters.",
    "**Credit Drops**: requests the load generator could not dispatch on the "
    "trace's schedule. Non-zero means the recorded timing was not reproduced, "
    "usually because the server was saturated.",
    "**Conn Reuse**: fraction of requests served on a reused connection. Below "
    "1.0 means connections were being torn down, which is what shows up as "
    "`ServerDisconnectedError`.",
    "**Branches**: subagent trajectories spawned during replay. Any errored, "
    "truncated, or delayed child, any parent failed by its child, or any "
    "dropped join means the fan-out did not replay as recorded.",
    "**Fail Threshold**: the error rate above which the scenario aborts the run.",
    "**Measured (s)** should track **Profiling Window (s)**; a large shortfall "
    "means the run ended early.",
]

SWARMONE_DEFINITIONS: List[str] = [
    "**Prefill Tok/s/Req**: prefill rate for one request measured over the "
    "whole prompt sent, so a prefix-cache hit reads as a very high rate. "
    "Capturing that cache-served prefill is the point of the metric -- it is "
    "what a fixed-ISL sweep cannot show -- but it is a per-request rate, not a "
    "serving-capacity figure, so do not read it against the run-averaged "
    "prefill columns.",
    "**TPOT** is deliberately absent: swo-bench's inter-token figure is not a "
    "usable latency (a live run reported 0.01 ms per token against a measured "
    "38.9 tok/s), so **Out Tok/s/User** and **TTFT** are the latency signals "
    "for these rows.",
    "**Total Tok/s (Active)**, **Concur Avg / Peak**: throughput and "
    "concurrency counted only while requests were in flight, which excludes "
    "the replay's simulated tool time. Compare against the requested "
    "**Concur** to see how much of it the recorded sessions actually asked "
    "for.",
    "**Ready Starved**: turns that were ready to dispatch but had no free "
    "slot. Non-zero means **Resident Sessions** bounded the run rather than "
    "the server.",
    "**Pace / Tool Idle (ms)**: time the harness deliberately spent idle "
    "reproducing the recorded think and tool-call time. It is inside "
    "**Measured (s)** but is not server time.",
]


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


def _resolve_field(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[str]:
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _tool_paragraphs(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Describe each harness that contributed a row.

    A mixed sweep runs two unrelated clients over two unrelated trace sets, so
    one blurb cannot honestly cover both -- and crediting AIPerf for a swo-bench
    row would misattribute the numbers.
    """
    sources = {str(row.get("trace_source") or "") for row in rows}
    paragraphs: List[str] = []

    if INFERENCEX_SOURCE in sources:
        client_ref = _resolve_client_ref(rows)
        ref_note = f" pinned at `{client_ref}`" if client_ref else ""
        paragraphs.append(
            "**`inferencex_agentx` rows:** the AIPerf fork vendored in "
            "[InferenceX](https://github.com/SemiAnalysisAI/InferenceX)"
            f"{ref_note}, driving the `inferencex-agentx-mvp` scenario. Each run "
            "replays recorded multi-turn agentic coding traces with their original "
            "timing, including subagent fan-out, so the load shape is the trace's "
            "rather than a synthetic ISL/OSL grid."
        )

    if SWARMONE_SOURCE in sources:
        version = _resolve_field(rows, "swo_bench_version")
        version_note = f" v{version}" if version else ""
        paragraphs.append(
            "**`swarmone` rows:** SwarmOne's "
            "[`swo-bench`](https://swarmone.ai/docs/swo-bench)"
            f"{version_note} replay engine. Each run replays recorded Claude-Code "
            "/ Codex coding sessions turn by turn with growing history, so the "
            "prompt is mostly a prefix-cache hit that grows through the session "
            "-- the shape a fixed-ISL sweep cannot produce. The replay plan is "
            "built server-side by the SwarmOne backend."
        )

    paragraphs.append(
        "Numbers are only comparable across runs from the same source on the "
        "same client revision and trace set."
    )
    return paragraphs


def _submission_caption(rows: Sequence[Mapping[str, Any]]) -> str:
    """Explain what the Submission verdict means for the sources present.

    The two harnesses reach the verdict differently, and reporting one
    explanation over both would overstate what swo-bench actually checks.
    """
    sources = {str(row.get("trace_source") or "") for row in rows}
    sentences: List[str] = []
    if INFERENCEX_SOURCE in sources:
        sentences.append(
            "For `inferencex_agentx` it is the scenario's own verdict, not "
            "ours: it folds static scenario-lock violations, a context-overflow "
            "rate above 1%, and early cancellation into one flag."
        )
    if SWARMONE_SOURCE in sources:
        sentences.append(
            "swo-bench publishes no such verdict, so for `swarmone` it is our "
            "own completeness check (the run returned successful requests with "
            "a plausible TTFT)."
        )
    sentences.append(
        "An invalid run is failed by the driver and never reaches this table."
    )
    return " ".join(sentences)


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

    parts.append("**Benchmarking Tools:**")
    parts.extend(_tool_paragraphs(rows))

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
            f"**Submission** {_submission_caption(rows)}\n\n{health}"
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

    sweep = _agentic_sweep_block(rows)
    if sweep:
        parts.append(sweep)

    parts.append(_definitions_block(rows))

    return "\n\n".join(parts)


def _agentic_sweep_block(rows: Sequence[Mapping[str, Any]]) -> str:
    """Emit the measured sweep in the requirements document's own shape.

    The tables above are for reading; this is for comparing. A requirements
    document states its expectations as an ``agenticSweep``, so emitting the
    measurement in that same shape lets the expected point and the observed one
    diff field for field instead of being eyeballed across a table.

    Only the InferenceX rows: ``agenticSweep`` is defined over AIPerf's metric
    set, and a swo-bench row would render as a point with nothing in it.
    """
    # Imported here, not at module scope: llm_module imports report_module.schema,
    # so a module-level import back would close the cycle.
    from llm_module.agentic_traces.sweep_export import to_agentic_sweep

    inferencex_rows = [
        row for row in rows if str(row.get("trace_source") or "") == INFERENCEX_SOURCE
    ]
    if not inferencex_rows:
        return ""
    sweep = to_agentic_sweep(inferencex_rows)
    body = json.dumps({"agenticSweep": sweep}, indent=2)
    return (
        "#### Measured `agenticSweep`\n\n"
        "The same runs in the shape a requirements document states its "
        "expectations in, one object per concurrency, so an expected point and "
        "the measured one line up field for field. Written alongside the raw "
        "results as `agentic_sweep.json` too.\n\n"
        f"```json\n{body}\n```"
    )


def _definitions_block(rows: Sequence[Mapping[str, Any]]) -> str:
    """Emit only the metric definitions the report's sources actually produced."""
    sources = {str(row.get("trace_source") or "") for row in rows}
    bullets = list(SHARED_DEFINITIONS)
    if INFERENCEX_SOURCE in sources:
        bullets.extend(INFERENCEX_DEFINITIONS)
    if SWARMONE_SOURCE in sources:
        bullets.extend(SWARMONE_DEFINITIONS)
    body = "\n".join(f"> - {bullet}" for bullet in bullets)
    return f"**Metric definitions:**\n{body}"


# Register at import time so any code path that imports report_module picks the
# renderer up.
register("agentic_traces")(render_agentic_traces)
