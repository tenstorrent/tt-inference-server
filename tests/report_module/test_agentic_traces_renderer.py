# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the ``agentic_traces`` report renderer.

Guards the split into latency/throughput/health/config tables, the transposed
config table, and the number formatting, since the generic renderer produced a
single very wide table that led with config echo and rendered latencies in
scientific notation.
"""

from __future__ import annotations

from report_module.agentic_traces_renderer import render_agentic_traces
from report_module.renderers import get_renderer
from report_module.schema import Block

METADATA = {"model_name": "Kimi-K2.7-Code", "device": "SUPER_CLUSTER"}


def _record(**overrides):
    """One run record shaped like the driver payload for a real completed run."""
    record = {
        "kind": "agentic_traces",
        "model": "moonshotai/Kimi-K2.7-Code",
        "trace_source": "inferencex_agentx",
        "label": "inferencex_agentx_semianalysis_cc_traces_weka_062126_256k_c1_ci",
        "mode": "ci",
        "scenario": "inferencex-agentx-mvp",
        "public_dataset": "semianalysis_cc_traces_weka_062126_256k",
        "concurrency": 1,
        "benchmark_duration": 900,
        "warmup_requests_per_lane": 3,
        "warmup_grace_period": 600,
        "max_context_length": 262144,
        "random_seed": 42,
        "failed_request_threshold": 0.1,
        "trajectory_start_min_ratio": 0.25,
        "trajectory_start_max_ratio": 0.75,
        "artifact_dir": "/very/long/path/that/should/not/reach/the/report",
        "mean_ttft_ms": 7868.93,
        "median_ttft_ms": 6015.4,
        "p90_ttft_ms": 13670.8,
        "p99_ttft_ms": 27090.7,
        "std_ttft_ms": 6038.8,
        "mean_tpot_ms": 11.8,
        "p90_tpot_ms": 19.33,
        "p99_tpot_ms": 28.44,
        "mean_e2el_ms": 22881.8,
        "median_e2el_ms": 8461.5,
        "p99_e2el_ms": 137518.3,
        "mean_ttfot_ms": 9660.4,
        "mean_effective_latency_ms": 22887.3,
        "p99_effective_latency_ms": 137522.2,
        "output_token_throughput_per_user": 110.8,
        "mean_e2e_norm_intvty": 31.3,
        "total_token_throughput": 4856.63,
        "output_token_throughput": 41.99,
        "request_throughput": 0.042,
        "effective_prefill_throughput": 4814.64,
        "active_prefill_throughput": 14882.3,
        "effective_decode_throughput": 41.95,
        "active_decode_throughput": 67.86,
        "effective_concurrency": 0.9596,
        "mean_tokens_in_flight": 173742.75,
        "max_tokens_in_flight": 243462.0,
        "mean_isl": 114811.4,
        "mean_osl": 1001.3,
        "total_input_tokens": 4477645,
        "total_output_tokens": 39052,
        "completed": 39,
        "error_request_count": 0,
        "error_rate_pct": 0.0,
        "context_overflow_count": 0,
        "osl_mismatch_count": 0,
        "osl_mismatch_diff_pct": 0.0,
        "measured_prefix_cache_hit_pct": 95.2617,
        "theoretical_prefix_cache_hit_pct": 95.13,
        "credit_drop_count": 2,
        "connection_reuse_rate": 0.9230,
        "measured_benchmark_duration": 919.4,
        "was_cancelled": False,
        "submission_valid": True,
        "branch_children_spawned": 4,
        "branch_children_errored": 0,
        "branch_children_truncated": 1,
        "branch_children_delayed": 1,
        "branch_parents_failed_due_to_child_error": 0,
        "branch_joins_suppressed": 0,
        "dataset_num_entries": 32,
        "dataset_hf_name": "semianalysisai/cc-traces-weka-062126-256k",
        "inferencex_git_ref": "e2dcfa91c86936cc011e3be0668eb3b1ca17288f",
        "aiperf_version": "0.8.0",
        "benchmark_id": "a70d73f2-53f3-431a-ab89-1f118bcd505f",
        "run_started_at": "2026-07-29T10:25:09.055955",
        "run_ended_at": "2026-07-29T10:40:39.061471",
    }
    record.update(overrides)
    return record


def _swo_record(**overrides):
    """One run record shaped like the swo-bench driver payload.

    The two harnesses share the ``agentic_traces`` kind and this renderer, but
    overlap only partially: swo-bench has no scenario verdict, no branch or
    cache counters, and contributes prefill/duty-cycle metrics of its own.
    """
    record = {
        "kind": "agentic_traces",
        "model": "moonshotai/Kimi-K2.7-Code",
        "backend": "swo-bench",
        "trace_source": "swarmone",
        "label": "swarmone_claude-code-mixed_sympy-bugfix_c1_ci",
        "mode": "ci",
        "scenario": "claude-code-mixed",
        "task": "sympy-bugfix",
        "concurrency": 1,
        "resident": 1,
        "cache_mode": "warm",
        "history_mode": "full",
        "max_tokens": 4096,
        "max_tokens_mode": "cap",
        "max_context_length": 262144,
        "artifact_dir": "/very/long/path/that/should/not/reach/the/report",
        "submission_valid": True,
        "mean_ttft_ms": 5321.4,
        "median_ttft_ms": 4102.8,
        "p90_ttft_ms": 11204.6,
        "p99_ttft_ms": 19883.1,
        "mean_e2el_ms": 18422.7,
        "median_e2el_ms": 7712.2,
        "p99_e2el_ms": 98211.5,
        # No TPOT: the driver suppresses swo-bench's unusable inter-token figure.
        "output_token_throughput_per_user": 96.3,
        "median_output_token_throughput_per_user": 101.7,
        "output_token_throughput": 38.2,
        "total_token_throughput": 38.2,
        "prefill_tok_per_sec_mean": 24518.9,
        "prefill_tok_per_sec_p50": 21044.3,
        "prefill_tok_per_sec_p90": 40122.7,
        "active_throughput_tok_per_s": 88.6,
        "concurrency_mean": 0.62,
        "concurrency_peak": 1,
        "ready_starved_events": 0,
        "pace_idle_ms": 41200,
        "tool_idle_ms": 18300,
        "completed": 27,
        "error_request_count": 0,
        "error_rate_pct": 0.0,
        "mean_isl": 88214.6,
        "mean_osl": 812.4,
        "total_input_tokens": 2381794,
        "total_output_tokens": 21935,
        "request_throughput": 0.031,
        "measured_benchmark_duration": 871.2,
        "swo_session_id": "3f1c9a20-77bd-4a0e-9d1e-2f5b6c8a1d44",
        "swo_bench_version": "3.1.2",
    }
    record.update(overrides)
    return record


def _render(*records):
    block = Block(
        kind="agentic_traces",
        id="kimi",
        targets={"model": "moonshotai/Kimi-K2.7-Code", "device": "SUPER_CLUSTER"},
        data={"records": list(records)},
    )
    return render_agentic_traces(block, METADATA)


class TestRegistration:
    def test_kind_resolves_to_this_renderer(self):
        assert get_renderer("agentic_traces") is render_agentic_traces

    def test_empty_block_renders_nothing(self):
        block = Block(kind="agentic_traces", id="x", targets={}, data={"records": []})
        assert render_agentic_traces(block, METADATA) == ""


class TestTableSplit:
    def test_emits_the_four_sections(self):
        out = _render(_record())
        assert "#### Per-run Latency (ms)" in out
        assert "#### Per-run Throughput & Load" in out
        assert "#### Run Health & Validity" in out
        assert "#### Run Configuration" in out

    def test_metric_tables_exclude_config_and_artifact_path(self):
        """The config echo dominated the old single table; it belongs below."""
        out = _render(_record())
        metrics_tables = out.split("#### Run Health")[0]
        assert "TTFT Avg" in metrics_tables
        assert "Profiling Window" not in metrics_tables
        assert "Random Seed" not in metrics_tables
        # The artifact path is in the raw JSON; it made the table unreadable.
        assert "/very/long/path" not in out

    def test_latency_and_throughput_are_separate_tables(self):
        """Together they are wide enough to need horizontal scrolling."""
        out = _render(_record())
        latency = out.split("#### Per-run Throughput")[0]
        assert "TTFT P99" in latency
        assert "Total Tok/s" not in latency

    def test_every_metric_column_is_labelled(self):
        """The generic renderer left raw keys like `p90_ttft_ms` as headers."""
        out = _render(_record())
        for raw_key in ("p90_ttft_ms", "median_tpot_ms", "mean_isl", "completed"):
            assert raw_key not in out

    def test_run_identity_appears_in_every_metric_table(self):
        out = _render(_record())
        assert out.count("| inferencex_agentx |") >= 3

    def test_columns_the_client_did_not_emit_are_dropped(self):
        """A clean run has no error-adjusted percentiles to report."""
        out = _render(_record())
        assert "Err-Adj TTFT P50" not in out
        assert "Top Errors" not in out

    def test_error_adjusted_percentiles_appear_when_present(self):
        out = _render(_record(p50_adj_ttft_ms=8123.4, p90_adj_ttft_ms=15987.6))
        assert "Err-Adj TTFT P50" in out
        assert "8,123.4" in out


class TestConfigTable:
    def test_config_is_transposed_to_one_row_per_field(self):
        """As columns, the long dataset and label values set the report width."""
        out = _render(_record())
        config = out.split("#### Run Configuration")[1]
        assert "| Field" in config
        assert "| Max Context          | 262,144" in config

    def test_config_column_header_identifies_the_run(self):
        out = _render(_record())
        config = out.split("#### Run Configuration")[1]
        assert "inferencex_agentx c1 ci" in config

    def test_one_column_per_run(self):
        out = _render(_record(concurrency=8), _record(concurrency=32))
        config = out.split("#### Run Configuration")[1]
        header = config.splitlines()[4]
        assert "inferencex_agentx c8 ci" in header
        assert "inferencex_agentx c32 ci" in header

    def test_client_run_identifiers_are_carried(self):
        """A number is only reproducible alongside the client that produced it."""
        config = _render(_record()).split("#### Run Configuration")[1]
        assert "0.8.0" in config
        assert "a70d73f2-53f3-431a-ab89-1f118bcd505f" in config


class TestFormatting:
    def test_latencies_avoid_scientific_notation(self):
        out = _render(_record())
        assert "13,670.8" in out
        assert "e+04" not in out

    def test_ratios_keep_their_precision(self):
        """1-decimal rounding turned 0.25/0.75 into 0.2/0.8."""
        out = _render(_record())
        assert "0.25" in out and "0.75" in out
        assert "0.10" in out

    def test_client_ref_is_full_in_both_the_preamble_and_the_config_table(self):
        """The transposed config table has the room the old one did not."""
        out = _render(_record())
        assert (
            "| Client Ref           | e2dcfa91c86936cc011e3be0668eb3b1ca17288f" in out
        )
        assert "pinned at `e2dcfa91c86936cc011e3be0668eb3b1ca17288f`" in out

    def test_counts_render_as_integers(self):
        out = _render(_record())
        assert "| 262,144 " in out
        assert "262144.0" not in out


class TestValidity:
    def test_a_valid_run_is_marked_valid(self):
        assert "✅ valid" in _render(_record())

    def test_an_invalid_run_shows_its_reasons(self):
        out = _render(
            _record(
                submission_valid=False,
                submission_invalid_reasons=["context_overflow_rate_exceeded"],
            )
        )
        assert "❌ INVALID (context_overflow_rate_exceeded)" in out

    def test_unknown_validity_falls_back_to_status_then_na(self):
        record = _record()
        record.pop("submission_valid")
        assert "N/A" in _render(record)

    def test_error_summary_is_surfaced(self):
        out = _render(
            _record(
                error_request_count=35,
                error_rate_pct=9.4339,
                error_summary="34xServerDisconnectedError, 1xInvalidInferenceResultError",
            )
        )
        assert "34xServerDisconnectedError" in out
        assert "9.43" in out


class TestPrefixCacheColumns:
    def test_measured_and_theoretical_hit_rates_sit_side_by_side(self):
        """The comparison is the point: measured is what the engine caught of
        the reuse the traces offered."""
        health = _render(_record()).split("#### Run Health")[1]
        assert "Cache Hit %" in health
        assert "Theo. Cache Hit %" in health
        assert health.index("| Cache Hit %") < health.index("| Theo. Cache Hit %")

    def test_hit_rates_keep_two_decimals(self):
        out = _render(_record())
        assert "95.26" in out
        assert "95.13" in out

    def test_a_server_without_counters_drops_only_the_measured_column(self):
        record = _record()
        record.pop("measured_prefix_cache_hit_pct")
        health = _render(record).split("#### Run Health")[1]
        assert "| Cache Hit %" not in health
        assert "Theo. Cache Hit %" in health


class TestMultipleRuns:
    def test_rows_sort_by_source_then_concurrency(self):
        out = _render(
            _record(concurrency=32, label="c32"),
            _record(concurrency=8, label="c8"),
        )
        latency_table = out.split("#### Per-run Throughput")[0]
        assert latency_table.index("| 8 ") < latency_table.index("| 32 ")


class TestSwarmOneRows:
    """swo-bench shares this renderer, so its metrics must reach the report and
    its rows must not be described as AIPerf's."""

    def test_swarmone_specific_metrics_reach_the_report(self):
        out = _render(_swo_record())
        for header in (
            "Prefill Tok/s/Req Avg",
            "Total Tok/s (Active)",
            "Concur Peak",
            "Ready Starved",
            "Pace Idle (ms)",
        ):
            assert header in out, f"{header} missing from the report"

    def test_swarmone_config_reaches_the_config_table(self):
        config = _render(_swo_record()).split("#### Run Configuration")[1]
        for label in ("Task", "Resident Sessions", "Cache Mode", "History Mode"):
            assert label in config, f"{label} missing from the config table"
        assert "sympy-bugfix" in config
        assert "3.1.2" in config
        assert "3f1c9a20-77bd-4a0e-9d1e-2f5b6c8a1d44" in config

    def test_a_healthy_run_is_not_reported_as_unknown(self):
        """The driver rejects an unusable run, so a row that reaches the report
        is valid; it used to render N/A because the field went unset."""
        health = _render(_swo_record()).split("#### Run Health")[1]
        assert "✅ valid" in health
        assert "| N/A" not in health.split("\n\n")[2]

    def test_the_tool_blurb_credits_swo_bench_not_aiperf(self):
        out = _render(_swo_record())
        assert "swo-bench" in out
        assert "AIPerf" not in out
        assert "InferenceX" not in out

    def test_aiperf_only_metrics_are_not_defined(self):
        """Defining metrics the report does not contain is worse than silence."""
        out = _render(_swo_record())
        for absent in ("CO-Adj E2EL", "Credit Drops", "Theo. Cache Hit %", "TTFOT"):
            assert absent not in out, f"{absent} defined for a swo-bench-only report"

    def test_tpot_is_absent_and_its_absence_explained(self):
        """swo-bench's inter-token figure is unusable, so the driver drops it;
        the column must then disappear rather than render a false 0.0."""
        out = _render(_swo_record())
        assert "TPOT Avg" not in out
        assert "**TPOT** is deliberately absent" in out

    def test_no_definition_promises_a_tpot_column(self):
        """The shared bullet used to read "TTFT / TPOT / E2EL", defining a
        column this report does not have and contradicting the note below."""
        out = _render(_swo_record())
        assert "**TTFT / E2EL**" in out
        assert "TTFT / TPOT / E2EL" not in out

    def test_an_aiperf_report_carries_no_swarmone_definitions(self):
        out = _render(_record())
        assert "Ready Starved" not in out
        assert "Prefill Tok/s/Req" not in out
        assert "deliberately absent" not in out

    def test_an_aiperf_report_still_defines_tpot(self):
        """AIPerf does report a usable TPOT, so moving the definition out of the
        shared group must not lose it."""
        out = _render(_record())
        assert "TPOT Avg" in out
        assert "**TPOT**: inter-token latency" in out


class TestMixedSweep:
    """A default sweep can run both harnesses into one section."""

    def test_both_tools_are_described(self):
        out = _render(_record(), _swo_record())
        assert "`inferencex_agentx` rows:" in out
        assert "`swarmone` rows:" in out

    def test_both_definition_sets_are_present(self):
        out = _render(_record(), _swo_record())
        assert "CO-Adj E2EL" in out
        assert "Ready Starved" in out

    def test_the_submission_caption_covers_both_verdicts(self):
        out = _render(_record(), _swo_record())
        caption = out.split("#### Run Health & Validity")[1].split("\n\n")[1]
        assert "scenario's own verdict" in caption
        assert "own completeness check" in caption

    def test_each_source_keeps_its_own_columns(self):
        """Columns are shared ground: a metric only one harness emits shows for
        that row and N/A for the other, rather than being dropped."""
        throughput = _render(_record(), _swo_record()).split("#### Per-run Throughput")[
            1
        ]
        assert "Prefill Tok/s/Req Avg" in throughput
        assert "Tok In Flight Avg" in throughput

    def test_rows_sort_with_inferencex_first(self):
        out = _render(_swo_record(), _record())
        latency = out.split("#### Per-run Throughput")[0]
        assert latency.index("| inferencex_agentx ") < latency.index("| swarmone ")


class TestMeasuredAgenticSweep:
    """The report also carries the sweep in the requirements document's shape.

    The tables are for reading; this block is for diffing a measured point
    against the expected one a document declares, so it must stay valid JSON
    under the document's own key.
    """

    def test_sweep_block_is_included_by_default(self):
        out = _render(_record())

        assert "#### Measured `agenticSweep`" in out

    def test_sweep_block_is_parseable_json_under_the_documents_key(self):
        import json

        out = _render(_record(concurrency=8))
        body = out.split("```json", 1)[1].split("```", 1)[0]

        assert [p["concurrency"] for p in json.loads(body)["agenticSweep"]] == [8]

    def test_one_point_per_run_ordered_by_concurrency(self):
        import json

        out = _render(_record(concurrency=16), _record(concurrency=4))
        body = out.split("```json", 1)[1].split("```", 1)[0]

        assert [p["concurrency"] for p in json.loads(body)["agenticSweep"]] == [4, 16]

    def test_swarmone_runs_are_left_out_of_the_sweep(self):
        """agenticSweep is defined over AIPerf's metrics; swo-bench has none."""
        out = _render(_swo_record())

        assert "#### Measured `agenticSweep`" not in out
