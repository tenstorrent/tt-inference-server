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
        "agentic_cache_warmup_duration": 120,
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
        "e2e_output_token_throughput_per_user": 31.3,
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


class TestMultipleRuns:
    def test_rows_sort_by_source_then_concurrency(self):
        out = _render(
            _record(concurrency=32, label="c32"),
            _record(concurrency=8, label="c8"),
        )
        latency_table = out.split("#### Per-run Throughput")[0]
        assert latency_table.index("| 8 ") < latency_table.index("| 32 ")
