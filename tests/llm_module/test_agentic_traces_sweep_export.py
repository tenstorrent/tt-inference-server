# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for rendering measured agentic-traces runs as an ``agenticSweep``.

The point of this export is that a customer's expected sweep point and our
measured one line up field for field, so the invariants worth pinning are the
ones that would silently break that comparison:

* the emitted key set is the document's, spelled exactly,
* a metric the run did not produce is absent rather than zero, so a gap cannot
  be mistaken for a measurement of zero,
* ``goodputPct`` is never emitted while the run passes no ``--goodput`` SLOs,
  since AIPerf reports none and a fabricated one would gate on nothing,
* the interactivity percentiles keep their inverted sense (a p90 is the slow
  tail, i.e. AIPerf's p10), which is the difference between passing and failing
  a contractual gate,
* one point per run, ordered by concurrency.
"""

from __future__ import annotations

import json

from llm_module.agentic_traces.sweep_export import (
    to_agentic_sweep,
    to_agentic_sweep_point,
    write_agentic_sweep,
)

# Shaped like ``parse_aiperf_output`` output for a single completed run.
_METRICS = {
    "mean_ttft_ms": 833.64,
    "median_ttft_ms": 728.85,
    "p90_ttft_ms": 1347.0,
    "p95_ttft_ms": 1765.23,
    "mean_tpot_ms": 3.21,
    "median_tpot_ms": 3.14,
    "p90_tpot_ms": 3.32,
    "p95_tpot_ms": 3.46,
    "mean_e2el_ms": 7084.63,
    "median_e2el_ms": 3380.72,
    "p90_e2el_ms": 17735.0,
    "p95_e2el_ms": 22518.11,
    "request_throughput": 0.07416,
    "total_token_throughput": 20057.37,
    "output_token_throughput": 146.77,
    "mean_isl": 268485.14,
    "p95_isl": 514129.4,
    "mean_osl": 1979.18,
    "p95_osl": 6907.9,
    "p75_e2e_norm_intvty": 188.16,
    "p90_e2e_norm_intvty": 114.28,
    "theoretical_prefix_cache_hit_pct": 98.08,
}


class TestToAgenticSweepPoint:
    def test_emits_the_documents_field_names(self):
        point = to_agentic_sweep_point(_METRICS, concurrency=1)

        assert point == {
            "concurrency": 1,
            "ttftMeanMs": 833.64,
            "ttftP50Ms": 728.85,
            "ttftP90Ms": 1347.0,
            "ttftP95Ms": 1765.23,
            "tpotMeanMs": 3.21,
            "tpotP50Ms": 3.14,
            "tpotP90Ms": 3.32,
            "tpotP95Ms": 3.46,
            "e2elMeanMs": 7084.63,
            "e2elP50Ms": 3380.72,
            "e2elP90Ms": 17735.0,
            "e2elP95Ms": 22518.11,
            "reqThroughputRps": 0.07416,
            "totalThroughputTps": 20057.37,
            "decodeThroughputTps": 146.77,
            "inputTokensMean": 268485.14,
            "inputTokensP95": 514129.4,
            "outputTokensMean": 1979.18,
            "outputTokensP95": 6907.9,
            "e2eNormIntvtyP75Tps": 188.16,
            "e2eNormIntvtyP90Tps": 114.28,
            "kvCacheHitRatePct": 98.08,
        }

    def test_omits_metrics_the_run_did_not_produce(self):
        """An absent metric must not surface as a measured zero."""
        point = to_agentic_sweep_point({"mean_ttft_ms": 100.0}, concurrency=8)

        assert point == {"concurrency": 8, "ttftMeanMs": 100.0}

    def test_omits_goodput_when_no_slos_graded_the_run(self):
        """AIPerf reports goodput only when the command passes --goodput SLOs."""
        point = to_agentic_sweep_point({**_METRICS, "goodput": 0}, concurrency=1)

        assert "goodputPct" not in point

    def test_derives_goodput_share_from_the_rate(self):
        """AIPerf reports good requests/sec; the document wants a share."""
        point = to_agentic_sweep_point(
            {**_METRICS, "goodput_slo": "time_to_first_token:2000", "goodput": 0.06},
            concurrency=1,
        )

        # 0.06 good req/s out of 0.07416 total.
        assert point["goodputPct"] == 80.91

    def test_reports_a_graded_zero_as_zero(self):
        """No request meeting the SLOs is a measurement, not a missing value."""
        point = to_agentic_sweep_point(
            {**_METRICS, "goodput_slo": "time_to_first_token:1", "goodput": 0.0},
            concurrency=1,
        )

        assert point["goodputPct"] == 0.0

    def test_prefers_the_measured_cache_hit_rate(self):
        point = to_agentic_sweep_point(
            {**_METRICS, "measured_prefix_cache_hit_pct": 94.61},
            concurrency=1,
        )

        assert point["kvCacheHitRatePct"] == 94.61

    def test_falls_back_to_the_theoretical_cache_hit_rate(self):
        """A server exposing no prefix-cache metrics still reports a rate."""
        assert to_agentic_sweep_point(_METRICS, concurrency=1)["kvCacheHitRatePct"] == (
            98.08
        )

    def test_keeps_interactivity_percentiles_on_the_slow_tail(self):
        """P90 interactivity must stay below P75: it is the slow tail."""
        point = to_agentic_sweep_point(_METRICS, concurrency=1)

        assert point["e2eNormIntvtyP90Tps"] < point["e2eNormIntvtyP75Tps"]

    def test_ignores_non_numeric_and_boolean_metrics(self):
        point = to_agentic_sweep_point(
            {"mean_ttft_ms": None, "median_ttft_ms": True, "p90_ttft_ms": "n/a"},
            concurrency=4,
        )

        assert point == {"concurrency": 4}


class TestToAgenticSweep:
    def test_emits_one_point_per_run_ordered_by_concurrency(self):
        runs = [
            {**_METRICS, "concurrency": 16},
            {**_METRICS, "concurrency": 1},
            {**_METRICS, "concurrency": 4},
        ]

        sweep = to_agentic_sweep(runs)

        assert [point["concurrency"] for point in sweep] == [1, 4, 16]

    def test_empty_when_no_runs_completed(self):
        assert to_agentic_sweep([]) == []


class TestWriteAgenticSweep:
    def test_writes_a_list_even_for_one_point(self, tmp_path):
        """A file written mid-sweep must not need special-casing to read."""
        path = write_agentic_sweep([{**_METRICS, "concurrency": 4}], tmp_path)

        assert json.loads(path.read_text()) == {
            "agenticSweep": [to_agentic_sweep_point(_METRICS, concurrency=4)]
        }

    def test_orders_points_by_concurrency(self, tmp_path):
        path = write_agentic_sweep(
            [{**_METRICS, "concurrency": c} for c in (16, 1, 8)], tmp_path
        )

        points = json.loads(path.read_text())["agenticSweep"]
        assert [p["concurrency"] for p in points] == [1, 8, 16]

    def test_creates_the_output_directory(self, tmp_path):
        path = write_agentic_sweep(
            [{**_METRICS, "concurrency": 1}], tmp_path / "nested" / "dir"
        )

        assert path.exists()
