# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.summary`` aggregation across runs."""

from __future__ import annotations

import pytest

from report_module.schema import Block, ReportSchema
from report_module.summary import (
    aggregate_benchmark_runs,
    compute_metric_stats,
)


class TestComputeMetricStats:
    def test_single_value_has_zero_spread(self):
        s = compute_metric_stats([10.0])
        assert s.n == 1
        assert s.mean == s.median == s.minimum == s.maximum == 10.0
        assert s.stdev == 0.0
        assert s.cov == 0.0
        assert s.p50 == s.p90 == s.p99 == 10.0

    def test_multi_value_stats(self):
        s = compute_metric_stats([1, 2, 3, 4])
        assert s.n == 4
        assert s.mean == 2.5
        assert s.median == 2.5
        assert s.minimum == 1.0 and s.maximum == 4.0
        assert s.p50 == 2.5  # interpolated midpoint

    def test_cov_is_stdev_over_mean(self):
        s = compute_metric_stats([2, 4])
        assert s.cov == pytest.approx(s.stdev / s.mean)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_metric_stats([])


def _bench_schema(model, device, metrics, *, title="Bench", task_type="image"):
    block = Block(kind="benchmarks", title=title, task_type=task_type, data=metrics)
    return ReportSchema(
        metadata={"report_id": "r", "model_name": model, "device": device},
        sections=[block],
    )


class TestAggregateBenchmarkRuns:
    def test_groups_same_identity_across_runs(self):
        schemas = [
            _bench_schema("m", "n300", {"ttft": 10.0, "tput": 100.0}),
            _bench_schema("m", "n300", {"ttft": 20.0, "tput": 200.0}),
        ]
        aggs = aggregate_benchmark_runs(schemas)
        assert len(aggs) == 1
        agg = aggs[0]
        assert agg.run_count == 2
        assert agg.metrics["ttft"].mean == 15.0
        assert agg.metrics["tput"].mean == 150.0

    def test_distinct_identities_stay_separate(self):
        schemas = [
            _bench_schema("m", "n300", {"ttft": 10.0}),
            _bench_schema("m", "t3k", {"ttft": 10.0}),
        ]
        aggs = aggregate_benchmark_runs(schemas)
        assert {a.device for a in aggs} == {"n300", "t3k"}

    def test_non_benchmark_blocks_ignored(self):
        schema = ReportSchema(
            metadata={"report_id": "r", "model_name": "m", "device": "n300"},
            sections=[Block(kind="evals", data={"score": 0.9})],
        )
        assert aggregate_benchmark_runs([schema]) == []

    def test_check_and_target_check_fields_excluded_from_metrics(self):
        schema = _bench_schema(
            "m", "n300",
            {"ttft": 10.0, "ttft_check": 2, "target_checks": {"target": {"x": 1}}},
        )
        agg = aggregate_benchmark_runs([schema])[0]
        assert set(agg.metrics) == {"ttft"}

    def test_nested_numeric_metrics_flattened_with_dotted_path(self):
        schema = _bench_schema("m", "n300", {"Benchmarks": {"ttft": 10.0}})
        agg = aggregate_benchmark_runs([schema])[0]
        assert "Benchmarks.ttft" in agg.metrics

    def test_bool_values_not_treated_as_numeric(self):
        schema = _bench_schema("m", "n300", {"ttft": 10.0, "streaming_enabled": True})
        agg = aggregate_benchmark_runs([schema])[0]
        assert set(agg.metrics) == {"ttft"}
