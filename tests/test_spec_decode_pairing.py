# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for benchmarking.spec_decode_common."""

import json
from pathlib import Path

import pytest

from benchmarking.spec_decode_common import (
    SpecDecodeRunSpec,
    merge_acceptance_rate,
    pair_and_compute_speedup,
)


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def test_run_spec_rejects_unknown_dataset_kind():
    with pytest.raises(ValueError, match="dataset_kind"):
        SpecDecodeRunSpec(
            dataset_kind="bogus",
            category="writing",
            output_len=128,
            max_concurrency=1,
            num_prompts=4,
        )


def test_run_spec_requires_subset_for_speed_bench():
    with pytest.raises(ValueError, match="speed_bench_subset"):
        SpecDecodeRunSpec(
            dataset_kind="speed_bench",
            category=None,
            output_len=128,
            max_concurrency=1,
            num_prompts=4,
        )


def test_run_spec_slug_for_spec_bench():
    spec = SpecDecodeRunSpec(
        dataset_kind="spec_bench",
        category="writing",
        output_len=128,
        max_concurrency=4,
        num_prompts=16,
    )
    assert spec.slug == "spec_bench_writing_osl-128_maxcon-4_n-16"


def test_run_spec_slug_for_speed_bench_with_no_category():
    spec = SpecDecodeRunSpec(
        dataset_kind="speed_bench",
        category=None,
        output_len=128,
        max_concurrency=4,
        num_prompts=16,
        speed_bench_subset="throughput_1k",
    )
    assert spec.slug == (
        "speed_bench_all_throughput_1k_osl-128_maxcon-4_n-16"
    )


def test_run_spec_slug_for_speed_bench_with_explicit_category():
    spec = SpecDecodeRunSpec(
        dataset_kind="speed_bench",
        category="coding",
        output_len=128,
        max_concurrency=4,
        num_prompts=16,
        speed_bench_subset="qualitative",
    )
    assert spec.slug == (
        "speed_bench_coding_qualitative_osl-128_maxcon-4_n-16"
    )


def test_merge_acceptance_rate_preserves_existing_fields(tmp_path):
    src = tmp_path / "result.json"
    _write(src, {"mean_e2el_ms": 100.0, "p99_e2el_ms": 250.0})
    merge_acceptance_rate(
        src, {"acceptance_rate": 0.85, "accepted_tokens": 425.0}
    )
    annotated = json.loads(src.read_text())
    assert annotated["mean_e2el_ms"] == 100.0
    assert annotated["p99_e2el_ms"] == 250.0
    assert annotated["spec_decode_metrics"]["acceptance_rate"] == 0.85
    assert annotated["spec_decode_metrics"]["accepted_tokens"] == 425.0


def test_merge_acceptance_rate_cleans_up_tmp_file(tmp_path):
    src = tmp_path / "result.json"
    _write(src, {"mean_e2el_ms": 100.0})
    merge_acceptance_rate(src, {"acceptance_rate": 0.5})
    assert not (tmp_path / "result.json.tmp").exists()


def test_pair_and_compute_speedup_basic_ratios(tmp_path):
    baseline = tmp_path / "baseline.json"
    spec = tmp_path / "spec.json"
    _write(
        baseline,
        {
            "mean_e2el_ms": 200.0,
            "p50_e2el_ms": 180.0,
            "p95_e2el_ms": 250.0,
            "p99_e2el_ms": 300.0,
            "p50_tpot_ms": 20.0,
            "p95_tpot_ms": 30.0,
            "p99_tpot_ms": 40.0,
            "output_throughput": 100.0,
        },
    )
    _write(
        spec,
        {
            "mean_e2el_ms": 100.0,
            "p50_e2el_ms": 90.0,
            "p95_e2el_ms": 125.0,
            "p99_e2el_ms": 150.0,
            "p50_tpot_ms": 10.0,
            "p95_tpot_ms": 15.0,
            "p99_tpot_ms": 20.0,
            "output_throughput": 200.0,
            "spec_decode_metrics": {"acceptance_rate": 0.85},
        },
    )
    result = pair_and_compute_speedup(baseline, spec)
    assert result["speedup_mean_e2el"] == pytest.approx(2.0)
    assert result["speedup_p50_e2el"] == pytest.approx(2.0)
    assert result["speedup_p95_e2el"] == pytest.approx(2.0)
    assert result["speedup_p99_e2el"] == pytest.approx(2.0)
    assert result["tpot_ratio_p50"] == pytest.approx(2.0)
    assert result["output_tput_ratio"] == pytest.approx(2.0)
    assert result["spec_acceptance_rate"] == 0.85
    assert result["baseline_acceptance_rate"] is None


def test_pair_handles_missing_fields_with_none(tmp_path):
    baseline = tmp_path / "baseline.json"
    spec = tmp_path / "spec.json"
    _write(baseline, {"output_throughput": 100.0})
    _write(spec, {})
    result = pair_and_compute_speedup(baseline, spec)
    assert result["output_tput_ratio"] is None
    assert result["speedup_p50_e2el"] is None
    assert result["baseline_acceptance_rate"] is None
    assert result["spec_acceptance_rate"] is None
