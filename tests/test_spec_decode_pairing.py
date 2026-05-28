# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for benchmarking.spec_decode_common."""

import json
from pathlib import Path

import pytest

from benchmarking.spec_decode_common import (
    SpecDecodeRunSpec,
    compute_speedup,
    merge_acceptance_rate,
)
from benchmarking.summary_report import _pair_spec_decode_results


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def test_run_spec_validates_and_formats_slug():
    with pytest.raises(ValueError, match="public_dataset"):
        SpecDecodeRunSpec(
            public_dataset="",
            output_len=128,
            max_concurrency=1,
            num_prompts=4,
        )
    spec = SpecDecodeRunSpec(
        public_dataset="spec_bench",
        output_len=128,
        max_concurrency=4,
        num_prompts=16,
    )
    assert spec.slug == "spec_bench_osl-128_maxcon-4_n-16"
    # Without an explicit output_len the model decodes to natural EOS, and
    # the slug omits the osl segment so summary_report's filename regex can
    # tell the two modes apart.
    natural_osl = SpecDecodeRunSpec(
        public_dataset="spec_bench",
        max_concurrency=4,
        num_prompts=16,
    )
    assert natural_osl.slug == "spec_bench_maxcon-4_n-16"
    # When num_prompts is also unset, aiperf consumes the entire public
    # dataset — slug must drop the n- segment so the filename round-trips
    # through summary_report's regex.
    full_dataset = SpecDecodeRunSpec(
        public_dataset="speed_bench_coding",
        max_concurrency=1,
    )
    assert full_dataset.slug == "speed_bench_coding_maxcon-1"


def test_merge_acceptance_rate_preserves_existing_fields(tmp_path):
    src = tmp_path / "result.json"
    _write(src, {"mean_e2el_ms": 100.0, "p99_e2el_ms": 250.0})
    merge_acceptance_rate(src, {"acceptance_rate": 0.85, "accepted_tokens": 425.0})
    annotated = json.loads(src.read_text())
    assert annotated["mean_e2el_ms"] == 100.0
    assert annotated["p99_e2el_ms"] == 250.0
    assert annotated["spec_decode_metrics"]["acceptance_rate"] == 0.85
    assert annotated["spec_decode_metrics"]["accepted_tokens"] == 425.0


def test_compute_speedup_ratios_and_missing_field_handling():
    baseline = {
        "mean_e2el_ms": 200.0,
        "p50_e2el_ms": 180.0,
        "p95_e2el_ms": 250.0,
        "output_throughput": 100.0,
    }
    spec = {
        "mean_e2el_ms": 100.0,
        "p50_e2el_ms": 90.0,
        "p95_e2el_ms": 125.0,
        "output_throughput": 200.0,
        "spec_decode_metrics": {"acceptance_rate": 0.8},
    }
    result = compute_speedup(baseline, spec)
    assert result["speedup_mean_e2el"] == pytest.approx(2.0)
    assert result["speedup_p50_e2el"] == pytest.approx(2.0)
    assert result["output_tput_ratio"] == pytest.approx(2.0)
    assert result["spec_acceptance_rate"] == 0.8
    # Baseline has no spec_decode_metrics block — acceptance rate stays None.
    assert result["baseline_acceptance_rate"] is None
    # TPOT fields weren't supplied on either side — ratios must be None, not
    # an exception (this is the "result is partial" failure mode in prod).
    assert result["tpot_ratio_p50"] is None


def _processed(role: str, **fields) -> dict:
    """Shape of a per-run dict as it leaves process_benchmark_file."""
    base = {
        "task_type": "spec_decode",
        "model_name": "modelid",
        "device": "gpu",
        "public_dataset": "spec_bench",
        "output_sequence_length": 128,
        "max_con": 4,
        "num_requests": 16,
        "endpoint_role": role,
        "timestamp": "2026-05-20_10-00-00",
    }
    base.update(fields)
    return base


def test_pair_spec_decode_results_matches_baseline_and_spec():
    baseline = _processed(
        "baseline",
        mean_e2el_ms=200.0,
        p50_e2el_ms=180.0,
        output_throughput=100.0,
    )
    spec = _processed(
        "spec",
        mean_e2el_ms=100.0,
        p50_e2el_ms=90.0,
        output_throughput=200.0,
        spec_decode_metrics={"acceptance_rate": 0.8},
    )
    pairs = _pair_spec_decode_results([baseline, spec])
    assert len(pairs) == 1
    assert pairs[0]["speedup_p50_e2el"] == pytest.approx(2.0)
    assert pairs[0]["output_tput_ratio"] == pytest.approx(2.0)
    assert pairs[0]["public_dataset"] == "spec_bench"


def test_pair_spec_decode_results_skips_unmatched_runs():
    # A lone baseline (no spec twin) or lone spec (no baseline twin) must not
    # produce a pair — otherwise the report would show speedup vs. nothing.
    assert _pair_spec_decode_results([_processed("baseline")]) == []
    assert _pair_spec_decode_results([_processed("spec")]) == []


def test_pair_spec_decode_results_uses_latest_timestamp_per_role():
    older_baseline = _processed(
        "baseline", mean_e2el_ms=400.0, timestamp="2026-05-20_09-00-00"
    )
    newer_baseline = _processed(
        "baseline", mean_e2el_ms=200.0, timestamp="2026-05-20_10-00-00"
    )
    spec = _processed("spec", mean_e2el_ms=100.0, timestamp="2026-05-20_10-05-00")
    pairs = _pair_spec_decode_results([older_baseline, newer_baseline, spec])
    assert len(pairs) == 1
    # newer baseline (200ms) was kept, so speedup is 200/100 = 2.0 (not 4.0)
    assert pairs[0]["speedup_mean_e2el"] == pytest.approx(2.0)
