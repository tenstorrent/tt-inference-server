# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import copy

import pytest

from report_module.generator import ReportGenerator
from report_module.qwen38_overview import measured_level, render_overview
from report_module.schema import Block, ReportSchema


def benchmark(**overrides):
    return {
        "concurrency": 8,
        "num_requests": 32,
        "input_sequence_length": 4096,
        "output_sequence_length": 252,
        "mean_ttft_ms": 4692.36,
        "mean_tpot_ms": 54.78,
        "tps_output_throughput": 109.3,
        "mean_e2el_ms": 18443.21,
        "target_check": 3,
        "target_checks": {
            "functional": {"ttft_check": 2, "tput_check": 2, "tpot_check": 1},
            "complete": {"ttft_check": 3, "tput_check": 3},
            "target": {"ttft_check": 3, "tput_check": 3},
        },
        **overrides,
    }


@pytest.mark.parametrize(
    "tier,label",
    [("target", "Target"), ("complete", "Complete"), ("functional", "Functional")],
)
def test_highest_fully_passing_tier(tier, label):
    row = benchmark()
    row["target_checks"][tier] = {"ttft_check": 2, "tput_check": 2, "tpot_check": 1}
    assert measured_level(row).endswith(label)


def test_any_failed_metric_prevents_promotion():
    row = benchmark()
    row["target_checks"]["functional"]["ttft_check"] = 3
    assert measured_level(row) == "🔴 Experimental"


@pytest.mark.parametrize(
    "checks",
    [
        {},
        {"target": {}},
        {"target": {"ttft_check": 1}},
        {"target": {"ttft_check": "unexpected"}},
    ],
)
def test_missing_na_unknown_checks_never_pass(checks):
    assert measured_level(benchmark(target_checks=checks)) == "⚪ Unrated"


@pytest.mark.parametrize(
    "override", [{"status": "error"}, {"status": "skip"}, {"error_request_count": 1}]
)
def test_invalid_runs_not_experimental_performance(override):
    assert measured_level(benchmark(**override)) == "⚪ Unrated"


def test_missing_measurements_never_pass():
    assert (
        measured_level({"target_checks": {"target": {"ttft_check": 2}}}) == "⚪ Unrated"
    )


def test_overview_preserves_data_and_does_not_inherit_waivers(tmp_path):
    payload = {
        "metadata": {
            "report_id": "r1",
            "model_impl": "qwen38-autoport-b8",
            "model_name": "Qwen3.8-27B",
            "device": "P300X2",
        },
        "sections": [
            {"kind": "benchmarks", "data": benchmark()},
            {
                "kind": "evals",
                "data": {
                    "task_name": "terminal_bench_2_1",
                    "accuracy": 80,
                    "accuracy_check": 2,
                },
            },
        ],
    }
    original = copy.deepcopy(payload)
    schema = ReportSchema.from_dict(payload)
    md = ReportGenerator().generate(schema, tmp_path).markdown
    assert "### Results at a glance" in md
    assert "🟡 Functional" in md and "❌ FAIL" in md
    assert "80.00" in md and "— (quality eval)" in md
    assert "<details>" in md and "all target checks" in md
    assert payload == original
    assert schema.to_dict() == ReportSchema.from_dict(original).to_dict()


def test_other_models_keep_existing_presentation():
    assert (
        render_overview(
            [Block(kind="benchmarks", data=benchmark())], {"model_impl": "other"}
        )
        == ""
    )
