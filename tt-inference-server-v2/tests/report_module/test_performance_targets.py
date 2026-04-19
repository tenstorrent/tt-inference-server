# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import pytest

from report_module.performance_targets import (
    COMPLETE_MULTIPLIER,
    FUNCTIONAL_MULTIPLIER,
    TARGET_MULTIPLIER,
    _metric_ratio_and_check,
    build_target_checks_audio,
    build_target_checks_cnn_image_video,
    build_target_checks_embedding,
    build_target_checks_text_vlm,
    build_target_checks_tts,
    calculate_target_metrics,
    check_text_vlm_targets,
    flatten_target_checks,
)
from workflows.utils_report import BenchmarkTaskParams, PerformanceTarget, PerformanceTargets
from workflows.workflow_types import ReportCheckTypes


# ── _metric_ratio_and_check ──────────────────────────────────────────


class TestMetricRatioAndCheck:
    def test_ascending_pass(self):
        ratio, check = _metric_ratio_and_check(120.0, 100.0, is_ascending=True)
        assert ratio == pytest.approx(1.2)
        assert check == ReportCheckTypes.PASS

    def test_ascending_fail(self):
        ratio, check = _metric_ratio_and_check(80.0, 100.0, is_ascending=True)
        assert ratio == pytest.approx(0.8)
        assert check == ReportCheckTypes.FAIL

    def test_descending_pass(self):
        ratio, check = _metric_ratio_and_check(80.0, 100.0, is_ascending=False)
        assert ratio == pytest.approx(0.8)
        assert check == ReportCheckTypes.PASS

    def test_descending_fail(self):
        ratio, check = _metric_ratio_and_check(120.0, 100.0, is_ascending=False)
        assert ratio == pytest.approx(1.2)
        assert check == ReportCheckTypes.FAIL

    def test_zero_reference(self):
        ratio, check = _metric_ratio_and_check(100.0, 0, is_ascending=True)
        assert ratio == "Undefined"
        assert check == "Undefined"

    def test_zero_measured(self):
        ratio, check = _metric_ratio_and_check(0, 100.0, is_ascending=True)
        assert ratio == 0.0
        assert check == ReportCheckTypes.NA

    def test_none_reference(self):
        ratio, check = _metric_ratio_and_check(100.0, None, is_ascending=True)
        assert ratio == "Undefined"


# ── calculate_target_metrics ─────────────────────────────────────────


class TestCalculateTargetMetrics:
    def test_latency_metric(self):
        config = [{
            "avg_metric": 50.0,
            "target_metric": 100.0,
            "field_name": "ttft",
            "is_ascending_metric": False,
        }]
        metrics = calculate_target_metrics(config)

        assert metrics["functional_ttft"] == 100.0 * FUNCTIONAL_MULTIPLIER
        assert metrics["complete_ttft"] == 100.0 * COMPLETE_MULTIPLIER
        assert metrics["target_ttft"] == 100.0 * TARGET_MULTIPLIER

        assert metrics["functional_ttft_check"] == ReportCheckTypes.PASS
        assert metrics["complete_ttft_check"] == ReportCheckTypes.PASS
        assert metrics["target_ttft_check"] == ReportCheckTypes.PASS

    def test_throughput_metric(self):
        config = [{
            "avg_metric": 60.0,
            "target_metric": 100.0,
            "field_name": "tput_user",
            "is_ascending_metric": True,
        }]
        metrics = calculate_target_metrics(config)

        assert metrics["functional_tput_user"] == 100.0 / FUNCTIONAL_MULTIPLIER
        assert metrics["complete_tput_user"] == 100.0 / COMPLETE_MULTIPLIER
        assert metrics["target_tput_user"] == 100.0 / TARGET_MULTIPLIER

        assert metrics["functional_tput_user_check"] == ReportCheckTypes.PASS
        assert metrics["complete_tput_user_check"] == ReportCheckTypes.PASS
        assert metrics["target_tput_user_check"] == ReportCheckTypes.FAIL

    def test_none_target_skipped(self):
        config = [{
            "avg_metric": 50.0,
            "target_metric": None,
            "field_name": "ttft",
        }]
        metrics = calculate_target_metrics(config)
        assert metrics == {}

    def test_multiple_metrics(self):
        config = [
            {"avg_metric": 50.0, "target_metric": 100.0, "field_name": "ttft", "is_ascending_metric": False},
            {"avg_metric": 80.0, "target_metric": 100.0, "field_name": "tput_user", "is_ascending_metric": True},
        ]
        metrics = calculate_target_metrics(config)
        assert "functional_ttft" in metrics
        assert "functional_tput_user" in metrics


# ── build_target_checks_text_vlm ─────────────────────────────────────


class TestBuildTargetChecksTextVlm:
    def test_all_metrics_pass(self):
        row = {"mean_ttft_ms": 90.0, "mean_tps": 110.0, "tps_decode_throughput": 550.0}
        target = PerformanceTarget(ttft_ms=100.0, tput_user=100.0, tput=500.0, tolerance=0.1)
        result = build_target_checks_text_vlm(row, "customer_functional", target)

        assert result["ttft_check"] == ReportCheckTypes.PASS
        assert result["tput_user_check"] == ReportCheckTypes.PASS
        assert result["tput_check"] == ReportCheckTypes.PASS
        assert result["ttft"] == 100.0
        assert result["ttft_ratio"] == pytest.approx(0.9)

    def test_ttft_fail(self):
        row = {"mean_ttft_ms": 200.0, "mean_tps": 100.0, "tps_decode_throughput": 500.0}
        target = PerformanceTarget(ttft_ms=100.0, tput_user=100.0, tput=500.0, tolerance=0.05)
        result = build_target_checks_text_vlm(row, "target", target)
        assert result["ttft_check"] == ReportCheckTypes.FAIL

    def test_none_targets_give_na(self):
        row = {"mean_ttft_ms": 100.0, "mean_tps": 50.0, "tps_decode_throughput": 200.0}
        target = PerformanceTarget(ttft_ms=None, tput_user=None, tput=None)
        result = build_target_checks_text_vlm(row, "test", target)
        assert result["ttft_check"] == ReportCheckTypes.NA
        assert result["tput_user_check"] == ReportCheckTypes.NA
        assert result["tput_check"] == ReportCheckTypes.NA


# ── build_target_checks_cnn_image_video ──────────────────────────────


class TestBuildTargetChecksCnnImageVideo:
    def test_functional_pass(self):
        targets = PerformanceTargets(tput_user=100.0, ttft_ms=50.0)
        evals_data = [{"tput_user": 15.0}]
        metrics = {
            "functional_ttft": 500.0,
            "functional_ttft_ratio": 0.5,
            "functional_ttft_check": ReportCheckTypes.PASS,
            "complete_ttft": 100.0,
            "complete_ttft_ratio": 0.9,
            "complete_ttft_check": ReportCheckTypes.PASS,
            "target_ttft": 50.0,
            "target_ttft_ratio": 1.5,
            "target_ttft_check": ReportCheckTypes.FAIL,
        }
        result = build_target_checks_cnn_image_video(targets, evals_data, metrics)

        assert "functional" in result
        assert "complete" in result
        assert "target" in result
        assert result["functional"]["tput_check"] == ReportCheckTypes.PASS
        assert result["target"]["tput_check"] == ReportCheckTypes.FAIL

    def test_no_tput_user_gives_na(self):
        targets = PerformanceTargets(tput_user=None, ttft_ms=50.0)
        evals_data = [{"tput_user": 10.0}]
        metrics = {
            "functional_ttft": 500.0, "functional_ttft_ratio": 0.5, "functional_ttft_check": ReportCheckTypes.PASS,
            "complete_ttft": 100.0, "complete_ttft_ratio": 0.8, "complete_ttft_check": ReportCheckTypes.PASS,
            "target_ttft": 50.0, "target_ttft_ratio": 1.2, "target_ttft_check": ReportCheckTypes.FAIL,
        }
        result = build_target_checks_cnn_image_video(targets, evals_data, metrics)
        assert result["functional"]["tput_check"] == ReportCheckTypes.NA


# ── build_target_checks_audio ────────────────────────────────────────


class TestBuildTargetChecksAudio:
    def test_structure(self):
        metrics = {
            "functional_ttft": 500.0, "functional_ttft_ratio": 0.5, "functional_ttft_check": ReportCheckTypes.PASS,
            "complete_ttft": 100.0, "complete_ttft_ratio": 0.8, "complete_ttft_check": ReportCheckTypes.PASS,
            "target_ttft": 50.0, "target_ttft_ratio": 1.2, "target_ttft_check": ReportCheckTypes.FAIL,
        }
        result = build_target_checks_audio(metrics)

        for level in ("functional", "complete", "target"):
            assert level in result
            assert result[level]["tput_check"] == ReportCheckTypes.NA
            assert "ttft_check" in result[level]


# ── build_target_checks_tts ─────────────────────────────────────────


class TestBuildTargetChecksTts:
    def test_with_rtr(self):
        metrics = {
            "functional_ttft": 500.0, "functional_ttft_ratio": 0.5, "functional_ttft_check": ReportCheckTypes.PASS,
            "functional_rtr_check": ReportCheckTypes.PASS,
            "complete_ttft": 100.0, "complete_ttft_ratio": 0.8, "complete_ttft_check": ReportCheckTypes.PASS,
            "complete_rtr_check": ReportCheckTypes.PASS,
            "target_ttft": 50.0, "target_ttft_ratio": 1.2, "target_ttft_check": ReportCheckTypes.FAIL,
            "target_rtr_check": ReportCheckTypes.FAIL,
        }
        result = build_target_checks_tts(metrics)

        assert result["functional"]["rtr_check"] == ReportCheckTypes.PASS
        assert result["target"]["rtr_check"] == ReportCheckTypes.FAIL
        assert result["functional"]["tput_check"] == ReportCheckTypes.NA


# ── build_target_checks_embedding ────────────────────────────────────


class TestBuildTargetChecksEmbedding:
    def test_structure(self):
        metrics = {}
        for level in ("functional", "complete", "target"):
            for base in ("tput_user", "tput_prefill", "e2el_ms"):
                metrics[f"{level}_{base}"] = 50.0
                metrics[f"{level}_{base}_ratio"] = 0.8
                metrics[f"{level}_{base}_check"] = ReportCheckTypes.PASS

        result = build_target_checks_embedding(metrics)

        for level in ("functional", "complete", "target"):
            assert "tput_user" in result[level]
            assert "tput_user_check" in result[level]
            assert "e2el_ms" in result[level]


# ── flatten_target_checks ────────────────────────────────────────────


class TestFlattenTargetChecks:
    def test_basic_flatten(self):
        rows = [{
            "isl": 128,
            "osl": 64,
            "target_checks": {
                "customer_functional": {"ttft_check": ReportCheckTypes.PASS, "ttft": 1000.0},
                "customer_complete": {"ttft_check": ReportCheckTypes.FAIL, "ttft": 200.0},
            },
        }]
        flat = flatten_target_checks(rows)

        assert len(flat) == 1
        assert flat[0]["isl"] == 128
        assert flat[0]["customer_functional_ttft_check"] == ReportCheckTypes.PASS
        assert flat[0]["customer_complete_ttft"] == 200.0
        assert "target_checks" not in flat[0]

    def test_no_target_checks(self):
        rows = [{"isl": 128, "osl": 64}]
        flat = flatten_target_checks(rows)
        assert flat == [{"isl": 128, "osl": 64}]


# ── check_text_vlm_targets (integration) ─────────────────────────────


class TestCheckTextVlmTargets:
    def test_text_targets_matched(self):
        vllm_results = [
            {
                "input_sequence_length": 128,
                "output_sequence_length": 128,
                "max_con": 1,
                "mean_ttft_ms": 50.0,
                "mean_tps": 80.0,
                "tps_decode_throughput": 400.0,
                "task_type": "text",
                "backend": "vllm",
            },
        ]
        perf_refs = [
            BenchmarkTaskParams(
                isl=128,
                osl=128,
                max_concurrency=1,
                theoretical_ttft_ms=100.0,
                theoretical_tput_user=100.0,
            ),
        ]
        results, md = check_text_vlm_targets(
            vllm_results, perf_refs, "text", "test-model", "n150",
        )

        assert len(results) == 1
        assert "target_checks" in results[0]
        assert "Text-to-Text Performance Benchmark Targets" in md
        assert results[0]["ttft"] == 50.0

    def test_missing_benchmark_row(self):
        vllm_results = [
            {
                "input_sequence_length": 256,
                "output_sequence_length": 256,
                "max_con": 1,
                "mean_ttft_ms": 50.0,
                "mean_tps": 80.0,
                "tps_decode_throughput": 400.0,
                "task_type": "text",
                "backend": "vllm",
            },
        ]
        perf_refs = [
            BenchmarkTaskParams(
                isl=128,
                osl=128,
                max_concurrency=1,
                theoretical_ttft_ms=100.0,
                theoretical_tput_user=100.0,
            ),
        ]
        results, md = check_text_vlm_targets(
            vllm_results, perf_refs, "text", "test-model", "n150",
        )

        assert results[0]["ttft"] == "N/A"
        for checks in results[0]["target_checks"].values():
            assert checks["ttft_check"] == ReportCheckTypes.NA
