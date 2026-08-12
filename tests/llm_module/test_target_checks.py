# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for per-run LLM benchmark grading (``llm_module.target_checks``).

Covers the grading itself and the contract that matters downstream: a
graded block reaches ``report_module.acceptance_criteria`` with the
``target_checks`` it needs, and an ungraded sweep point is reported as
NA rather than passing silently.
"""

from __future__ import annotations

import pytest

from llm_module.config import LLMRunConfig
from llm_module.parsers.vllm import VLLMBenchParser
from llm_module.target_checks import apply_target_checks, build_target_checks
from report_module import acceptance_criteria_check
from report_module.schema import Block, ReportSchema
from workflows.utils_report import PerformanceTarget
from workflows.workflow_types import ReportCheckTypes

TARGETS = {
    "functional": PerformanceTarget(ttft_ms=890.0, tput_user=1.7, tput=1.7),
    "complete": PerformanceTarget(ttft_ms=178.0, tput_user=8.5, tput=8.5),
    "target": PerformanceTarget(ttft_ms=89.0, tput_user=17.0, tput=17.0),
}


def _record(ttft=350.0, tpot=69.8, tput=13.9, **extra):
    record = {
        "concurrency": 1,
        "num_requests": 8,
        "input_sequence_length": 128,
        "output_sequence_length": 128.0,
        "mean_ttft_ms": ttft,
        "mean_tpot_ms": tpot,
        "tps_decode_throughput": tput,
        # A parsed record always carries this. 0 means the point served every
        # request, which acceptance now requires (RFP G.2.6).
        "error_request_count": 0,
    }
    record.update(extra)
    return record


def _cfg(targets=None, isl=128, osl=128, max_concurrency=1):
    return LLMRunConfig(
        isl=isl,
        osl=osl,
        max_concurrency=max_concurrency,
        num_prompts=8,
        targets=dict(targets or {}),
    )


def _block(record=None, kind="benchmarks", title="vLLM Benchmark"):
    return Block(
        kind=kind, data=record if record is not None else _record(), title=title
    )


class TestBuildTargetChecks:
    def test_grades_every_tier_per_metric(self):
        checks, verdict = build_target_checks(TARGETS, _record())

        assert set(checks) == {"functional", "complete", "target"}
        # 350ms TTFT clears the 890ms functional bar but not the 178ms one.
        assert checks["functional"]["ttft_check"] == ReportCheckTypes.PASS
        assert checks["complete"]["ttft_check"] == ReportCheckTypes.FAIL
        assert checks["functional"]["ttft"] == 890.0
        assert checks["functional"]["ttft_ratio"] == pytest.approx(350.0 / 890.0)
        # verdict follows the strictest tier only
        assert verdict == ReportCheckTypes.FAIL

    def test_passes_when_strictest_tier_is_met(self):
        _, verdict = build_target_checks(
            TARGETS, _record(ttft=50.0, tpot=40.0, tput=25.0)
        )
        assert verdict == ReportCheckTypes.PASS

    def test_derives_tput_user_from_tpot_when_tool_omits_it(self):
        """``vllm bench serve`` reports no per-user throughput; TPOT gives it."""
        checks, _ = build_target_checks(TARGETS, _record(tpot=100.0))
        # 1000 / 100ms = 10 tok/s/user against the 8.5 "complete" target
        assert checks["complete"]["tput_user_ratio"] == pytest.approx(10.0 / 8.5)
        assert checks["complete"]["tput_user_check"] == ReportCheckTypes.PASS

    def test_prefers_the_tools_own_tput_user(self):
        checks, _ = build_target_checks(TARGETS, _record(tpot=100.0, tput_user=4.0))
        assert checks["complete"]["tput_user_ratio"] == pytest.approx(4.0 / 8.5)

    def test_tolerance_widens_the_bar(self):
        tolerant = {"target": PerformanceTarget(ttft_ms=100.0, tolerance=0.2)}
        strict = {"target": PerformanceTarget(ttft_ms=100.0, tolerance=0.0)}
        record = _record(ttft=110.0)

        assert build_target_checks(tolerant, record)[1] == ReportCheckTypes.PASS
        assert build_target_checks(strict, record)[1] == ReportCheckTypes.FAIL

    def test_undefined_target_and_unmeasured_metric_are_na_not_failures(self):
        targets = {"target": PerformanceTarget(ttft_ms=89.0)}  # no tput targets
        checks, _ = build_target_checks(targets, {"mean_ttft_ms": None})

        assert checks["target"]["ttft_check"] == ReportCheckTypes.NA
        assert checks["target"]["tput_check"] == ReportCheckTypes.NA
        assert checks["target"]["tput_user_check"] == ReportCheckTypes.NA


class TestApplyTargetChecks:
    def test_graded_block_carries_checks_and_a_per_config_title(self):
        block = apply_target_checks(_block(), _cfg(TARGETS))

        assert (
            block.data["target_checks"]["target"]["ttft_check"] == ReportCheckTypes.FAIL
        )
        assert block.data["target_check"] == ReportCheckTypes.FAIL
        # per-config title: keeps blocker keys distinct and keeps the nested
        # tier table out of the collapsed sweep table
        assert (
            block.title == "vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1"
        )
        assert "status" not in block.data

    def test_ungraded_sweep_point_is_na_and_keeps_the_shared_title(self):
        block = apply_target_checks(_block(), _cfg())

        assert block.data["status"] == "na"
        assert block.data["target_check"] == ReportCheckTypes.NA
        assert "target_checks" not in block.data
        assert block.title == "vLLM Benchmark"

    def test_leaves_non_benchmark_kinds_untouched(self):
        """Prefix-cache / spec-decode blocks keep their own kind and shape."""
        block = _block(kind="aiperf_prefix_cache")
        assert apply_target_checks(block, _cfg(TARGETS)) is block

    def test_grades_a_real_parsed_vllm_block(self):
        raw = {
            "model_id": "meta-llama/Llama-3.3-70B-Instruct",
            "date": "20260727-035000",
            "completed": 8,
            "max_concurrency": 1,
            "total_input_tokens": 1024,
            "total_output_tokens": 1024,
            "mean_ttft_ms": 350.3233,
            "mean_tpot_ms": 69.809,
            "output_throughput": 13.8884,
        }
        block = apply_target_checks(
            VLLMBenchParser().parse(raw, device="P300X2"), _cfg(TARGETS)
        )
        assert block.kind == "benchmarks"
        assert block.data["target_checks"]["functional"]["ttft_check"] == (
            ReportCheckTypes.PASS
        )


class TestAcceptanceIntegration:
    """The point of the exercise: benchmarks reach the acceptance verdict."""

    @staticmethod
    def _benchmarks(*blocks):
        schema = ReportSchema(
            metadata={"report_id": "r", "model_name": "m", "device": "d"},
            sections=list(blocks),
        )
        accepted, blockers, categories = acceptance_criteria_check(schema)
        category = next(c for c in categories if c.name == "Benchmarks")
        return accepted, blockers, category

    def test_a_regressed_sweep_point_blocks_the_run(self):
        regressed = apply_target_checks(
            _block(_record(ttft=35000.0, tpot=6980.0, tput=0.14)), _cfg(TARGETS)
        )
        accepted, blockers, category = self._benchmarks(regressed)

        assert accepted is False
        assert category.status == "FAIL"
        assert any("ttft_check" in key for key in blockers)

    def test_a_met_target_passes_with_the_block_counted(self):
        graded = apply_target_checks(
            _block(_record(ttft=50.0, tpot=40.0, tput=25.0)), _cfg(TARGETS)
        )
        accepted, _, category = self._benchmarks(graded)

        assert accepted is True
        assert (category.status, category.total, category.passed) == ("PASS", 1, 1)

    def test_ungraded_blocks_are_na_not_a_silent_pass(self):
        ungraded = apply_target_checks(_block(), _cfg())
        accepted, blockers, category = self._benchmarks(ungraded)

        # non-blocking, but visible: 1 block present and ungraded, never
        # "no blocks present" and never a PASS nobody measured
        assert (accepted, blockers) == (True, {})
        assert (category.status, category.total, category.na) == ("NA", 1, 1)

    def test_blocker_keys_stay_distinct_across_graded_sweep_points(self):
        first = apply_target_checks(
            _block(_record(ttft=35000.0, tpot=6980.0, tput=0.14)), _cfg(TARGETS)
        )
        second = apply_target_checks(
            _block(_record(ttft=35000.0, tpot=6980.0, tput=0.14)),
            _cfg(TARGETS, isl=1024, max_concurrency=32),
        )
        _, blockers, category = self._benchmarks(first, second)

        assert category.failed == 2
        assert sum("ISL 1024" in key for key in blockers) > 0
        assert sum("ISL 128" in key for key in blockers) > 0
