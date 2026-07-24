# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the prefix-cache parser SLA verdicts and the customer trace.

Covers:
* ``_compute_sla_checks`` -- per-run PASS/FAIL against the customer KPIs
  (TTFT P50/P90/P99, output speed, hit rate), including the None-safe
  "incomplete data" handling.
* The bundled ``customer_mooncake.jsonl`` fixture is byte-stable with its
  generator (deterministic seed), so the committed trace stays in sync.
"""

from __future__ import annotations

import importlib.util

from workflows.utils import get_repo_root_path

from llm_module.parsers.aiperf_prefix_cache import (
    SLA_HIT_RATE_MIN,
    SLA_OUTPUT_SPEED_MIN_TPS_PER_USER,
    SLA_TTFT_P50_MAX_MS,
    SLA_TTFT_P90_MAX_MS,
    SLA_TTFT_P99_MAX_MS,
    _compute_sla_checks,
)

_TRACE_DIR = get_repo_root_path() / "llm_module" / "prefix_cache" / "sample_traces"


def _passing_metrics() -> dict:
    return {
        "median_ttft_ms": SLA_TTFT_P50_MAX_MS - 1,
        "p90_ttft_ms": SLA_TTFT_P90_MAX_MS - 1,
        "p99_ttft_ms": SLA_TTFT_P99_MAX_MS - 1,
        "output_token_throughput_per_user": SLA_OUTPUT_SPEED_MIN_TPS_PER_USER + 1,
        "prefix_cache_hit_rate": SLA_HIT_RATE_MIN + 0.05,
    }


class TestComputeSlaChecks:
    def test_all_pass(self):
        checks = _compute_sla_checks(_passing_metrics())
        assert checks["sla_ttft_p50_pass"] is True
        assert checks["sla_ttft_p90_pass"] is True
        assert checks["sla_ttft_p99_pass"] is True
        assert checks["sla_output_speed_pass"] is True
        assert checks["sla_hit_rate_pass"] is True
        assert checks["sla_pass"] is True

    def test_one_fail_makes_overall_fail(self):
        metrics = _passing_metrics()
        metrics["p99_ttft_ms"] = SLA_TTFT_P99_MAX_MS + 1  # blow the P99 ceiling
        checks = _compute_sla_checks(metrics)
        assert checks["sla_ttft_p99_pass"] is False
        assert checks["sla_pass"] is False

    def test_output_speed_below_target_fails(self):
        metrics = _passing_metrics()
        metrics["output_token_throughput_per_user"] = (
            SLA_OUTPUT_SPEED_MIN_TPS_PER_USER - 1
        )
        checks = _compute_sla_checks(metrics)
        assert checks["sla_output_speed_pass"] is False
        assert checks["sla_pass"] is False

    def test_missing_hit_rate_makes_overall_incomplete(self):
        # Hit-rate unavailable (worker /metrics down) -> None check ->
        # overall is None (incomplete), not False, when nothing failed.
        metrics = _passing_metrics()
        metrics["prefix_cache_hit_rate"] = None
        checks = _compute_sla_checks(metrics)
        assert checks["sla_hit_rate_pass"] is None
        assert checks["sla_pass"] is None

    def test_fail_dominates_missing(self):
        metrics = _passing_metrics()
        metrics["prefix_cache_hit_rate"] = None  # incomplete
        metrics["median_ttft_ms"] = SLA_TTFT_P50_MAX_MS + 1  # fail
        checks = _compute_sla_checks(metrics)
        assert checks["sla_pass"] is False

    def test_aggregated_run_has_no_role_checks(self):
        # No prefill/decode data -> no role checks, overall unchanged.
        checks = _compute_sla_checks(_passing_metrics())
        assert "sla_hit_rate_prefill_pass" not in checks
        assert "sla_hit_rate_decode_pass" not in checks
        assert checks["sla_pass"] is True

    def test_disagg_both_roles_pass(self):
        metrics = _passing_metrics()
        metrics["prefix_cache_hit_rate_prefill"] = SLA_HIT_RATE_MIN + 0.02
        metrics["prefix_cache_hit_rate_decode"] = SLA_HIT_RATE_MIN + 0.03
        checks = _compute_sla_checks(metrics)
        assert checks["sla_hit_rate_prefill_pass"] is True
        assert checks["sla_hit_rate_decode_pass"] is True
        assert checks["sla_pass"] is True

    def test_disagg_prefill_informational_does_not_gate(self):
        # Decode gates; prefill is informational. A failing prefill is still
        # reported (False) but must NOT sink the overall verdict when decode
        # (the gating role) passes. Prefill structurally handles misses.
        metrics = _passing_metrics()
        metrics["prefix_cache_hit_rate_prefill"] = SLA_HIT_RATE_MIN - 0.50
        metrics["prefix_cache_hit_rate_decode"] = SLA_HIT_RATE_MIN + 0.05
        checks = _compute_sla_checks(metrics)
        assert checks["sla_hit_rate_prefill_pass"] is False  # reported
        assert checks["sla_hit_rate_decode_pass"] is True
        assert checks["sla_pass"] is True  # decode gates, prefill ignored

    def test_disagg_decode_below_target_fails_overall(self):
        # Decode is the gating role: if it misses, overall fails regardless of
        # prefill (and regardless of the blended combined rate).
        metrics = _passing_metrics()
        metrics["prefix_cache_hit_rate_prefill"] = SLA_HIT_RATE_MIN + 0.05
        metrics["prefix_cache_hit_rate_decode"] = SLA_HIT_RATE_MIN - 0.10
        checks = _compute_sla_checks(metrics)
        assert checks["sla_hit_rate_decode_pass"] is False
        assert checks["sla_pass"] is False

    def test_disagg_decode_supersedes_blended_combined(self):
        # Combined blend would PASS (weighted avg high) but decode itself is
        # below target -> overall must FAIL on decode, not be rescued by the
        # blended combined rate.
        metrics = _passing_metrics()
        metrics["prefix_cache_hit_rate"] = SLA_HIT_RATE_MIN + 0.05  # blend passes
        metrics["prefix_cache_hit_rate_decode"] = SLA_HIT_RATE_MIN - 0.10  # fails
        checks = _compute_sla_checks(metrics)
        assert checks["sla_hit_rate_pass"] is True  # combined still reported
        assert checks["sla_pass"] is False  # but decode gates


def _load_generator():
    path = _TRACE_DIR / "generate_customer_mooncake.py"
    spec = importlib.util.spec_from_file_location("gen_customer_mooncake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCustomerTraceFixture:
    def test_committed_trace_matches_generator(self, tmp_path, monkeypatch):
        gen = _load_generator()
        committed = (_TRACE_DIR / "customer_mooncake.jsonl").read_bytes()
        out = tmp_path / "customer_mooncake.jsonl"
        monkeypatch.setattr(gen, "OUT_PATH", out)
        gen.main()
        assert out.read_bytes() == committed, (
            "customer_mooncake.jsonl is stale; regenerate with "
            "generate_customer_mooncake.py"
        )

    def test_trace_shape_hits_90pct_reuse(self):
        gen = _load_generator()
        reuse = gen.SHARED_PREFIX_TOKENS / gen.INPUT_LENGTH
        assert reuse >= 0.90
