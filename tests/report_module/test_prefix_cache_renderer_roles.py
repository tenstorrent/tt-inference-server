# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Renderer tests for the disaggregated (prefill/decode) hit-rate columns.

The prefill/decode columns are injected only when the sweep actually
carries role data, so aggregated deployments render exactly as before.
"""

from __future__ import annotations

from report_module.prefix_cache_renderer import render_aiperf_prefix_cache
from report_module.schema import Block


def _block(records: list) -> Block:
    return Block(
        id="prefix_cache",
        kind="aiperf_prefix_cache",
        data={"records": records},
    )


def _base_record(**overrides) -> dict:
    record = {
        "kind": "aiperf_prefix_cache",
        "scenario": "shared_system",
        "label": "run0",
        "concurrency": 8,
        "arrival_pattern": "poisson",
        "isl_mean": 1024,
        "osl_mean": 128,
        "request_count": 100,
        "prefix_cache_hit_rate": 0.55,
        "prefix_cache_hit_rate_pct": 55.0,
        "mean_ttft_ms": 100.0,
        "median_ttft_ms": 90.0,
        "p90_ttft_ms": 120.0,
        "p99_ttft_ms": 150.0,
    }
    record.update(overrides)
    return record


class TestRoleColumns:
    def test_columns_absent_for_aggregated_run(self):
        md = render_aiperf_prefix_cache(_block([_base_record()]), {})
        assert "Cache Hit %" in md
        assert "Prefill Hit %" not in md
        assert "Decode Hit %" not in md

    def test_columns_present_for_disaggregated_run(self):
        record = _base_record(
            prefix_cache_hit_rate_prefill_pct=20.0,
            prefix_cache_hit_rate_decode_pct=90.0,
        )
        md = render_aiperf_prefix_cache(_block([record]), {})
        assert "Prefill Hit %" in md
        assert "Decode Hit %" in md
        # Combined column is still shown alongside the per-role split.
        assert "Cache Hit %" in md
        # Values rendered to one decimal.
        assert "20.0" in md
        assert "90.0" in md

    def test_sla_table_decode_gates_prefill_informational(self):
        # Decode failing, prefill "passing" but informational: the SLA table
        # shows a gated Decode column (FAIL) and an informational Prefill
        # column (percentage, no PASS/FAIL). Overall follows decode.
        record = _base_record(
            prefix_cache_hit_rate_prefill_pct=30.0,
            prefix_cache_hit_rate_decode_pct=95.0,
            sla_ttft_p50_pass=True,
            sla_hit_rate_pass=True,
            sla_hit_rate_prefill_pass=False,
            sla_hit_rate_decode_pass=True,
            sla_pass=True,
        )
        md = render_aiperf_prefix_cache(_block([record]), {})
        # Decode is the gated column.
        assert "Decode Hit % (>=90)" in md
        # Prefill is informational, not a >=90 gate.
        assert "Prefill Hit % (info)" in md
        assert "Prefill Hit % (>=90)" not in md
        # The blended combined "Hit % (>=90)" gate is replaced by decode.
        assert "| Hit % (>=90) " not in md
