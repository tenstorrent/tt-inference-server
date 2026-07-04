# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for prefix-cache scenario/preset expansion.

Focuses on the ``highcache_50k`` preset, which encodes the customer
trillion-scale traffic shape: a 50K shared (cacheable) system prefix plus a
5K new ISL and 500 OSL at concurrency 32. The cacheable fraction of each
warm request is ``50000 / (50000 + 5000) = ~90.9%``, matching the customer's
>= 90% per-session KV-cache hit-rate requirement.
"""

from __future__ import annotations

from llm_module.prefix_cache import build_runs


def _run_by_scenario(preset: str):
    return {r.scenario: r for r in build_runs(preset=preset)}


def test_highcache_50k_shared_system_shape():
    runs = build_runs(preset="highcache_50k")
    by_scenario = {r.scenario: r for r in runs}

    assert set(by_scenario) == {"shared_system", "baseline"}

    shared = by_scenario["shared_system"]
    assert shared.shared_system_prompt_length == 50000
    assert shared.isl_mean == 5000
    assert shared.osl_mean == 500
    assert shared.concurrency == 32
    assert shared.arrival_pattern == "constant"
    assert shared.request_count == 256


def test_highcache_50k_hit_rate_is_at_least_90_percent():
    shared = _run_by_scenario("highcache_50k")["shared_system"]
    cacheable = shared.shared_system_prompt_length
    total = shared.shared_system_prompt_length + shared.isl_mean
    steady_state_hit_rate = cacheable / total
    assert steady_state_hit_rate >= 0.90


def test_highcache_50k_baseline_matches_load_minus_prefix():
    by_scenario = _run_by_scenario("highcache_50k")
    shared = by_scenario["shared_system"]
    baseline = by_scenario["baseline"]

    # The baseline isolates the TTFT uplift from caching: same new ISL / OSL /
    # concurrency, but no shared (cacheable) prefix.
    assert baseline.shared_system_prompt_length is None
    assert baseline.isl_mean == shared.isl_mean
    assert baseline.osl_mean == shared.osl_mean
    assert baseline.concurrency == shared.concurrency
