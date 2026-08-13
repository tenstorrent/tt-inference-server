# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the scaling-quality three-point coverage rule (readiness §5.7).

The scaling-quality rubric line fits time-to-first-token against input length
*separately at each graded concurrency level* (RFP Appendix B.1/B.2/F.1). A
regression needs at least three points, so every graded concurrency level must
carry at least three distinct input lengths. These tests pin the validator, the
feasibility analyzer that sizes the device token pool, and the fail-fast guard
in ``get_llm_configs``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_module.benchmark_configs import _enforce_scaling_quality_coverage
from reference_config.benchmarking.benchmark_config import (
    SCALING_QUALITY_MIN_INPUT_LENGTHS,
    input_lengths_reaching_concurrency,
    max_gradeable_concurrency,
    min_token_pool_for_concurrency,
    scaling_quality_coverage,
    scaling_quality_coverage_violations,
)
from workflows.workflow_types import DeviceTypes

# The Milestone-0 Blackhole Galaxy device configuration (dev/llm.yaml).
M0_MAX_CONTEXT = 262144
M0_MAX_CONCURRENCY = 128


# ── validator ────────────────────────────────────────────────────────────────


def test_coverage_groups_distinct_input_lengths_by_concurrency():
    graded = [(128, 1), (1024, 1), (128, 1), (128, 128), (1024, 128)]
    assert scaling_quality_coverage(graded) == {1: [128, 1024], 128: [128, 1024]}


def test_violation_flags_concurrency_with_too_few_input_lengths():
    # conc 1 has 3 ISLs (ok); conc 128 has only 2 ISLs (fails the fit).
    graded = [
        (128, 1),
        (1024, 1),
        (2048, 1),
        (128, 128),
        (1024, 128),
    ]
    violations = scaling_quality_coverage_violations(graded)
    assert violations == {128: [128, 1024]}


def test_no_violation_when_every_level_has_three_input_lengths():
    graded = [
        (128, 1),
        (1024, 1),
        (2048, 1),
        (128, 128),
        (1024, 128),
        (2048, 128),
    ]
    assert scaling_quality_coverage_violations(graded) == {}


def test_validator_ignores_missing_fields():
    assert scaling_quality_coverage_violations([(None, 1), (128, None)]) == {}


# ── feasibility analyzer ──────────────────────────────────────────────────────


def test_native_context_pool_starves_top_concurrency():
    """With the token pool defaulted to max_context, only ISLs 128 and 1024 reach
    concurrency 128 — the exact §5.7 failure this issue guards against."""
    reachable = input_lengths_reaching_concurrency(
        M0_MAX_CONCURRENCY,
        max_context=M0_MAX_CONTEXT,
        max_tokens_all_users=M0_MAX_CONTEXT,
        model_max_concurrency=M0_MAX_CONCURRENCY,
    )
    assert reachable == [128, 1024]
    assert len(reachable) < SCALING_QUALITY_MIN_INPUT_LENGTHS


def test_max_gradeable_concurrency_is_third_largest_ceiling():
    # Per-ISL ceilings at pool=262144 are [128,128,120,62,...]; the third largest
    # (the highest concurrency 3 distinct ISLs can share) is 120.
    assert (
        max_gradeable_concurrency(
            max_context=M0_MAX_CONTEXT,
            max_tokens_all_users=M0_MAX_CONTEXT,
            model_max_concurrency=M0_MAX_CONCURRENCY,
        )
        == 120
    )


def test_min_token_pool_lifts_third_input_length_to_top_concurrency():
    pool = min_token_pool_for_concurrency(
        M0_MAX_CONCURRENCY, max_context=M0_MAX_CONTEXT
    )
    assert pool == 128 * 2176  # 278528: concurrency * (isl=2048 + osl=128)

    reachable = input_lengths_reaching_concurrency(
        M0_MAX_CONCURRENCY,
        max_context=M0_MAX_CONTEXT,
        max_tokens_all_users=pool,
        model_max_concurrency=M0_MAX_CONCURRENCY,
    )
    assert reachable[:3] == [128, 1024, 2048]
    assert len(reachable) >= SCALING_QUALITY_MIN_INPUT_LENGTHS


# ── device flag ───────────────────────────────────────────────────────────────


def test_only_blackhole_galaxy_grades_scaling_quality():
    assert DeviceTypes.BLACKHOLE_GALAXY.grades_scaling_quality() is True
    for device in (DeviceTypes.GALAXY, DeviceTypes.T3K, DeviceTypes.P150X4):
        assert device.grades_scaling_quality() is False


# ── get_llm_configs guard ─────────────────────────────────────────────────────


def _spec(max_tokens_all_users=M0_MAX_CONTEXT):
    return SimpleNamespace(
        model_id="stub/model",
        device_model_spec=SimpleNamespace(
            max_context=M0_MAX_CONTEXT,
            max_tokens_all_users=max_tokens_all_users,
            max_concurrency=M0_MAX_CONCURRENCY,
        ),
    )


def _cfg(isl, max_concurrency, graded=True):
    return SimpleNamespace(
        isl=isl,
        max_concurrency=max_concurrency,
        targets={"target": object()} if graded else {},
    )


def test_guard_raises_on_undergraded_top_concurrency():
    configs = [
        _cfg(128, 1),
        _cfg(1024, 1),
        _cfg(2048, 1),
        _cfg(128, 128),
        _cfg(1024, 128),
    ]
    with pytest.raises(ValueError) as exc:
        _enforce_scaling_quality_coverage(
            _spec(), DeviceTypes.BLACKHOLE_GALAXY, configs
        )
    msg = str(exc.value)
    assert "concurrency=128" in msg
    # Actionable hint points at the token-pool floor and the reachable ceiling.
    assert "278528" in msg
    assert "concurrency<=120" in msg


def test_guard_passes_when_top_concurrency_has_three_input_lengths():
    configs = [
        _cfg(128, 1),
        _cfg(1024, 1),
        _cfg(2048, 1),
        _cfg(128, 128),
        _cfg(1024, 128),
        _cfg(2048, 128),
    ]
    _enforce_scaling_quality_coverage(_spec(), DeviceTypes.BLACKHOLE_GALAXY, configs)


def test_guard_skips_non_scaling_quality_devices():
    configs = [_cfg(128, 128), _cfg(1024, 128)]  # would violate on BH Galaxy
    _enforce_scaling_quality_coverage(_spec(), DeviceTypes.GALAXY, configs)


def test_guard_ignores_ungraded_sweep_points():
    # Sweep points without targets are not graded, so the rule does not apply.
    configs = [_cfg(128, 128, graded=False), _cfg(1024, 128, graded=False)]
    _enforce_scaling_quality_coverage(_spec(), DeviceTypes.BLACKHOLE_GALAXY, configs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
