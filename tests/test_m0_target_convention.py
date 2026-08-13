# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the Milestone-0 single-tier target configuration.

See ``docs/rfp/m0-target-convention.md``. The thing being defended here is that
the acceptance gate actually *bites*: before this configuration all three
Milestone-0 models sat at ``EXPERIMENTAL``, which enforces no target tiers and —
because ``evals_enforced`` reuses the same signal — waives accuracy failures too.
A submission could fail everything and still be accepted.

Two failure modes are silent and both fail in the safe-looking direction, so they
are asserted directly rather than left to follow from behaviour:

* a tier whose name is not in ``status.required_target_tiers`` is computed,
  reported and then ignored;
* ``_verdict`` reading a tier by a hardcoded name returns ``NA`` when that name is
  absent, and an ``NA`` point is counted ungradable rather than failed.
"""

from __future__ import annotations

import pytest

from llm_module.target_checks import apply_target_checks, graded_tier
from report_module.acceptance_criteria import acceptance_criteria_check
from report_module.schema import Block, ReportSchema
from report_module.submission import graded_points
from workflows.model_spec import get_perf_reference_map
from workflows.utils_report import BenchmarkTaskParams, PerformanceTarget
from workflows.workflow_types import DeviceTypes, ModelStatusTypes, ReportCheckTypes

M0_TIER = "functional"
M0_STATUS = ModelStatusTypes.FUNCTIONAL
M0_TOLERANCE = 0.10

# The DeepSeek-V4-Flash-0731 mid point, downrated per the convention.
TTFT_TARGET, TPUT_USER_TARGET, TPUT_TARGET = 2820.0, 25.0, 1603.0


def _cfg(tolerance: float = M0_TOLERANCE, tier: str = M0_TIER) -> BenchmarkTaskParams:
    return BenchmarkTaskParams(
        isl=8192,
        osl=128,
        max_concurrency=64,
        num_prompts=128,
        targets={
            tier: PerformanceTarget(
                ttft_ms=TTFT_TARGET,
                tput_user=TPUT_USER_TARGET,
                tput=TPUT_TARGET,
                tolerance=tolerance,
            )
        },
    )


def _graded_block(ttft_ms: float, cfg: BenchmarkTaskParams | None = None) -> Block:
    cfg = cfg or _cfg()
    record = {
        "concurrency": cfg.max_concurrency,
        "input_sequence_length": cfg.isl,
        "output_sequence_length": cfg.osl,
        "mean_ttft_ms": ttft_ms,
        "tput_user": TPUT_USER_TARGET,
        "tps_decode_throughput": TPUT_TARGET,
        "error_request_count": 0,
    }
    return apply_target_checks(
        Block(kind="benchmarks", id="b", title="B", data=record, targets={}), cfg
    )


def _accepts(ttft_ms: float, status: str, cfg=None) -> tuple[bool, list[str]]:
    accepted, blockers, _ = acceptance_criteria_check(
        ReportSchema(
            metadata={"report_id": "r"}, sections=[_graded_block(ttft_ms, cfg)]
        ),
        model_status=status,
    )
    return accepted, sorted(blockers)


# --------------------------------------------------------------------------
# The tier name must be one the status enforces
# --------------------------------------------------------------------------


def test_the_m0_tier_name_is_enforced_by_the_m0_status():
    """The pairing the whole configuration rests on. Rename one alone and the
    bar is computed, reported and silently ignored."""
    assert {M0_TIER} <= set(M0_STATUS.required_target_tiers)


def test_experimental_enforces_neither_targets_nor_evals():
    """Why the status had to change at all — this is the hole being closed."""
    assert ModelStatusTypes.EXPERIMENTAL.required_target_tiers == []
    assert ModelStatusTypes.EXPERIMENTAL.evals_enforced is False
    assert M0_STATUS.evals_enforced is True


def test_a_tier_the_status_does_not_enforce_is_ignored():
    """Concrete proof of the trap: same failure, tier renamed, silently passes."""
    over_target = TTFT_TARGET * 1.20
    assert _accepts(over_target, M0_STATUS.name)[0] is False
    # Only the tier name differs.
    assert _accepts(over_target, M0_STATUS.name, cfg=_cfg(tier="target"))[0] is True


# --------------------------------------------------------------------------
# The gate bites
# --------------------------------------------------------------------------


def test_a_point_over_target_blocks_acceptance():
    accepted, blockers = _accepts(TTFT_TARGET * 1.20, M0_STATUS.name)
    assert accepted is False
    assert any("ttft_check" in b for b in blockers)


def test_the_same_point_is_waived_at_experimental():
    """Isolates the status as the cause, so the fix cannot be misattributed."""
    assert _accepts(TTFT_TARGET * 1.20, "EXPERIMENTAL")[0] is True


def test_a_point_within_tolerance_passes():
    assert _accepts(TTFT_TARGET * 1.05, M0_STATUS.name)[0] is True


def test_a_point_beating_target_passes():
    assert _accepts(TTFT_TARGET * 0.90, M0_STATUS.name)[0] is True


def test_zero_tolerance_fails_the_same_point_that_ten_percent_passes():
    """Proves the published 0.10 is live rather than inherited."""
    five_percent_over = TTFT_TARGET * 1.05
    assert (
        _accepts(five_percent_over, M0_STATUS.name, cfg=_cfg(tolerance=0.0))[0] is False
    )
    assert (
        _accepts(five_percent_over, M0_STATUS.name, cfg=_cfg(tolerance=0.10))[0] is True
    )


# --------------------------------------------------------------------------
# The verdict must not hardcode a tier name
# --------------------------------------------------------------------------


def test_verdict_grades_the_strictest_tier_present_not_one_named_target():
    """Regression: reading "target" by name returned NA for a single-tier spec,
    and an NA point is counted ungradable rather than failed — so a point missing
    its target by any margin was accepted."""
    block = _graded_block(TTFT_TARGET * 1.20)
    assert block.data["target_checks"][M0_TIER]["ttft_check"] == ReportCheckTypes.FAIL
    assert block.data["target_check"] == ReportCheckTypes.FAIL


def test_verdict_still_prefers_target_when_the_full_ladder_is_present():
    """Backward compatibility for every model on the default ladder."""
    cfg = BenchmarkTaskParams(
        isl=8192,
        osl=128,
        max_concurrency=64,
        num_prompts=128,
        targets={
            # Passes the weak tier, fails the strict one. The verdict must be FAIL.
            "functional": PerformanceTarget(ttft_ms=TTFT_TARGET * 10),
            "target": PerformanceTarget(ttft_ms=TTFT_TARGET),
        },
    )
    block = _graded_block(TTFT_TARGET * 1.20, cfg)
    assert (
        block.data["target_checks"]["functional"]["ttft_check"] == ReportCheckTypes.PASS
    )
    assert block.data["target_check"] == ReportCheckTypes.FAIL


# --------------------------------------------------------------------------
# Deriving tiers from the reference file
# --------------------------------------------------------------------------


def test_a_single_tier_map_yields_one_tier_holding_the_value_verbatim():
    ref = get_perf_reference_map("DeepSeek-V4-Flash-0731", {M0_TIER: 1.0})
    points = ref[DeviceTypes.BLACKHOLE_GALAXY]
    assert points, "no blackhole_galaxy points resolved"
    for point in points:
        assert set(point.targets) == {M0_TIER}

    mid = next(p for p in points if p.isl == 8192)
    target = mid.targets[M0_TIER]
    assert (target.ttft_ms, target.tput_user, target.tput) == (
        TTFT_TARGET,
        TPUT_USER_TARGET,
        TPUT_TARGET,
    )


def test_tolerance_is_read_from_the_reference_entry():
    ref = get_perf_reference_map("DeepSeek-V4-Flash-0731", {M0_TIER: 1.0})
    for point in ref[DeviceTypes.BLACKHOLE_GALAXY]:
        assert point.targets[M0_TIER].tolerance == pytest.approx(M0_TOLERANCE)


def test_an_entry_without_a_tolerance_still_defaults_to_zero():
    """No existing model's behaviour changes by adding the field."""
    ref = get_perf_reference_map("Mistral-Small-3.1-24B-Instruct-2503", {"target": 1.0})
    points = [p for p in ref[DeviceTypes.BLACKHOLE_GALAXY] if p.targets]
    assert points
    assert all(p.targets["target"].tolerance == 0.0 for p in points)


def test_the_device_override_beats_the_model_wide_ladder():
    """DeviceModelSpec.perf_targets_map existed but was never read when deriving
    tiers, so setting it did nothing. Milestone-0 needs it: one model's
    BLACKHOLE_GALAXY row grades on a single tier while its other devices keep the
    ordinary ladder."""
    ref = get_perf_reference_map(
        "DeepSeek-V4-Flash-0731",
        {"functional": 0.10, "complete": 0.50, "target": 1.0},
        {DeviceTypes.BLACKHOLE_GALAXY: {M0_TIER: 1.0}},
    )
    for point in ref[DeviceTypes.BLACKHOLE_GALAXY]:
        assert set(point.targets) == {M0_TIER}
        assert point.targets[M0_TIER].ttft_ms != TTFT_TARGET / 0.10


# --------------------------------------------------------------------------
# The dev catalog is actually configured this way
# --------------------------------------------------------------------------

M0_WEIGHTS = (
    "google/gemma-4-31B-it",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "mistralai/Mistral-Small-4-119B-2603",
)


def _m0_templates():
    """The dev templates carrying a BLACKHOLE_GALAXY row, read straight from YAML.

    Parsed directly rather than through the catalog loader so the test asserts
    what is committed, independently of MODEL_SPECS_ENV.
    """
    import yaml

    from workflows.utils import get_repo_root_path

    path = get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    templates = yaml.safe_load(path.read_text())["templates"]
    return [
        t
        for t in templates
        if any(
            d.get("device") == "BLACKHOLE_GALAXY"
            for d in (t.get("device_model_specs") or [])
        )
        and t["weights"][0] in M0_WEIGHTS
    ]


def test_all_three_m0_models_have_a_blackhole_galaxy_spec():
    assert {t["weights"][0] for t in _m0_templates()} == set(M0_WEIGHTS)


@pytest.mark.parametrize("weights", M0_WEIGHTS)
def test_each_m0_spec_pairs_its_tier_name_with_an_enforcing_status(weights):
    """The invariant that keeps the gate live. Renaming the tier or lowering the
    status alone leaves the bar computed, reported and ignored."""
    template = next(t for t in _m0_templates() if t["weights"][0] == weights)
    status = ModelStatusTypes.resolve(template["status"])
    assert status is not None, template["status"]

    row = next(
        d for d in template["device_model_specs"] if d["device"] == "BLACKHOLE_GALAXY"
    )
    tiers = set(row.get("perf_targets_map") or {})
    assert tiers, f"{weights} BLACKHOLE_GALAXY row defines no perf_targets_map"
    assert tiers <= set(status.required_target_tiers), (
        f"{weights}: tier(s) {tiers} are not enforced by status {status.name} "
        f"(enforces {status.required_target_tiers}) — the bar would be ignored"
    )
    assert status.evals_enforced, f"{weights}: evals must be enforced at Milestone-0"


@pytest.mark.parametrize("weights", M0_WEIGHTS)
def test_each_m0_spec_grades_against_exactly_one_tier_at_full_value(weights):
    template = next(t for t in _m0_templates() if t["weights"][0] == weights)
    row = next(
        d for d in template["device_model_specs"] if d["device"] == "BLACKHOLE_GALAXY"
    )
    assert row["perf_targets_map"] == {M0_TIER: 1.0}


# --------------------------------------------------------------------------
# Every consumer of target_checks must resolve the tier, not name it
# --------------------------------------------------------------------------


def _report_with_tier(tier: str) -> dict:
    return {
        "sections": [
            {
                "kind": "benchmarks",
                "title": "B",
                "data": {
                    "concurrency": 64,
                    "input_sequence_length": 8192,
                    "p50_ttft": 2700.0,
                    "p90_ttft": 3100.0,
                    "p99_ttft": 3400.0,
                    "tput_user": TPUT_USER_TARGET,
                    "tps_decode_throughput": TPUT_TARGET,
                    "target_checks": {
                        tier: {
                            "ttft": TTFT_TARGET,
                            "ttft_check": ReportCheckTypes.PASS,
                            "tput_user": TPUT_USER_TARGET,
                            "tput": TPUT_TARGET,
                        }
                    },
                },
            }
        ]
    }


def test_graded_tier_picks_the_strictest_present():
    assert graded_tier({"functional": {"ttft": 9}})[0] == "functional"
    assert (
        graded_tier({"functional": {"ttft": 9}, "target": {"ttft": 1}})[0] == "target"
    )
    assert graded_tier({}) is None


@pytest.mark.parametrize("tier", ["functional", "target"])
def test_the_submission_assembler_finds_targets_under_either_tier_name(tier):
    """report_module.submission read target_checks["target"] by name, so under the
    Milestone-0 single-tier config it found no targets and silently dropped every
    graded point from the submission — producing a scorecard of zeros."""
    points = graded_points(_report_with_tier(tier))
    assert len(points) == 1
    assert points[0]["target_ttft_ms"] == TTFT_TARGET
    assert points[0]["target_tput_user"] == TPUT_USER_TARGET
    assert points[0]["target_decode_throughput"] == TPUT_TARGET


def test_without_an_override_the_default_ladder_still_derives_three_tiers():
    ref = get_perf_reference_map(
        "DeepSeek-V4-Flash-0731", {"functional": 0.10, "complete": 0.50, "target": 1.0}
    )
    mid = next(p for p in ref[DeviceTypes.BLACKHOLE_GALAXY] if p.isl == 8192)
    assert set(mid.targets) == {"functional", "complete", "target"}
    # Latency divided by the percentage, throughput multiplied by it.
    assert mid.targets["target"].ttft_ms == pytest.approx(TTFT_TARGET)
    assert mid.targets["functional"].ttft_ms == pytest.approx(TTFT_TARGET / 0.10)
    assert mid.targets["functional"].tput_user == pytest.approx(TPUT_USER_TARGET * 0.10)
