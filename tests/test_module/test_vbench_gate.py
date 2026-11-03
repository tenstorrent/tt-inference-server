# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The Wan2.2 I2V VBench gate's checks, verdict and config."""

from __future__ import annotations

import copy

import pytest

from test_module._test_common import vbench_gate as gate
from test_module._test_common.report_types import ReportCheckTypes

R1, R2 = "refs/r1", "refs/r2/tt"
CLIPS = [
    {
        "file": f"{i:02d}_vbi2v_{i:04d}_832x480.mp4",
        "video_id": f"vbi2v_{i:04d}_832x480",
        "width": 832,
        "height": 480,
    }
    for i in range(3)
]
CONFIG = {
    "models": ["Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
    "gated_dimensions": ["i2v_subject", "imaging_quality"],
    "reported_dimensions": ["dynamic_degree"],
    "tolerance_pct": {"drift": 1.0, "budget": 2.0},
    "output": {
        "480p": {
            "max_clip_lpips": 0.5,
            "mean_margin": {"psnr": 1.5, "ssim": 0.05, "lpips": 0.05},
            "calibration": {
                run: {
                    c["video_id"]: {"psnr": 22.0, "ssim": 0.7, "lpips": 0.2}
                    for c in CLIPS
                }
                for run in (R1, R2)
            },
        },
        "720p": None,
    },
}


def comparison(
    run,
    *,
    subject=(0.98, 0.98),
    imaging=(0.70, 0.70),
    dynamic=(0.5, 1.0),
    psnr=22.0,
    ssim=0.7,
    lpips=0.2,
):
    def dim(tt, gpu):
        return {
            "tt": tt,
            "gpu": gpu,
            "delta": tt - gpu,
            "clips": len(CLIPS),
            "per_clip": {c["file"]: tt - gpu for c in CLIPS},
        }

    return {
        "reference": f"Tenstorrent/wan2.2-i2v-vbench-refs/{run}",
        "dimensions": {
            "i2v_subject": dim(*subject),
            "imaging_quality": dim(*imaging),
            "dynamic_degree": dim(*dynamic),
        },
        "psnr": {c["file"]: {"mean": psnr} for c in CLIPS},
        "ssim": {c["file"]: {"mean": ssim} for c in CLIPS},
        "lpips": {c["file"]: {"mean": lpips} for c in CLIPS},
    }


def grade(*references, runs=None, config=CONFIG):
    runs = runs or gate.reference_runs(False)
    return gate.grade(
        {"clips": CLIPS, "gpu_references": list(references)}, runs, config
    )


def statuses(result):
    return {(c["reference"], c["check"]): c["status"] for c in result["checks"]}


def test_a_cached_run_is_compared_for_drift_with_r2_and_for_budget_with_r1():
    assert gate.reference_runs(True) == [(R2, "drift"), (R1, "budget")]
    assert gate.reference_runs(False) == [(R1, "drift")]
    assert gate.reference_runs(None) == [(R1, "drift")]


def test_matching_the_gpu_passes():
    result = grade(comparison(R1))
    assert result["accuracy_check"] == ReportCheckTypes.PASS


@pytest.mark.parametrize(
    "tt, verdict", [(0.6931, ReportCheckTypes.PASS), (0.692, ReportCheckTypes.FAIL)]
)
def test_a_drift_dimension_may_drop_one_percent(tt, verdict):
    # 0.6931 is 0.99% below 0.70; 0.692 is 1.14% below.
    result = grade(comparison(R1, imaging=(tt, 0.70)))
    assert result["accuracy_check"] == verdict


def test_scoring_higher_than_the_gpu_never_fails():
    result = grade(comparison(R1, imaging=(0.80, 0.70)))
    assert result["accuracy_check"] == ReportCheckTypes.PASS


def test_the_budget_comparison_allows_two_percent():
    # 1.5% below R1: within the 2% budget, though it would fail drift's 1%.
    runs = gate.reference_runs(True)
    result = grade(comparison(R2), comparison(R1, imaging=(0.6895, 0.70)), runs=runs)
    assert statuses(result)[(R1, "imaging_quality")] == "PASS"
    assert result["accuracy_check"] == ReportCheckTypes.PASS
    result = grade(comparison(R2, imaging=(0.6895, 0.70)), comparison(R1), runs=runs)
    assert statuses(result)[(R2, "imaging_quality")] == "FAIL"


def test_motion_dimensions_are_reported_not_gated():
    result = grade(comparison(R1, dynamic=(0.0, 1.0)))
    assert statuses(result)[(R1, "dynamic_degree")] == "REPORTED"
    assert result["accuracy_check"] == ReportCheckTypes.PASS


def test_a_clip_over_the_lpips_cap_fails():
    reference = comparison(R1)
    reference["lpips"][CLIPS[1]["file"]]["mean"] = 0.6
    result = grade(reference)
    (check,) = [c for c in result["checks"] if c["check"] == "clip_lpips"]
    assert check["status"] == "FAIL"
    assert CLIPS[1]["video_id"] in check["detail"]


@pytest.mark.parametrize(
    "shift, failing",
    [
        ({"psnr": 20.6}, "mean_psnr"),  # 1.4 dB down: within 1.5
        ({"psnr": 20.4}, "mean_psnr"),  # 1.6 dB down
        ({"ssim": 0.64}, "mean_ssim"),
        ({"lpips": 0.26}, "mean_lpips"),
    ],
)
def test_the_run_means_may_move_only_a_margin_from_the_calibration(shift, failing):
    result = grade(comparison(R1, **shift))
    expected = "PASS" if shift == {"psnr": 20.6} else "FAIL"
    assert statuses(result)[(R1, failing)] == expected


def test_output_is_not_gated_where_there_is_no_calibration():
    config = copy.deepcopy(CONFIG)
    config["output"]["480p"]["calibration"] = {}
    result = grade(comparison(R1), config=config)
    assert statuses(result)[(R1, "mean_shift")] == "NA"
    assert statuses(result)[(R1, "clip_lpips")] == "PASS"
    config["output"]["480p"] = None
    assert statuses(grade(comparison(R1), config=config))[(R1, "output")] == "NA"


def test_a_comparison_that_could_not_be_made_is_na_not_a_pass():
    assert grade()["accuracy_check"] == ReportCheckTypes.NA
    errored = {"reference": f"x/{R1}", "error": "no HF_TOKEN"}
    result = grade(errored)
    assert result["accuracy_check"] == ReportCheckTypes.NA
    assert result["checks"][0]["detail"] == "no HF_TOKEN"


def test_the_config_calibrates_480p_for_both_references():
    config = gate.load_config()
    assert config["tolerance_pct"] == {"drift": 1.0, "budget": 2.0}
    assert set(config["gated_dimensions"]).isdisjoint(config["reported_dimensions"])
    calibration = config["output"]["480p"]["calibration"]
    # R1 from the uncached ci run's 10 clips; R2 from the cached full run's 40.
    assert {run: len(clips) for run, clips in calibration.items()} == {R1: 10, R2: 40}
    assert set(calibration[R1]) < set(calibration[R2])
    for clips in calibration.values():
        assert all(v.endswith("_832x480") for v in clips)
        assert (
            max(c["lpips"] for c in clips.values())
            < config["output"]["480p"]["max_clip_lpips"]
        )
    assert config["output"]["720p"] is None
