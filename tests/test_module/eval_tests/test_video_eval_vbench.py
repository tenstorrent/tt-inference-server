# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The video eval scores its saved clips with VBench, reports them in their own section, and gates on them."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from report_module.renderers import get_renderer
from test_module.eval_tests import video_eval_tests as mod

RESULT = {
    "device": "cpu",
    "num_clips": 1,
    "clips": [
        {
            "file": "00_vbi2v_0913_832x480.mp4",
            "video_id": "vbi2v_0913_832x480",
            "prompt": "a lake",
        }
    ],
    "dimensions": {
        "i2v_subject": {
            "score": 0.98,
            "per_clip": {"00_vbi2v_0913_832x480.mp4": 0.98},
            "seconds": 30.0,
        }
    },
}
CLIP = "00_vbi2v_0913_832x480.mp4"


def _comparison(run: str, tt: float = 0.98, gpu: float = 0.98) -> dict:
    return {
        "reference": f"Tenstorrent/wan2.2-i2v-vbench-refs/{run}",
        "gpu": "H100",
        "clips": 1,
        "dimensions": {
            "i2v_subject": {
                "tt": tt,
                "gpu": gpu,
                "delta": round(tt - gpu, 4),
                "clips": 1,
                "tt_lower": int(tt < gpu),
                "tt_higher": int(tt > gpu),
                "per_clip": {CLIP: round(tt - gpu, 4)},
            }
        },
        "psnr": {
            CLIP: {
                "frame0": 30.0,
                "mean": 22.0,
                "min": 18.0,
                "frames": [30.0, 22.0, 18.0],
            }
        },
        "ssim": {CLIP: {"mean": 0.7, "min": 0.5}},
        "lpips": {CLIP: {"mean": 0.2, "max": 0.3}},
    }


RESULT["gpu_references"] = [_comparison("refs/r1")]
RESULT["gpu_reference"] = RESULT["gpu_references"][0]


@pytest.fixture
def ctx(tmp_path):
    videos = tmp_path / "videos" / "evals"
    videos.mkdir(parents=True)
    (videos / "index.json").write_text("[]")
    return SimpleNamespace(
        output_path=str(tmp_path),
        model_spec=SimpleNamespace(
            model_name="Wan2.2-I2V-A14B-Diffusers",
            hf_model_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        ),
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )


def fake_scorer(monkeypatch, result=RESULT, code=0, raises=None):
    calls = []

    def run(command, cwd, timeout):
        calls.append(command)
        videos_dir = command[command.index("--videos-dir") + 1]
        if result is not None:
            with open(f"{videos_dir}/vbench.json", "w") as f:
                json.dump(result, f)
        if raises:
            raise raises
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(mod.subprocess, "run", run)
    return calls


def test_no_saved_clips_no_section(tmp_path):
    ctx = SimpleNamespace(output_path=str(tmp_path))
    assert mod._run_vbench(ctx) == []


def test_the_scores_become_a_report_section(monkeypatch, ctx):
    calls = fake_scorer(monkeypatch)
    block, _gate = mod._run_vbench(ctx)

    assert calls[0][1:4] == [
        "-m",
        "test_module._test_common.vbench_i2v",
        "--videos-dir",
    ]
    assert block.kind == "vbench"
    assert block.data["scores"] == {"i2v_subject": 0.98}
    assert "error" not in block.data
    # Per-frame lists stay in vbench.json, out of the report.
    assert block.data["gpu_references"][0]["psnr"][CLIP] == {
        "frame0": 30.0,
        "mean": 22.0,
        "min": 18.0,
    }
    rendered = get_renderer("vbench")(block, {})
    assert rendered.startswith(
        "### VBench: Wan-AI/Wan2.2-I2V-A14B-Diffusers on BLACKHOLE_GALAXY"
    )
    # Clips are labelled by their video id.
    assert "| vbi2v_0913_832x480 | a lake | 0.980 |" in rendered
    assert "| vbi2v_0913_832x480 | a lake | 30.0 / 22.0 / 18.0 |" in rendered


@pytest.mark.parametrize(
    "env_vars, reference_runs",
    [
        ({}, ["refs/r1"]),
        ({"WAN22_DBCACHE": "off"}, ["refs/r1"]),
        # The cached GPU run for drift, then the uncached one for the budget.
        ({"WAN22_DBCACHE": "on"}, ["refs/r2/tt", "refs/r1"]),
    ],
)
def test_a_caching_deployment_is_compared_with_both_gpu_runs(
    monkeypatch, ctx, env_vars, reference_runs
):
    ctx.model_spec.env_vars = env_vars
    calls = fake_scorer(monkeypatch)
    mod._run_vbench(ctx)
    command = calls[0]
    runs = [command[i + 1] for i, a in enumerate(command) if a == "--gpu-reference-run"]
    assert runs == reference_runs


@pytest.mark.parametrize("tt, verdict", [(0.98, "PASS"), (0.95, "FAIL")])
def test_the_gate_is_an_eval_block_acceptance_grades(monkeypatch, ctx, tt, verdict):
    from report_module.acceptance_criteria import _check_evals
    from report_module.schema import ReportSchema

    fake_scorer(
        monkeypatch,
        result={**RESULT, "gpu_references": [_comparison("refs/r1", tt=tt)]},
    )
    section, gate = mod._run_vbench(ctx)

    assert (gate.kind, gate.title) == ("evals", "VBench gate")
    assert mod.ReportCheckTypes(gate.data["accuracy_check"]).name == verdict
    assert gate.data["references"] == "refs/r1 (drift)"
    assert f"#### VBench gate: {verdict}" in section.data["markdown"]
    result = _check_evals(ReportSchema(metadata={}, sections=[gate]))
    assert result.status == verdict


def test_a_model_the_gate_does_not_name_gets_no_gate(monkeypatch, ctx):
    # The references are of the base model; a distilled variant runs another schedule.
    ctx.model_spec.hf_model_repo = "Wan-AI/Wan2.2-I2V-Lightning"
    fake_scorer(monkeypatch)
    (section,) = mod._run_vbench(ctx)
    assert section.kind == "vbench"
    assert "VBench gate" not in section.data["markdown"]


def test_a_timeout_keeps_what_finished_and_says_so(monkeypatch, ctx):
    fake_scorer(monkeypatch, raises=subprocess.TimeoutExpired("vbench", 5400))
    block, _gate = mod._run_vbench(ctx)
    assert block.data["error"] == "VBench stopped at its 4 h limit"
    assert block.data["scores"] == {"i2v_subject": 0.98}
    assert "the scores above are what finished" in block.data["markdown"]


def test_a_scorer_that_scored_nothing_is_reported_not_raised(monkeypatch, ctx):
    fake_scorer(monkeypatch, result=None, code=1)
    block, gate = mod._run_vbench(ctx)
    assert block.data["error"] == "VBench exited with code 1"
    assert "scores" not in block.data
    assert "VBench exited with code 1" in block.data["markdown"]
    # Nothing to grade is not a pass.
    assert mod.ReportCheckTypes(gate.data["accuracy_check"]).name == "NA"
