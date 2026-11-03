# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The I2V eval scores its saved clips with VBench, shows them against the GPU, and grades on them."""

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
            "file": "00_vbi2v_9999_832x480.mp4",
            "video_id": "vbi2v_9999_832x480",
            "prompt": "a lake",
            "width": 832,
            "height": 480,
        }
    ],
    "dimensions": {
        "i2v_subject": {
            "score": 0.98,
            "per_clip": {"00_vbi2v_9999_832x480.mp4": 0.98},
            "seconds": 30.0,
        }
    },
}
CLIP = "00_vbi2v_9999_832x480.mp4"


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


def test_no_saved_clips_no_section(ctx, tmp_path):
    (tmp_path / "videos" / "evals" / "index.json").unlink()
    assert mod._run_vbench(ctx) == (None, None)


def test_a_t2v_model_is_not_scored(monkeypatch, ctx):
    ctx.model_spec.hf_model_repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    calls = fake_scorer(monkeypatch)
    assert mod._run_vbench(ctx) == (None, None)
    assert calls == []


def render(block):
    return get_renderer("vbench")(block, {})


def test_the_section_is_the_gpu_comparison_in_two_short_tables(monkeypatch, ctx):
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
    # The report carries means; per-clip detail stays in vbench.json.
    (reference,) = block.data["references"]
    assert (reference["run"], reference["role"]) == ("refs/r1", "drift")
    assert reference["output"]["480p"] == {
        "clips": 1,
        "psnr": 22.0,
        "ssim": 0.7,
        "lpips": 0.2,
        "worst_lpips": 0.2,
        "worst_clip": "vbi2v_9999_832x480",
    }
    assert "per_clip" not in json.dumps(block.data)

    rendered = render(block)
    assert rendered.startswith(
        "### VBench: Wan-AI/Wan2.2-I2V-A14B-Diffusers on BLACKHOLE_GALAXY"
    )
    assert "1 clips, scored on cpu. Gate: ✅ PASS (0 of " in rendered
    assert "| Dimension   | TT     | GPU refs/r1 (drift) | Δ % vs refs/r1 (≥ −1) |" in (
        rendered
    )
    assert "| i2v_subject | 0.9800 | 0.9800              | +0.00" in rendered
    assert "| refs/r1       | 480p       | 1     | 22.00     | 0.700 | 0.200 |" in (
        rendered
    )
    # Nothing per clip.
    assert "a lake" not in rendered


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


def eval_block(monkeypatch, ctx, *, tt=0.98, generation=None, result=None):
    fake_scorer(
        monkeypatch,
        result=result or {**RESULT, "gpu_references": [_comparison("refs/r1", tt=tt)]},
    )
    section, gate = mod._run_vbench(ctx)
    eval_result = generation or {
        "num_prompts": 1,
        "num_generated": 1,
        "accuracy_check": mod.ReportCheckTypes.NA,
    }
    return section, mod._i2v_eval_block(ctx, eval_result, gate)


@pytest.mark.parametrize("tt, verdict", [(0.98, "PASS"), (0.95, "FAIL")])
def test_the_gate_grades_the_i2v_eval(monkeypatch, ctx, tt, verdict):
    from report_module.acceptance_criteria import _check_evals
    from report_module.schema import ReportSchema

    section, block = eval_block(monkeypatch, ctx, tt=tt)

    assert (block.kind, block.title) == ("evals", "Video Eval")
    assert block.data["task_name"] == "vbench_i2v"
    assert mod.ReportCheckTypes(block.data["accuracy_check"]).name == verdict
    assert block.data["references"] == "refs/r1 (drift)"
    assert "average_clip" not in block.data
    rendered = render(section)
    assert f"Gate: {'✅' if verdict == 'PASS' else '❌'} {verdict}" in rendered
    if verdict == "FAIL":
        assert "| -3.06 ❌" in rendered
    # Every check is listed, passed or not, with its limit.
    table = rendered.split("#### Gate checks")[1].splitlines()
    row = next(r for r in table if r.startswith("| i2v_subject "))
    assert "refs/r1 (drift)" in row
    assert f"{-3.06 if verdict == 'FAIL' else 0.0:+.2f} %" in row
    assert "≥ -1 %" in row
    assert verdict in row
    background = next(r for r in table if r.startswith("| i2v_background "))
    assert "NA" in background and "not scored" in background
    result = _check_evals(ReportSchema(metadata={}, sections=[block]))
    assert result.status == verdict


def test_videos_the_eval_could_not_make_fail_it_whatever_the_gate_says(
    monkeypatch, ctx
):
    _section, block = eval_block(
        monkeypatch,
        ctx,
        generation={
            "num_prompts": 2,
            "num_generated": 1,
            "rejected_videos": [{"video_id": "v", "problem": "wrong shape"}],
            "accuracy_check": mod.ReportCheckTypes.FAIL,
        },
    )
    assert block.data["accuracy_check"] == mod.ReportCheckTypes.FAIL
    assert block.data["rejected"] == "v: wrong shape"


def test_a_model_the_gate_does_not_name_is_scored_without_references(monkeypatch, ctx):
    # The references are of the base model; a distilled variant runs another schedule.
    ctx.model_spec.hf_model_repo = "Wan-AI/Wan2.2-I2V-Lightning"
    calls = fake_scorer(monkeypatch, result={**RESULT, "gpu_references": []})
    section, gate = mod._run_vbench(ctx)
    assert calls[0][-2:] == ["--gpu-reference", "none"]
    assert gate is None
    assert "Gate" not in render(section)
    block = mod._i2v_eval_block(ctx, {"num_prompts": 1, "num_generated": 1}, gate)
    assert block.data["accuracy_check"] == mod.ReportCheckTypes.NA


def test_a_timeout_keeps_what_finished_and_says_so(monkeypatch, ctx):
    fake_scorer(monkeypatch, raises=subprocess.TimeoutExpired("vbench", 5400))
    block, _gate = mod._run_vbench(ctx)
    assert block.data["error"] == "VBench stopped at its 4 h limit"
    assert block.data["scores"] == {"i2v_subject": 0.98}
    assert "4 h limit; the scores are what finished" in render(block)


def test_a_scorer_that_scored_nothing_is_reported_not_raised(monkeypatch, ctx):
    fake_scorer(monkeypatch, result=None, code=1)
    block, gate = mod._run_vbench(ctx)
    assert block.data["error"] == "VBench exited with code 1"
    assert "scores" not in block.data
    assert "VBench exited with code 1" in render(block)
    # Nothing to grade is not a pass.
    assert gate["accuracy_check"] == mod.ReportCheckTypes.NA
