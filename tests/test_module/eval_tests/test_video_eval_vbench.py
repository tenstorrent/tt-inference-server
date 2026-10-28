# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The video eval scores its saved clips with VBench and reports them in their own section."""

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
        {"file": "00_vbi2v_0913_832x480.mp4", "video_id": "vbi2v_0913_832x480", "prompt": "a lake"}
    ],
    "dimensions": {
        "i2v_subject": {
            "score": 0.98,
            "per_clip": {"00_vbi2v_0913_832x480.mp4": 0.98},
            "seconds": 30.0,
        }
    },
    "gpu_reference": {
        "reference": "refs/r1",
        "gpu": "H100",
        "clips": 1,
        "dimensions": {},
        "psnr": {
            "00_vbi2v_0913_832x480.mp4": {
                "frame0": 30.0,
                "mean": 22.0,
                "min": 18.0,
                "frames": [30.0, 22.0, 18.0],
            }
        },
    },
}


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
    assert mod._run_vbench(ctx) is None


def test_the_scores_become_a_report_section(monkeypatch, ctx):
    calls = fake_scorer(monkeypatch)
    block = mod._run_vbench(ctx)

    assert calls[0][1:4] == ["-m", "test_module._test_common.vbench_i2v", "--videos-dir"]
    assert block.kind == "vbench"
    assert block.data["scores"] == {"i2v_subject": 0.98}
    assert "error" not in block.data
    # Per-frame lists stay in vbench.json, out of the report.
    assert block.data["gpu_reference"]["psnr"]["00_vbi2v_0913_832x480.mp4"] == {
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


def test_a_timeout_keeps_what_finished_and_says_so(monkeypatch, ctx):
    fake_scorer(monkeypatch, raises=subprocess.TimeoutExpired("vbench", 5400))
    block = mod._run_vbench(ctx)
    assert block.data["error"] == "VBench stopped at its 90 min limit"
    assert block.data["scores"] == {"i2v_subject": 0.98}
    assert "the scores above are what finished" in block.data["markdown"]


def test_a_scorer_that_scored_nothing_is_reported_not_raised(monkeypatch, ctx):
    fake_scorer(monkeypatch, result=None, code=1)
    block = mod._run_vbench(ctx)
    assert block.data["error"] == "VBench exited with code 1"
    assert "scores" not in block.data
    assert "VBench exited with code 1" in block.data["markdown"]
