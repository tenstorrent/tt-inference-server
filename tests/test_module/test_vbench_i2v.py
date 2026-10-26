# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the VBench I2V scorer's glue: inputs it builds and reports it writes."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from test_module._test_common import vbench_i2v as mod
from workflow_module.engine_types import WorkflowVenvType
from workflows.workflow_venvs import REQUIREMENTS_DIR, VENV_CONFIGS

INDEX = [
    {
        "file": "00_a.mp4",
        "prompt": "A bird on a feeder.",
        "conditioning_image": "bird.jpg",
    },
    {
        "file": "01_b.mp4",
        "prompt": "A man at the lake.",
        "conditioning_image": "lake.jpg",
    },
]


def test_full_info_pairs_each_clip_with_its_conditioning_image(tmp_path):
    info = mod.build_full_info(INDEX, tmp_path / "v", tmp_path / "img", ["i2v_subject"])
    assert info[0] == {
        "prompt_en": "A bird on a feeder.",
        "dimension": ["i2v_subject"],
        "video_list": [str(tmp_path / "v" / "00_a.mp4")],
        "custom_image_path": str(tmp_path / "img" / "bird.jpg"),
    }
    assert info[1]["custom_image_path"].endswith("lake.jpg")


def test_full_info_omits_the_image_for_a_t2v_clip(tmp_path):
    entry = {"file": "00.mp4", "prompt": "a sunset"}
    (info,) = mod.build_full_info([entry], tmp_path, tmp_path, ["aesthetic_quality"])
    assert "custom_image_path" not in info


def test_per_clip_scores_are_keyed_by_file_and_on_the_aggregate_scale():
    results = [
        {"video_path": "/x/00_a.mp4", "video_results": 61.234},
        {"video_path": "/x/01_b.mp4", "video_results": 70.0},
    ]
    assert mod.per_clip_scores("imaging_quality", results) == {
        "00_a.mp4": 0.6123,
        "01_b.mp4": 0.7,
    }
    assert mod.per_clip_scores(
        "dynamic_degree", [{"video_path": "a.mp4", "video_results": True}]
    ) == {"a.mp4": 1.0}


def _result():
    return {
        "device": "cpu",
        "num_clips": 2,
        "clips": [{"file": e["file"], "prompt": e["prompt"]} for e in INDEX],
        "dimensions": {
            "i2v_subject": {
                "score": 0.95,
                "per_clip": {"00_a.mp4": 0.97, "01_b.mp4": 0.93},
                "seconds": 12.0,
            },
            "motion_smoothness": {
                "error": "RuntimeError: out of memory",
                "seconds": 3.0,
            },
        },
    }


def test_markdown_reports_scores_minimums_and_errors():
    md = mod.to_markdown(_result(), "Wan2.2_evals")
    assert "### VBench: Wan2.2_evals" in md
    assert "| i2v_subject | 0.9500 | 0.9300 | 12.0 |" in md
    assert "| motion_smoothness | error: RuntimeError: out of memory |" in md
    # The per-clip table covers only the dimensions that ran.
    assert "| Clip | Prompt | i2v_subject |" in md
    assert "| 01 | A man at the lake. | 0.930 |" in md


def test_summarize_keeps_dimensions_that_ran():
    assert mod.summarize(_result()) == {"i2v_subject": 0.95}


def test_missing_index_is_a_no_op(tmp_path, capsys):
    assert mod.main(["--videos-dir", str(tmp_path)]) == 0
    assert "VBench skipped" in capsys.readouterr().out


def test_vbench_venv_is_registered_with_its_requirements():
    venv = VENV_CONFIGS[WorkflowVenvType.VBENCH_I2V]
    assert (REQUIREMENTS_DIR / venv.requirements_file).is_file()
    assert venv.setup_function is not None


def test_default_dimensions_leave_out_camera_motion():
    # Our eval prompts ask for no camera move, so camera_motion would score noise.
    assert "camera_motion" not in mod.DEFAULT_DIMENSIONS
    assert set(mod.DEFAULT_DIMENSIONS) >= {"i2v_subject", "i2v_background"}


def test_default_images_dir_is_the_eval_fixture_set():
    assert (mod.DEFAULT_IMAGES_DIR / "manifest.json").is_file()
    manifest = json.loads((mod.DEFAULT_IMAGES_DIR / "manifest.json").read_text())
    assert manifest
    assert isinstance(mod.DEFAULT_IMAGES_DIR, Path)


def test_raft_weights_are_unpacked_without_an_unzip_binary(tmp_path, monkeypatch):
    def _fake_download(url, dest):
        assert url == mod.RAFT_URL
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("models/raft-things.pth", b"weights")

    monkeypatch.setattr(mod.urllib.request, "urlretrieve", _fake_download)
    mod.ensure_raft_weights(tmp_path)
    target = tmp_path / "raft_model" / "models" / "raft-things.pth"
    assert target.read_bytes() == b"weights"
    assert not (tmp_path / "raft_model" / "models.zip").exists()

    # Already there: no second download.
    monkeypatch.setattr(mod.urllib.request, "urlretrieve", None)
    mod.ensure_raft_weights(tmp_path)
