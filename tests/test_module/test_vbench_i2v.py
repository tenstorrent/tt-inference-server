# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the VBench I2V scorer's glue: inputs it builds and reports it writes."""

from __future__ import annotations

import json
import zipfile

import pytest

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
    (tmp_path / "img").mkdir()
    for name in ("bird.jpg", "lake.jpg"):
        (tmp_path / "img" / name).write_bytes(b"jpg")
    info = mod.build_full_info(
        INDEX, tmp_path / "v", tmp_path / "img", ["i2v_subject", "camera_motion"]
    )
    assert info[0] == {
        "prompt_en": "A bird on a feeder.",
        # camera_motion runs separately, on the clips that ask for a move.
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


def test_conditioning_images_are_found_in_either_pair_set():
    for pair_set in ("coco_i2v", "vbench_i2v"):
        manifest = json.loads(
            (mod.DATASETS_DIR / pair_set / "manifest.json").read_text()
        )
        found = mod.find_conditioning_image(manifest[0]["image"])
        assert found == mod.DATASETS_DIR / pair_set / manifest[0]["image"]


def test_every_manifest_camera_move_is_one_vbench_reads():
    manifest = json.loads(
        (mod.DATASETS_DIR / "vbench_i2v" / "manifest.json").read_text()
    )
    moves = [e["camera_motion"] for e in manifest if "camera_motion" in e]
    assert len(moves) == 4
    for entry in manifest:
        if "camera_motion" in entry:
            assert entry["caption"].endswith(mod.CAMERA_PHRASES[entry["camera_motion"]])


def test_camera_clips_are_linked_under_names_carrying_the_move(tmp_path):
    videos, links = tmp_path / "v", tmp_path / "links"
    videos.mkdir()
    links.mkdir()
    for name in ("00_a.mp4", "01_b.mp4"):
        (videos / name).write_bytes(b"mp4")
    index = [
        {"file": "00_a.mp4", "prompt": "a wave"},
        {
            "file": "01_b.mp4",
            "prompt": "a wave, camera pans left",
            "camera_motion": "pan_left",
        },
    ]
    names = mod.link_camera_clips(index, videos, links)
    assert names == {"01_b camera pans left.mp4": "01_b.mp4"}
    assert (links / "01_b camera pans left.mp4").resolve() == (
        videos / "01_b.mp4"
    ).resolve()


def test_markdown_shows_a_skipped_dimension():
    result = _result()
    result["dimensions"]["camera_motion"] = {
        "skipped": "no clip asks for a camera move",
        "seconds": 0.0,
    }
    md = mod.to_markdown(result, "x")
    assert "| camera_motion | skipped: no clip asks for a camera move | | |" in md
    assert "camera_motion |" not in md.split("| Clip | Prompt |")[1].splitlines()[0]


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


def test_only_checksum_failures_count_as_a_corrupt_download():
    assert mod.is_corrupt_download(
        RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )
    )
    assert not mod.is_corrupt_download(
        AssertionError("Torch not compiled with CUDA enabled")
    )


def _write_reference(folder, index, width=1280, height=720, per_clip=None):
    folder.mkdir(parents=True)
    (folder / "index.json").write_text(json.dumps(index))
    (folder / "provenance.json").write_text(
        json.dumps({"gpu": "NVIDIA H100 NVL", "width": width, "height": height})
    )
    (folder / "vbench.json").write_text(
        json.dumps({"dimensions": {"i2v_subject": {"per_clip": per_clip or {}}}})
    )


def test_gpu_reference_must_hold_every_request_at_the_same_resolution(tmp_path):
    gpu_index = [{**e, "file": f"{i:02d}_gpu.mp4"} for i, e in enumerate(INDEX)]
    _write_reference(tmp_path / "wan" / "a_480p", gpu_index, 832, 480)
    _write_reference(tmp_path / "wan" / "a_720p", gpu_index)
    _write_reference(tmp_path / "wan" / "b_720p", gpu_index[:1])

    found = mod.find_gpu_reference(INDEX, 1280, 720, tmp_path)
    assert found == tmp_path / "wan" / "a_720p"
    # A run of fewer clips still matches; a different size or request does not.
    assert mod.find_gpu_reference(INDEX[:1], 832, 480, tmp_path).name == "a_480p"
    assert mod.find_gpu_reference(INDEX, 640, 360, tmp_path) is None
    other = [{**INDEX[0], "prompt": "A cat."}]
    assert mod.find_gpu_reference(other, 1280, 720, tmp_path) is None


def test_checked_in_gpu_references_cover_both_pair_sets_at_720p():
    for pair_set in ("coco_i2v", "vbench_i2v"):
        manifest = json.loads(
            (mod.DATASETS_DIR / pair_set / "manifest.json").read_text()
        )
        index = [
            {
                "file": f"{i:02d}_x.mp4",
                "prompt": p["caption"],
                "conditioning_image": p["image"],
            }
            for i, p in enumerate(manifest)
        ]
        found = mod.find_gpu_reference(index, 1280, 720)
        assert found is not None and found.name == f"{pair_set}_720p"
        for entry in json.loads((found / "index.json").read_text()):
            assert (found / entry["file"]).is_file()


def test_psnr_is_capped_for_identical_frames_and_follows_the_error():
    np = pytest.importorskip("numpy")
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    assert mod.psnr(frame, frame) == mod.PSNR_IDENTICAL_DB
    # Off by one everywhere: MSE 1, so 20*log10(255).
    assert mod.psnr(frame, frame + 1) == pytest.approx(48.13, abs=0.01)


def test_comparison_pairs_clips_by_request_not_file_name(tmp_path, monkeypatch):
    reference = tmp_path / "ref"
    gpu_index = [{**e, "file": f"{i:02d}_gpu.mp4"} for i, e in enumerate(INDEX)]
    # The reference lists the clips in the other order.
    _write_reference(
        reference,
        gpu_index[::-1],
        per_clip={"00_gpu.mp4": 0.98, "01_gpu.mp4": 0.90},
    )
    monkeypatch.setattr(
        mod,
        "frame_psnr",
        lambda clip, ref: {"frame0": 35.0, "mean": 20.0, "min": 15.0, "ref": ref.name},
    )
    comparison = mod.compare_to_gpu_reference(_result(), INDEX, tmp_path, reference)

    subject = comparison["dimensions"]["i2v_subject"]
    assert subject["per_clip"] == {"00_a.mp4": -0.01, "01_b.mp4": 0.03}
    assert (subject["tt"], subject["gpu"], subject["delta"]) == (0.95, 0.94, 0.01)
    assert (subject["tt_lower"], subject["tt_higher"], subject["clips"]) == (1, 1, 2)
    # A dimension that errored on TT has nothing to compare.
    assert "motion_smoothness" not in comparison["dimensions"]
    assert comparison["psnr"]["01_b.mp4"]["ref"] == "01_gpu.mp4"

    result = {**_result(), "gpu_reference": comparison}
    md = mod.to_markdown(result, "x")
    assert "| i2v_subject | 0.9500 | 0.9400 | +0.0100 | 1 / 1 of 2 |" in md
    assert "| 00 | A bird on a feeder. | 35.0 | 20.0 | 15.0 |" in md
