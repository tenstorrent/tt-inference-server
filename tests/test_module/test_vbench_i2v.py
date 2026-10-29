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

# Two clips of an eval run at different shapes, as the I2V eval saves them.
INDEX = [
    {
        "file": "00_a.mp4",
        "prompt": "A bird on a feeder.",
        "conditioning_image": "bird.jpg",
        "video_id": "vbi2v_0001_832x480",
        "width": 832,
        "height": 480,
        "seed": 1,
    },
    {
        "file": "01_b.mp4",
        "prompt": "A man at the lake.",
        "conditioning_image": "lake.jpg",
        "video_id": "vbi2v_0002_1280x720",
        "width": 1280,
        "height": 720,
        "seed": 2,
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
        "clips": [
            {k: e[k] for k in ("file", "prompt", "video_id", "width", "height")}
            for e in INDEX
        ],
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


def test_the_report_summary_has_the_scores_and_what_did_not_score():
    summary = mod.report_summary(_result())
    assert summary["scores"] == {"i2v_subject": 0.95}
    assert summary["problems"] == {"motion_smoothness": "RuntimeError: out of memory"}
    assert summary["references"] == []


def test_summarize_keeps_dimensions_that_ran():
    assert mod.summarize(_result()) == {"i2v_subject": 0.95}


def test_missing_index_is_a_no_op(tmp_path, capsys):
    assert mod.main(["--videos-dir", str(tmp_path)]) == 0
    assert "VBench skipped" in capsys.readouterr().out


def test_vbench_venv_is_registered_with_its_requirements():
    venv = VENV_CONFIGS[WorkflowVenvType.VBENCH_I2V]
    assert (REQUIREMENTS_DIR / venv.requirements_file).is_file()
    assert venv.setup_function is not None


def test_a_conditioning_image_needs_a_folder_or_a_shape():
    with pytest.raises(ValueError, match="no images dir and no shape"):
        mod.find_conditioning_image("bird.jpg")


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


def test_the_report_summary_says_why_a_dimension_was_skipped():
    result = _result()
    result["dimensions"]["camera_motion"] = {
        "skipped": "no clip asks for a camera move",
        "seconds": 0.0,
    }
    problems = mod.report_summary(result)["problems"]
    assert problems["camera_motion"] == "no clip asks for a camera move"


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


def _write_reference(folder, index, per_clip=None):
    folder.mkdir(parents=True)
    (folder / "index.json").write_text(json.dumps(index))
    (folder / "provenance.json").write_text(json.dumps({"gpu": "NVIDIA H100 NVL"}))
    (folder / "vbench.json").write_text(
        json.dumps({"dimensions": {"i2v_subject": {"per_clip": per_clip or {}}}})
    )


def _reference_run(run, per_clip=None):
    """A GPU reference run as the GPU script writes it: one folder per shape."""
    for entry in INDEX:
        shape = f"{entry['width']}x{entry['height']}"
        gpu = {**entry, "file": f"{entry['video_id']}.mp4"}
        _write_reference(run / shape, [gpu], per_clip)


def test_gpu_references_are_matched_per_video_across_shapes(tmp_path):
    _reference_run(tmp_path / "r1")
    matches = mod.find_gpu_references(INDEX, tmp_path / "r1")
    assert matches == {
        "00_a.mp4": (tmp_path / "r1" / "832x480", "vbi2v_0001_832x480.mp4"),
        "01_b.mp4": (tmp_path / "r1" / "1280x720", "vbi2v_0002_1280x720.mp4"),
    }
    # A clip the run has no reference for is left out, not mismatched.
    other = [{**INDEX[0], "video_id": "vbi2v_0003_832x480"}]
    assert mod.find_gpu_references(other, tmp_path / "r1") == {}


def test_hf_references_are_matched_from_metadata_then_only_they_fetched(
    tmp_path, monkeypatch
):
    _reference_run(tmp_path / "refs" / "r1")
    fetched = []

    def fake_snapshot(patterns):
        fetched.append(list(patterns))
        return tmp_path

    monkeypatch.setattr(mod, "hf_snapshot", fake_snapshot)
    matches = mod.find_gpu_references(INDEX[:1])
    assert matches == {
        "00_a.mp4": (
            tmp_path / "refs" / "r1" / "832x480",
            "vbi2v_0001_832x480.mp4",
        )
    }
    assert fetched == [
        mod.reference_metadata("refs/r1"),
        ["refs/r1/832x480/vbi2v_0001_832x480.mp4"],
    ]


def test_the_cached_reference_run_is_fetched_from_its_own_folder(tmp_path, monkeypatch):
    _reference_run(tmp_path / "refs" / "r2" / "tt")
    fetched = []

    def fake_snapshot(patterns):
        fetched.append(list(patterns))
        return tmp_path

    monkeypatch.setattr(mod, "hf_snapshot", fake_snapshot)
    matches = mod.find_gpu_references(INDEX[:1], hf_run=mod.GPU_REFERENCE_RUN_CACHED)
    assert matches == {
        "00_a.mp4": (
            tmp_path / "refs" / "r2" / "tt" / "832x480",
            "vbi2v_0001_832x480.mp4",
        )
    }
    assert fetched[0] == [
        "refs/r2/tt/*/index.json",
        "refs/r2/tt/*/provenance.json",
        "refs/r2/tt/*/vbench.json",
    ]
    assert fetched[1] == ["refs/r2/tt/832x480/vbi2v_0001_832x480.mp4"]


def test_hf_reference_is_named_by_its_path_in_the_dataset(tmp_path):
    cached = (
        tmp_path / "datasets--Tenstorrent--x" / "snapshots" / "51206b33" / "refs" / "r1"
    )
    assert mod.describe_reference(cached) == f"{mod.I2V_EVAL_HF_REPO}/refs/r1"


def test_the_eval_data_revision_is_pinned():
    assert len(mod.I2V_EVAL_HF_REVISION) == 40
    int(mod.I2V_EVAL_HF_REVISION, 16)


def test_psnr_is_capped_for_identical_frames_and_follows_the_error():
    np = pytest.importorskip("numpy")
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    assert mod.psnr(frame, frame) == mod.PSNR_IDENTICAL_DB
    # Off by one everywhere: MSE 1, so 20*log10(255).
    assert mod.psnr(frame, frame + 1) == pytest.approx(48.13, abs=0.01)


def _frame_metrics(clip, ref, lpips):
    frames = {"frame0": 35.0, "mean": 20.0, "min": 15.0, "ref": ref.name}
    return {
        "psnr": frames,
        "ssim": {"frame0": 0.9, "mean": 0.8, "min": 0.7},
        "lpips": {"frame0": 0.05, "mean": 0.1, "max": 0.2},
    }


def test_comparison_pairs_clips_by_video_across_shape_folders(tmp_path, monkeypatch):
    run = tmp_path / "r1"
    _reference_run(
        run,
        per_clip={"vbi2v_0001_832x480.mp4": 0.98, "vbi2v_0002_1280x720.mp4": 0.90},
    )
    monkeypatch.setattr(mod, "frame_metrics", _frame_metrics)
    monkeypatch.setattr(mod, "lpips_scorer", lambda device: "lpips")
    matches = mod.find_gpu_references(INDEX, run)
    comparison = mod.compare_to_gpu_reference(_result(), tmp_path, matches, run)

    subject = comparison["dimensions"]["i2v_subject"]
    assert subject["per_clip"] == {"00_a.mp4": -0.01, "01_b.mp4": 0.03}
    assert (subject["tt"], subject["gpu"], subject["delta"]) == (0.95, 0.94, 0.01)
    assert (subject["tt_lower"], subject["tt_higher"], subject["clips"]) == (1, 1, 2)
    # A dimension that errored on TT has nothing to compare.
    assert "motion_smoothness" not in comparison["dimensions"]
    assert comparison["psnr"]["01_b.mp4"]["ref"] == "vbi2v_0002_1280x720.mp4"
    assert comparison["lpips"]["00_a.mp4"]["max"] == 0.2
    assert (comparison["gpu"], comparison["clips"]) == ("NVIDIA H100 NVL", 2)
    # Without the eval's index there is no first frame or seed to check.
    assert (comparison["first_frame"], comparison["noise"]) == ({}, {})

    # The report's summary: the means, each resolution on its own.
    (summary,) = mod.report_summary({**_result(), "gpu_references": [comparison]})[
        "references"
    ]
    assert summary["dimensions"] == {
        "i2v_subject": {"tt": 0.95, "gpu": 0.94, "clips": 2}
    }
    assert summary["output"] == {
        res: {
            "clips": 1,
            "psnr": 20.0,
            "ssim": 0.8,
            "lpips": 0.1,
            "worst_lpips": 0.1,
            "worst_clip": video_id,
        }
        for res, video_id in (
            ("480p", "vbi2v_0001_832x480"),
            ("720p", "vbi2v_0002_1280x720"),
        )
    }
    assert summary["failed_clips"] == []


def test_comparison_checks_first_frames_and_the_saved_noise(tmp_path, monkeypatch):
    run = tmp_path / "r1"
    _reference_run(run)
    noise = tmp_path / "noise"
    (noise / "832x480").mkdir(parents=True)
    (noise / "832x480" / "vbi2v_0001_832x480.pt").write_bytes(b"")
    monkeypatch.setattr(mod, "frame_metrics", _frame_metrics)
    monkeypatch.setattr(mod, "lpips_scorer", lambda device: "lpips")
    monkeypatch.setattr(
        mod,
        "first_frame_fidelity",
        lambda clip, image, lpips: {
            "psnr": 30.0 if clip.parent == tmp_path else 32.0,
            "ssim": 0.9,
            "lpips": 0.05,
        },
    )
    checked = []

    def fake_redraw(seed, path):
        checked.append((seed, path.name))
        return {"identical": True, "max_abs_diff": 0.0, "torch": "2.5.1"}

    monkeypatch.setattr(mod, "redraws_saved_noise", fake_redraw)
    (tmp_path / "img").mkdir()
    for name in ("bird.jpg", "lake.jpg"):
        (tmp_path / "img" / name).write_bytes(b"jpg")

    matches = mod.find_gpu_references(INDEX, run)
    comparison = mod.compare_to_gpu_reference(
        _result(),
        tmp_path,
        matches,
        run,
        index=INDEX,
        images_dir=tmp_path / "img",
        noise_dir=noise,
    )
    assert comparison["first_frame"]["00_a.mp4"] == {
        "tt": {"psnr": 30.0, "ssim": 0.9, "lpips": 0.05},
        "gpu": {"psnr": 32.0, "ssim": 0.9, "lpips": 0.05},
    }
    # Each clip's own seed, against its own video's saved noise.
    assert checked == [(1, "vbi2v_0001_832x480.pt")]
    assert comparison["noise"]["01_b.mp4"] == {
        "skipped": "no saved noise for this video"
    }

    (summary,) = mod.report_summary({**_result(), "gpu_references": [comparison]})[
        "references"
    ]
    assert (summary["noise_differs"], summary["noise_checked"]) == (0, 1)


def test_lpips_failing_to_load_keeps_psnr_and_ssim(tmp_path, monkeypatch):
    run = tmp_path / "r1"
    _reference_run(run)
    lpips_given = []

    def no_lpips(device):
        raise OSError("weights download failed")

    def metrics(clip, ref, lpips):
        lpips_given.append(lpips)
        return {"psnr": {"frame0": 30.0, "mean": 30.0, "min": 30.0}}

    monkeypatch.setattr(mod, "lpips_scorer", no_lpips)
    monkeypatch.setattr(mod, "frame_metrics", metrics)
    matches = mod.find_gpu_references(INDEX, run)
    comparison = mod.compare_to_gpu_reference(_result(), tmp_path, matches, run)
    assert lpips_given == [None, None]
    assert "weights download failed" in comparison["lpips_error"]
    (summary,) = mod.report_summary({**_result(), "gpu_references": [comparison]})[
        "references"
    ]
    assert "weights download failed" in summary["lpips_error"]
    assert summary["output"]["480p"]["lpips"] is None


def test_the_saved_noise_is_fetched_only_for_matched_clips(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(
        mod, "hf_snapshot", lambda patterns: fetched.extend(patterns) or tmp_path
    )
    matches = {"01_b.mp4": (tmp_path, "vbi2v_0002_1280x720.mp4")}
    assert mod.saved_noise_dir(INDEX, matches) == tmp_path / "noise"
    assert fetched == ["noise/1280x720/vbi2v_0002_1280x720.pt"]


def test_a_redraw_from_the_seed_is_compared_bit_for_bit(tmp_path):
    torch = pytest.importorskip("torch")
    drawn = torch.randn(
        (1, 16, 2, 4, 6), generator=torch.Generator("cpu").manual_seed(7)
    )
    torch.save(drawn, tmp_path / "same.pt")
    torch.save(drawn + 1e-6, tmp_path / "off.pt")
    same = mod.redraws_saved_noise(7, tmp_path / "same.pt")
    assert (same["identical"], same["max_abs_diff"]) == (True, 0.0)
    off = mod.redraws_saved_noise(7, tmp_path / "off.pt")
    assert not off["identical"]
    assert off["max_abs_diff"] == pytest.approx(1e-6, rel=0.1)


def test_frame_metrics_score_every_frame_and_keep_the_worst(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("skimage")
    imageio = pytest.importorskip("imageio")
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    imageio.mimwrite(tmp_path / "a.mp4", frames, fps=16, quality=10)
    noisy = [np.clip(f.astype(int) + 20, 0, 255).astype(np.uint8) for f in frames]
    imageio.mimwrite(tmp_path / "b.mp4", noisy, fps=16, quality=10)

    same = mod.frame_metrics(tmp_path / "a.mp4", tmp_path / "a.mp4")
    assert same["psnr"]["min"] == mod.PSNR_IDENTICAL_DB
    assert same["ssim"]["min"] == pytest.approx(1.0)
    assert "lpips" not in same  # no scorer given
    other = mod.frame_metrics(
        tmp_path / "a.mp4", tmp_path / "b.mp4", lpips=lambda a, b: 0.25
    )
    assert len(other["psnr"]["frames"]) == len(other["ssim"]["frames"]) == 3
    assert other["psnr"]["min"] < 40 and other["ssim"]["mean"] < 1
    assert other["lpips"] == {
        "frame0": 0.25,
        "mean": 0.25,
        "max": 0.25,
        "frames": [0.25] * 3,
    }


def test_a_clips_first_frame_comes_from_hf_at_its_shape(tmp_path, monkeypatch):
    image = tmp_path / "images" / "480x832" / "vbi2v_0048.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpg")
    fetched = []

    def fake_snapshot(patterns):
        fetched.extend(patterns)
        return tmp_path

    monkeypatch.setattr(mod, "hf_snapshot", fake_snapshot)
    index = [
        {
            "file": "00_vbi2v_0048_480x832.mp4",
            "prompt": "neon",
            "conditioning_image": "vbi2v_0048.jpg",
            "width": 480,
            "height": 832,
        }
    ]
    [record] = mod.build_full_info(index, tmp_path, None, ["i2v_subject"])
    assert record["custom_image_path"] == str(image)
    assert fetched == ["images/480x832/vbi2v_0048.jpg"]
