# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for scripts/wan_i2v_gpu_reference.py that need no GPU or model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "wan_i2v_gpu_reference.py"

spec = importlib.util.spec_from_file_location("wan_i2v_gpu_reference", SCRIPT)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def make_data(root: Path) -> Path:
    """A two-video vbench_i2v_50 data folder: one dev 16:9, one ci-only 9:16."""
    entries = []
    for pair_id, (w, h), dev in (
        ("vbi2v_0001", (832, 480), True),
        ("vbi2v_0002", (480, 832), False),
    ):
        image = f"images/{w}x{h}/{pair_id}.jpg"
        (root / image).parent.mkdir(parents=True, exist_ok=True)
        (root / image).write_bytes(pair_id.encode())
        entries.append(
            {
                "video_id": f"{pair_id}_{w}x{h}",
                "pair_id": pair_id,
                "image": image,
                "caption": f"caption {pair_id}",
                "width": w,
                "height": h,
                "seed": int(pair_id[-4:]),
                "camera_motion": "pan_left" if dev else None,
                "dev": dev,
                "ci": True,
                "image_sha256": hashlib.sha256(pair_id.encode()).hexdigest(),
            }
        )
    (root / "pairs").mkdir()
    (root / "pairs" / "manifest.json").write_text(json.dumps(entries))
    return root


def args_for(data: Path, out: Path, **kw) -> argparse.Namespace:
    defaults = {
        "pairs": "vbench_i2v_50",
        "data": data,
        "out": out,
        "subset": "all",
        "shape": None,
        "limit": None,
        "shard": (0, 1),
    }
    return argparse.Namespace(**{**defaults, **kw})


def test_jobs_follow_the_manifest(tmp_path):
    data = make_data(tmp_path / "data")
    jobs = ref.jobs(args_for(data, tmp_path / "out"))
    assert [j["file"] for j in jobs] == [
        "vbi2v_0001_832x480.mp4",
        "vbi2v_0002_480x832.mp4",
    ]
    first = jobs[0]
    assert first["out_dir"] == tmp_path / "out" / "832x480"
    assert (
        first["noise_path"]
        == tmp_path / "out" / "noise" / "832x480" / "vbi2v_0001_832x480.pt"
    )
    assert (first["width"], first["height"], first["seed"]) == (832, 480, 1)
    assert first["conditioning_image"] == "vbi2v_0001.jpg"
    assert first["camera_motion"] == "pan_left"


def test_jobs_pick_subset_and_shape(tmp_path):
    data = make_data(tmp_path / "data")
    out = tmp_path / "out"
    assert [j["video_id"] for j in ref.jobs(args_for(data, out, subset="dev"))] == [
        "vbi2v_0001_832x480"
    ]
    assert [
        j["video_id"] for j in ref.jobs(args_for(data, out, shape=["480x832"]))
    ] == ["vbi2v_0002_480x832"]


def test_jobs_refuse_an_image_that_does_not_match(tmp_path):
    data = make_data(tmp_path / "data")
    (data / "images" / "832x480" / "vbi2v_0001.jpg").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="sha256"):
        ref.jobs(args_for(data, tmp_path / "out"))


def test_jobs_need_data_for_the_validation_set(tmp_path):
    with pytest.raises(SystemExit, match="--data"):
        ref.jobs(args_for(None, tmp_path / "out"))


def test_legacy_set_runs_at_one_resolution_and_seed(tmp_path):
    jobs = ref.jobs(
        argparse.Namespace(
            pairs="coco_i2v",
            out=tmp_path,
            resolution="480p",
            seed=7,
            limit=2,
            shard=(0, 1),
        )
    )
    assert [(j["file"], j["width"], j["height"], j["seed"]) for j in jobs] == [
        ("00_gpu.mp4", 832, 480, 7),
        ("01_gpu.mp4", 832, 480, 7),
    ]


def test_wan_schedule_is_wan22_i2v_reference_config():
    # wan/configs/wan_i2v_A14B.py: sample_shift 5.0, boundary 0.900, guide (3.5, 3.5).
    assert ref.SCHEDULES["wan"] == {
        "flow_shift": 5.0,
        "boundary_ratio": 0.9,
        "guidance_scale": 3.5,
        "guidance_scale_2": 3.5,
    }
    assert ref.WAN_NEGATIVE_PROMPT.startswith(
        "色调艳丽"
    ) and ref.WAN_NEGATIVE_PROMPT.endswith("倒着走")


def test_resume_refuses_other_settings(tmp_path):
    path = tmp_path / "provenance.json"
    record = {"model": "m", "steps": 40, "guidance_scale": 3.5}
    ref.check_resumable(path, record)  # nothing made yet
    path.write_text(json.dumps(record))
    ref.check_resumable(path, dict(record, gpu="another"))  # not a generation setting
    with pytest.raises(SystemExit, match="guidance_scale"):
        ref.check_resumable(path, dict(record, guidance_scale=4.0))


def test_initial_noise_is_the_seeded_cpu_draw():
    torch = pytest.importorskip("torch")

    class Pipe:
        class vae:
            class config:
                z_dim = 16

        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8

    noise = ref.initial_noise(Pipe, 5, 832, 480)
    assert noise.shape == (1, 16, 21, 60, 104)
    assert noise.dtype == torch.float32
    expected = torch.randn(noise.shape, generator=torch.Generator("cpu").manual_seed(5))
    assert torch.equal(noise, expected)


def test_shards_split_the_clips_between_them(tmp_path):
    data = make_data(tmp_path / "data")
    out = tmp_path / "out"
    first = ref.jobs(args_for(data, out, shard=(0, 2)))
    second = ref.jobs(args_for(data, out, shard=(1, 2)))
    assert [j["video_id"] for j in first] == ["vbi2v_0001_832x480"]
    assert [j["video_id"] for j in second] == ["vbi2v_0002_480x832"]
    assert ref.parse_shard("1/4") == (1, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        ref.parse_shard("4/4")


def test_a_pair_set_can_be_a_folder_with_a_manifest(tmp_path):
    (tmp_path / "set").mkdir()
    pairs = [{"image": f"{i}.jpg", "caption": f"c{i}"} for i in range(101)]
    (tmp_path / "set" / "manifest.json").write_text(json.dumps(pairs))
    jobs = ref.jobs(
        argparse.Namespace(
            pairs=str(tmp_path / "set"),
            out=tmp_path / "out",
            resolution="720p",
            seed=42,
            limit=None,
            shard=(0, 1),
        )
    )
    # Clip numbers widen past two digits for sets over 100 clips.
    assert jobs[0]["file"] == "000_gpu.mp4" and jobs[-1]["file"] == "100_gpu.mp4"
    assert jobs[5]["image_path"] == tmp_path / "set" / "5.jpg"


def test_records_rebuild_the_index_and_adopt_an_older_index(tmp_path):
    # A folder from before per-clip records: an index.json and its clips.
    (tmp_path / "a.mp4").write_bytes(b"a")
    old = [{"file": "a.mp4"}, {"file": "gone.mp4"}]
    (tmp_path / "index.json").write_text(json.dumps(old))
    assert set(ref.clip_records(tmp_path)) == {"a.mp4"}

    ref.write_record(tmp_path, {"file": "b.mp4"})
    ref.write_index(tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["file"] for e in index] == ["a.mp4", "b.mp4"]
