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


class NoisePipe:
    """What the noise helpers read from a Wan pipeline."""

    class vae:
        class config:
            z_dim = 16

    vae_scale_factor_temporal = 4
    vae_scale_factor_spatial = 8


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
        "data": data,
        "out": out,
        "subset": "all",
        "shape": None,
        "limit": None,
        "shard": (0, 1),
        "noise_dir": None,
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
    assert first["saved_noise"] is None


def test_jobs_load_noise_saved_by_an_earlier_run(tmp_path):
    data = make_data(tmp_path / "data")
    noise_dir = tmp_path / "r1" / "noise"
    with pytest.raises(SystemExit, match="no saved noise"):
        ref.jobs(args_for(data, tmp_path / "out", noise_dir=noise_dir))
    for shape, video_id in (
        ("832x480", "vbi2v_0001_832x480"),
        ("480x832", "vbi2v_0002_480x832"),
    ):
        (noise_dir / shape).mkdir(parents=True, exist_ok=True)
        (noise_dir / shape / f"{video_id}.pt").write_bytes(b"")
    jobs = ref.jobs(args_for(data, tmp_path / "out", noise_dir=noise_dir))
    assert jobs[0]["saved_noise"] == noise_dir / "832x480" / "vbi2v_0001_832x480.pt"


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


def test_a_cache_dit_run_does_not_resume_into_a_folder_made_without_it(tmp_path):
    # An R1 folder's provenance predates the cache_dit setting.
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"model": "m", "steps": 40}))
    ref.check_resumable(path, {"model": "m", "steps": 40, "cache_dit": None})
    with pytest.raises(SystemExit, match="cache_dit"):
        ref.check_resumable(
            path,
            {"model": "m", "steps": 40, "cache_dit": {"residual_diff_threshold": 0.08}},
        )
    assert ref.cache_dit_settings(None) is None


def test_cache_dit_presets_are_tts_dbcache_recipes():
    # tt-metal WanDBCacheConfig.default(): F1B0, 0.12, <= 2 in a row, warmup
    # interval 2, TaylorSeer 1; high-noise 4 warmup / 8 cached, low-noise 2 / 20.
    tt = ref.cache_dit_settings("tt")
    assert tt == {
        "preset": "tt",
        "residual_diff_threshold": 0.12,
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_continuous_cached_steps": 2,
        "warmup_interval": 2,
        "cfg_diff_compute_separate": True,
        "taylorseer_order": 1,
        "high_noise": {"max_warmup_steps": 4, "max_cached_steps": 8},
        "low_noise": {"max_warmup_steps": 2, "max_cached_steps": 20},
    }
    quality = ref.cache_dit_settings("tt-quality")
    assert {k: v for k, v in quality.items() if tt.get(k) != v} == {
        "preset": "tt-quality",
        "residual_diff_threshold": 0.06,
        "warmup_interval": 1,
        "taylorseer_order": 0,
    }


def test_initial_noise_is_the_seeded_cpu_draw():
    torch = pytest.importorskip("torch")

    noise = ref.initial_noise(NoisePipe, 5, 832, 480)
    assert noise.shape == (1, 16, 21, 60, 104)
    assert noise.dtype == torch.float32
    expected = torch.randn(noise.shape, generator=torch.Generator("cpu").manual_seed(5))
    assert torch.equal(noise, expected)


def test_saved_noise_must_fit_the_video(tmp_path):
    torch = pytest.importorskip("torch")
    pipe = NoisePipe
    path = tmp_path / "noise.pt"
    torch.save(ref.initial_noise(pipe, 5, 832, 480), path)
    assert torch.equal(
        ref.saved_noise(pipe, path, 832, 480), ref.initial_noise(pipe, 5, 832, 480)
    )
    with pytest.raises(SystemExit, match="expected float32"):
        ref.saved_noise(pipe, path, 480, 832)


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
