# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for scripts/select_vbench_i2v_pairs.py and the manifest it committed."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "select_vbench_i2v_pairs.py"

spec = importlib.util.spec_from_file_location("select_vbench_i2v_pairs", SCRIPT)
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)

# VBench-I2V's image counts per category; the last four get camera prompts.
CATEGORIES = {
    "single-human": 64,
    "animal": 46,
    "food": 38,
    "multiple-human": 34,
    "transportation": 31,
    "plant": 23,
    "other": 10,
    "architecture": 40,
    "scenery": 40,
    "indoor": 21,
    "abstract": 8,
}
BACKGROUND = {"architecture", "scenery", "indoor", "abstract"}
SHAPES = {(832, 480): 40, (480, 832): 10, (1280, 720): 11, (720, 1280): 4}
CI = {(832, 480): 10, (480, 832): 4, (1280, 720): 4, (720, 1280): 2}


def fake_full_info() -> list[dict]:
    """VBench-I2V's layout: one base prompt per image, seven camera ones per background image."""
    info = []
    for category, count in CATEGORIES.items():
        for i in range(count):
            name = f"{category} {i}.jpg"
            info.append(
                {
                    "prompt_en": f"{category} {i}",
                    "image_type": category,
                    "image_name": name,
                    "dimension": ["aesthetic_quality"],
                }
            )
            if category in BACKGROUND:
                info += [
                    {
                        "prompt_en": f"{category} {i}, {phrase}",
                        "image_type": category,
                        "image_name": name,
                        "dimension": ["camera_motion"],
                    }
                    for phrase in select.CAMERA_MOVES
                ]
    return info


def check_videos(entries: list[dict]) -> None:
    shapes = Counter((e["width"], e["height"]) for e in entries)
    assert shapes == SHAPES
    assert Counter((e["width"], e["height"]) for e in entries if e["ci"]) == CI
    dev = [(e["width"], e["height"]) for e in entries if e["dev"]]
    assert sorted(Counter(dev).values()) == [1, 1, 1, 2]
    assert Counter(dev)[(832, 480)] == 2
    assert all(e["ci"] for e in entries if e["dev"])
    # Every 720p video has a 480p twin of the same aspect.
    small = {
        (e["pair_id"], e["width"] > e["height"])
        for e in entries
        if max(e["width"], e["height"]) == 832
    }
    for e in entries:
        if max(e["width"], e["height"]) == 1280:
            assert (e["pair_id"], e["width"] > e["height"]) in small

    pairs = {e["pair_id"]: e for e in entries}
    assert len(pairs) == 50
    assert len({e["vbench_image"] for e in pairs.values()}) == 50
    assert set(Counter(e["vbench_category"] for e in pairs.values())) == set(CATEGORIES)
    moves = [e["camera_motion"] for e in pairs.values() if e["camera_motion"]]
    assert len(moves) == 10
    assert set(moves) == set(select.CAMERA_MOVES.values())
    assert all(
        pairs[p]["vbench_category"] in BACKGROUND
        for p in pairs
        if pairs[p]["camera_motion"]
    )


def test_selection_is_deterministic_and_has_the_planned_counts():
    info = fake_full_info()
    first = select.videos(select.select_pairs(info))
    assert first == select.videos(select.select_pairs(info))
    check_videos(first)
    assert first != select.videos(select.select_pairs(info, seed=1))


def test_portrait_exclude_draws_another_pair():
    info = fake_full_info()
    portrait = [
        e for e in select.videos(select.select_pairs(info)) if e["height"] > e["width"]
    ]
    excluded = portrait[0]["vbench_image"]
    again = select.videos(
        select.select_pairs(info, portrait_exclude=frozenset({excluded}))
    )
    assert excluded not in {
        e["vbench_image"] for e in again if e["height"] > e["width"]
    }
    check_videos(again)


def test_committed_manifest_has_the_planned_counts():
    manifest = json.loads(select.MANIFEST_PATH.read_text())
    check_videos(manifest)
    for e in manifest:
        assert e["image"] == f"images/{e['width']}x{e['height']}/{e['pair_id']}.jpg"
        assert len(e["image_sha256"]) == 64
        assert e["seed"] == int(e["pair_id"].split("_")[1])


def test_category_quotas_cover_every_category():
    quotas = select.category_quotas(Counter(CATEGORIES), 50)
    assert sum(quotas.values()) == 50
    assert min(quotas.values()) == 1
    assert quotas["single-human"] == 9


@pytest.mark.parametrize(
    "size, shape, box",
    [
        ((3024, 1701), (1280, 720), (0, 0, 3024, 1701)),
        ((3024, 1701), (832, 480), (38, 0, 2986, 1701)),
        ((3024, 3024), (480, 832), (639, 0, 2384, 3024)),
        ((3024, 3024), (1280, 720), (0, 661, 3024, 2362)),
    ],
)
def test_center_crop_box(size, shape, box):
    assert select.center_crop_box(size, shape) == box


def test_prepare_first_frame_is_the_output_shape(tmp_path):
    from PIL import Image

    for ratio, size in (("16-9", (320, 180)), ("1-1", (300, 300))):
        (tmp_path / ratio).mkdir()
        Image.new("RGB", size, "red").save(tmp_path / ratio / "x.jpg")
    for shape in SHAPES:
        assert select.prepare_first_frame(tmp_path, "x.jpg", shape).size == shape


def test_camera_move_reads_vbench_prompts():
    assert select.camera_move("a lake, camera pans left") == "pan_left"
    assert select.camera_move("a lake, camera static") == "static"
    assert select.camera_move("a lake") is None
