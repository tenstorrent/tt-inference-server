# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Choosing the I2V eval's pair set, and the VBench-I2V pairs themselves."""

from __future__ import annotations

import json

import pytest

from test_module._test_common import i2v_eval_num_pairs, load_i2v_eval_pairs
from test_module._test_common.video_generation_routing import (
    I2V_EVAL_DATASETS_DIR,
    I2V_EVAL_PAIRS_ENV,
)

VBENCH_DIR = I2V_EVAL_DATASETS_DIR / "vbench_i2v"
VBENCH_MANIFEST = json.loads((VBENCH_DIR / "manifest.json").read_text())


def test_coco_is_the_default_set(monkeypatch):
    monkeypatch.delenv(I2V_EVAL_PAIRS_ENV, raising=False)
    assert i2v_eval_num_pairs() == 5
    assert load_i2v_eval_pairs(1)[0]["image"].startswith("coco_")


def test_vbench_set_runs_all_fifteen_pairs(monkeypatch):
    monkeypatch.setenv(I2V_EVAL_PAIRS_ENV, "vbench_i2v")
    pairs = load_i2v_eval_pairs(i2v_eval_num_pairs())
    assert len(pairs) == 15
    assert [p["image"] for p in pairs] == [e["image"] for e in VBENCH_MANIFEST]
    # Only the four camera-move prompts carry a move.
    assert sum("camera_motion" in p for p in pairs) == 4


def test_unknown_set_is_refused(monkeypatch):
    monkeypatch.setenv(I2V_EVAL_PAIRS_ENV, "imagenet")
    with pytest.raises(ValueError, match="I2V_EVAL_PAIRS"):
        i2v_eval_num_pairs()


def test_vbench_manifest_covers_every_vbench_category():
    categories = {e["vbench_category"] for e in VBENCH_MANIFEST}
    assert len(categories) == 11
    for entry in VBENCH_MANIFEST:
        assert (VBENCH_DIR / entry["image"]).is_file()
        assert entry["caption"].startswith(entry["vbench_caption"])
        assert entry["concepts"]


@pytest.mark.parametrize("entry", VBENCH_MANIFEST, ids=lambda e: e["image"])
def test_vbench_images_are_the_wan_canvas(entry):
    image_module = pytest.importorskip("PIL.Image")
    size = image_module.open(VBENCH_DIR / entry["image"]).size
    assert list(size) == entry["size"] == [1280, 720]
