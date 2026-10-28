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


VALIDATION_MANIFEST = json.loads(
    (I2V_EVAL_DATASETS_DIR / "vbench_i2v_50" / "manifest.json").read_text()
)


@pytest.fixture
def fake_eval_data(tmp_path, monkeypatch):
    """The HF eval data as a local folder: each first frame matching its sha256."""
    import hashlib

    from test_module._test_common import vbench_i2v

    data = {}
    for entry in VALIDATION_MANIFEST:
        payload = entry["video_id"].encode()
        data[entry["image"]] = payload
        entry_path = tmp_path / entry["image"]
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_bytes(payload)
    hashes = {
        e["image"]: hashlib.sha256(data[e["image"]]).hexdigest()
        for e in VALIDATION_MANIFEST
    }
    fetched = []

    def fake_snapshot(patterns):
        fetched.extend(patterns)
        return tmp_path

    monkeypatch.setattr(vbench_i2v, "hf_snapshot", fake_snapshot)
    monkeypatch.setattr(
        "test_module._test_common.video_generation_routing._validation_videos",
        lambda subset: [
            {**e, "image_sha256": hashes[e["image"]]}
            for e in VALIDATION_MANIFEST
            if subset == "all" or e[subset]
        ],
    )
    return fetched


@pytest.mark.parametrize(
    "value,count",
    [("vbench_i2v_50", 20), ("vbench_i2v_50:dev", 5), ("vbench_i2v_50:all", 65)],
)
def test_validation_set_runs_its_subset_ci_by_default(monkeypatch, value, count):
    monkeypatch.setenv(I2V_EVAL_PAIRS_ENV, value)
    assert i2v_eval_num_pairs() == count


@pytest.mark.parametrize("value", ["vbench_i2v_50:nightly", "vbench_i2v:dev"])
def test_bad_subsets_are_refused(monkeypatch, value):
    monkeypatch.setenv(I2V_EVAL_PAIRS_ENV, value)
    with pytest.raises(ValueError, match="I2V_EVAL_PAIRS"):
        i2v_eval_num_pairs()


def test_validation_videos_carry_their_request(monkeypatch, fake_eval_data):
    from test_module._test_common.video_generation_routing import (
        WAN22_NEGATIVE_PROMPT,
    )

    monkeypatch.setenv(I2V_EVAL_PAIRS_ENV, "vbench_i2v_50:dev")
    pairs = load_i2v_eval_pairs(i2v_eval_num_pairs())
    dev = [e for e in VALIDATION_MANIFEST if e["dev"]]
    assert fake_eval_data == [e["image"] for e in dev]
    assert {p["shape"] for p in pairs} == {
        (832, 480),
        (480, 832),
        (1280, 720),
        (720, 1280),
    }
    for pair, entry in zip(pairs, dev):
        assert pair["video_id"] == entry["video_id"]
        assert pair["seed"] == entry["seed"]
        assert pair["prompt"] == entry["caption"]
        assert pair["image"] == f"{entry['pair_id']}.jpg"
        assert pair["negative_prompt"] == WAN22_NEGATIVE_PROMPT
    assert any(p.get("camera_motion") for p in pairs)


def test_a_first_frame_that_does_not_match_is_refused(
    monkeypatch, fake_eval_data, tmp_path
):
    monkeypatch.setenv(I2V_EVAL_PAIRS_ENV, "vbench_i2v_50:dev")
    first = next(e for e in VALIDATION_MANIFEST if e["dev"])
    (tmp_path / first["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        load_i2v_eval_pairs(5)


def test_wan_negative_prompt_matches_the_gpu_reference_script():
    import importlib.util

    from test_module._test_common.video_generation_routing import (
        WAN22_NEGATIVE_PROMPT,
    )

    script = I2V_EVAL_DATASETS_DIR.parents[1] / "scripts" / "wan_i2v_gpu_reference.py"
    spec = importlib.util.spec_from_file_location("wan_ref", script.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.WAN_NEGATIVE_PROMPT == WAN22_NEGATIVE_PROMPT


def test_payload_sends_the_videos_seed_and_negative_prompt():
    from test_module._test_common.video_generation_routing import (
        build_video_generation_payload,
    )

    payload = build_video_generation_payload(
        prompt="p",
        num_inference_steps=40,
        model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        image_b64="ZmFrZQ==",
        shape=(480, 832),
        seed=913,
        negative_prompt="n",
    )
    assert (payload["seed"], payload["negative_prompt"]) == (913, "n")
    assert (payload["aspect_ratio"], payload["size"]) == ("9:16", "480p")
