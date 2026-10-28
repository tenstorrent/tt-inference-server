# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The I2V eval's videos (the validation set), and that the eval sends each its own request."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from test_module._test_common import i2v_eval_num_pairs, load_i2v_eval_pairs
from test_module._test_common.video_generation_routing import (
    I2V_EVAL_MANIFEST,
    I2V_EVAL_SUBSET_ENV,
    WAN22_NEGATIVE_PROMPT,
    build_video_generation_payload,
)

MANIFEST = json.loads(I2V_EVAL_MANIFEST.read_text())
SHAPES = {(832, 480), (480, 832), (1280, 720), (720, 1280)}


@pytest.fixture
def eval_data(tmp_path, monkeypatch):
    """The HF eval data as a local folder, each first frame matching its sha256."""
    from test_module._test_common import vbench_i2v, video_generation_routing

    manifest = []
    for entry in MANIFEST:
        path = tmp_path / entry["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry["video_id"].encode())
        sha = hashlib.sha256(entry["video_id"].encode()).hexdigest()
        manifest.append({**entry, "image_sha256": sha})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(video_generation_routing, "I2V_EVAL_MANIFEST", manifest_path)
    fetched = []

    def fake_snapshot(patterns):
        fetched.extend(patterns)
        return tmp_path

    monkeypatch.setattr(vbench_i2v, "hf_snapshot", fake_snapshot)
    return SimpleNamespace(root=tmp_path, fetched=fetched)


@pytest.mark.parametrize(
    "subset,count", [(None, 20), ("dev", 5), ("ci", 20), ("all", 65)]
)
def test_the_eval_runs_its_subset_ci_by_default(monkeypatch, subset, count):
    if subset:
        monkeypatch.setenv(I2V_EVAL_SUBSET_ENV, subset)
    else:
        monkeypatch.delenv(I2V_EVAL_SUBSET_ENV, raising=False)
    assert i2v_eval_num_pairs() == count


def test_an_unknown_subset_is_refused(monkeypatch):
    monkeypatch.setenv(I2V_EVAL_SUBSET_ENV, "nightly")
    with pytest.raises(ValueError, match=I2V_EVAL_SUBSET_ENV):
        i2v_eval_num_pairs()


def test_each_video_carries_its_request(monkeypatch, eval_data):
    monkeypatch.setenv(I2V_EVAL_SUBSET_ENV, "dev")
    pairs = load_i2v_eval_pairs(i2v_eval_num_pairs())
    dev = [e for e in MANIFEST if e["dev"]]
    assert eval_data.fetched == [e["image"] for e in dev]
    assert {p["shape"] for p in pairs} == SHAPES
    for pair, entry in zip(pairs, dev):
        assert pair["video_id"] == entry["video_id"]
        assert pair["seed"] == entry["seed"]
        assert pair["prompt"] == entry["caption"]
        assert pair["image"] == f"{entry['pair_id']}.jpg"
        assert pair["negative_prompt"] == WAN22_NEGATIVE_PROMPT
    assert any(p.get("camera_motion") for p in pairs)


def test_a_first_frame_that_does_not_match_is_refused(monkeypatch, eval_data):
    monkeypatch.setenv(I2V_EVAL_SUBSET_ENV, "dev")
    first = next(e for e in MANIFEST if e["dev"])
    (eval_data.root / first["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        load_i2v_eval_pairs(5)


def test_the_eval_refuses_more_videos_than_its_subset_has(monkeypatch, eval_data):
    monkeypatch.setenv(I2V_EVAL_SUBSET_ENV, "dev")
    with pytest.raises(ValueError, match="wants 6 videos"):
        load_i2v_eval_pairs(6)


def test_wan_negative_prompt_matches_the_gpu_reference_script():
    import importlib.util

    script = I2V_EVAL_MANIFEST.parents[3] / "scripts" / "wan_i2v_gpu_reference.py"
    spec = importlib.util.spec_from_file_location("wan_ref", script.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.WAN_NEGATIVE_PROMPT == WAN22_NEGATIVE_PROMPT


def test_payload_sends_the_videos_shape_seed_and_negative_prompt():
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


def test_the_eval_sends_each_video_its_request_and_rejects_wrong_shapes(
    monkeypatch, tmp_path, eval_data
):
    # The eval module imports the CLIP encoder, which needs torch.
    pytest.importorskip("torch")
    from test_module.eval_tests import video_generation_eval_test as mod

    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append(json)
        return SimpleNamespace(status_code=202, json=lambda: {"id": f"job{len(sent)}"})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod, "DATASET_DIR", str(tmp_path / "videos"))
    # The deployment serves 1280x720 only: every other shape comes back wrong.
    monkeypatch.setattr(
        mod,
        "check_video_output",
        lambda path, **kw: None if kw["shape"] == (1280, 720) else "wrong shape",
    )
    test = mod.VideoGenerationEvalsTest.__new__(mod.VideoGenerationEvalsTest)
    monkeypatch.setattr(test, "wait_for_server_ready", lambda: True, raising=False)
    monkeypatch.setattr(
        test, "_poll_video_completion", lambda **kw: f"/tmp/{kw['job_id']}.mp4"
    )
    monkeypatch.setenv(I2V_EVAL_SUBSET_ENV, "dev")

    pairs = load_i2v_eval_pairs(5)
    videos = test._generate_videos(
        prompts=[p["prompt"] for p in pairs],
        model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        server_url="http://x",
        num_inference_steps=40,
        conditioning=pairs,
    )

    assert [s["seed"] for s in sent] == [p["seed"] for p in pairs]
    assert [s["image_prompts"][0]["image"] for s in sent] == [
        p["image_b64"] for p in pairs
    ]
    assert {s["negative_prompt"] for s in sent} == {WAN22_NEGATIVE_PROMPT}
    assert [v["video_id"] for v in videos] == [
        p["video_id"] for p in pairs if p["shape"] == (1280, 720)
    ]
    assert len(test.rejected_videos) == 4
    assert {v["problem"] for v in test.rejected_videos} == {"wrong shape"}
