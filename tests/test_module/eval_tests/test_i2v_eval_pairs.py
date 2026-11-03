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
    WAN22_NEGATIVE_PROMPT,
    build_video_generation_payload,
)

SHAPES = {(832, 480), (480, 832), (1280, 720), (720, 1280)}
# The validation set's layout: the dev set's 5 videos (each shape, plus a
# second 832x480 one) lead, within the ci set's 20, within all 65.
DEV_SHAPES = [(832, 480), (832, 480), (480, 832), (1280, 720), (720, 1280)]


def _manifest() -> list[dict]:
    manifest = []
    for i in range(65):
        width, height = DEV_SHAPES[i] if i < 5 else (832, 480)
        pair_id = f"vbi2v_{i:04d}"
        video_id = f"{pair_id}_{width}x{height}"
        manifest.append(
            {
                "pair_id": pair_id,
                "video_id": video_id,
                "image": f"images/{width}x{height}/{pair_id}.jpg",
                "image_sha256": hashlib.sha256(video_id.encode()).hexdigest(),
                "caption": f"prompt {i}" + (", camera pans left" if i == 1 else ""),
                "camera_motion": "pan_left" if i == 1 else None,
                "width": width,
                "height": height,
                "seed": 1000 + i,
                "dev": i < 5,
                "ci": i < 20,
            }
        )
    return manifest


MANIFEST = _manifest()


@pytest.fixture
def eval_data(tmp_path, monkeypatch):
    """The HF eval data as a local folder, each first frame matching its sha256."""
    from test_module._test_common import vbench_i2v

    for entry in MANIFEST:
        path = tmp_path / entry["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry["video_id"].encode())
    manifest_path = tmp_path / I2V_EVAL_MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(MANIFEST))
    fetched = []

    def fake_snapshot(patterns):
        fetched.extend(patterns)
        return tmp_path

    monkeypatch.setattr(vbench_i2v, "hf_snapshot", fake_snapshot)
    return SimpleNamespace(root=tmp_path, fetched=fetched)


@pytest.mark.parametrize(
    "subset,count", [(None, 20), ("dev", 5), ("ci", 20), ("all", 65)]
)
def test_each_subset_and_ci_by_default(eval_data, subset, count):
    assert i2v_eval_num_pairs(subset) == count


@pytest.mark.parametrize(
    "limit_samples_mode,subset",
    [
        # CI's run-full-evals: no --ci-mode, no limit.
        (None, "all"),
        # --ci-mode, CI's default.
        ("ci-nightly", "ci"),
        ("ci-commit", "ci"),
        ("ci-long", "ci"),
        ("smoke-test", "dev"),
    ],
)
def test_the_runs_eval_scope_picks_the_subset(limit_samples_mode, subset):
    from test_module._test_common import i2v_eval_subset

    assert i2v_eval_subset(limit_samples_mode) == subset


def test_the_dev_subset_is_the_manifests_dev_videos(eval_data):
    pairs = load_i2v_eval_pairs(5, "dev")
    assert [p["video_id"] for p in pairs] == [
        e["video_id"] for e in MANIFEST if e["dev"]
    ]


@pytest.mark.parametrize(
    "limit_samples_mode,count,subset", [(None, 65, "all"), ("ci-nightly", 20, "ci")]
)
def test_the_eval_requests_the_runs_subset(
    monkeypatch, eval_data, limit_samples_mode, count, subset
):
    import sys

    from test_module.eval_tests import video_eval_tests as mod

    seen = {}

    class FakeTest:
        def __init__(self, config, targets):
            seen["request"] = targets["request"]

        def run_tests(self):
            return SimpleNamespace(data={"eval_results": {}})

    # The real generation eval imports the CLIP encoder, which needs torch.
    monkeypatch.setitem(
        sys.modules,
        "test_module.eval_tests.video_generation_eval_test",
        SimpleNamespace(
            VideoGenerationEvalsTest=FakeTest,
            VideoGenerationEvalsTestRequest=SimpleNamespace,
        ),
    )
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(
            hf_model_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers", env_vars={}
        ),
        runtime_config=SimpleNamespace(limit_samples_mode=limit_samples_mode),
        base_url="http://x",
    )
    mod._run_video_generation_eval(ctx)
    assert (seen["request"].num_prompts, seen["request"].i2v_subset) == (
        count,
        subset,
    )


def test_an_unknown_subset_is_refused(eval_data):
    with pytest.raises(ValueError, match="nightly"):
        i2v_eval_num_pairs("nightly")


def test_each_video_carries_its_request(eval_data):
    pairs = load_i2v_eval_pairs(i2v_eval_num_pairs("dev"), "dev")
    dev = [e for e in MANIFEST if e["dev"]]
    assert eval_data.fetched == [I2V_EVAL_MANIFEST] * 2 + [e["image"] for e in dev]
    assert {p["shape"] for p in pairs} == SHAPES
    for pair, entry in zip(pairs, dev):
        assert pair["video_id"] == entry["video_id"]
        assert pair["seed"] == entry["seed"]
        assert pair["prompt"] == entry["caption"]
        assert pair["image"] == f"{entry['pair_id']}.jpg"
        assert pair["negative_prompt"] == WAN22_NEGATIVE_PROMPT
    assert any(p.get("camera_motion") for p in pairs)


def test_a_first_frame_that_does_not_match_is_refused(eval_data):
    first = next(e for e in MANIFEST if e["dev"])
    (eval_data.root / first["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        load_i2v_eval_pairs(5, "dev")


def test_the_eval_refuses_more_videos_than_its_subset_has(eval_data):
    with pytest.raises(ValueError, match="wants 6 videos"):
        load_i2v_eval_pairs(6, "dev")


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

    pairs = load_i2v_eval_pairs(5, "dev")
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
