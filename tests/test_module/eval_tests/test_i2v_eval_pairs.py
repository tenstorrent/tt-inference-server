# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The I2V eval's COCO image-caption pairs, and that the eval sends them."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from test_module._test_common import load_i2v_eval_pairs
from test_module._test_common.video_generation_routing import I2V_EVAL_PAIRS_DIR

MANIFEST = json.loads((I2V_EVAL_PAIRS_DIR / "manifest.json").read_text())


def test_manifest_has_five_pairs_with_files():
    assert len(MANIFEST) == 5
    assert len({entry["image"] for entry in MANIFEST}) == 5
    for entry in MANIFEST:
        assert (I2V_EVAL_PAIRS_DIR / entry["image"]).is_file()
        assert entry["caption"].strip()


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda e: e["image"])
def test_images_are_16_by_9_and_match_the_manifest(entry):
    image_module = pytest.importorskip("PIL.Image")
    size = image_module.open(I2V_EVAL_PAIRS_DIR / entry["image"]).size
    assert list(size) == entry["size"]
    assert abs(size[0] / size[1] - 16 / 9) < 0.01


def test_loader_pairs_each_caption_with_its_image():
    pairs = load_i2v_eval_pairs(5)
    assert [p["prompt"] for p in pairs] == [e["caption"] for e in MANIFEST]
    for pair, entry in zip(pairs, MANIFEST):
        assert pair["image"] == entry["image"]
        assert (
            base64.b64decode(pair["image_b64"])
            == (I2V_EVAL_PAIRS_DIR / entry["image"]).read_bytes()
        )


def test_loader_refuses_more_pairs_than_exist():
    with pytest.raises(ValueError, match="has 5"):
        load_i2v_eval_pairs(6)


def test_i2v_eval_sends_each_prompt_with_its_own_image(monkeypatch, tmp_path):
    # The eval module imports the CLIP encoder, which needs torch.
    pytest.importorskip("torch")
    from test_module.eval_tests import video_generation_eval_test as mod

    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append(json)
        return SimpleNamespace(status_code=202, json=lambda: {"id": f"job{len(sent)}"})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod, "DATASET_DIR", str(tmp_path / "videos"))
    test = mod.VideoGenerationEvalsTest.__new__(mod.VideoGenerationEvalsTest)
    monkeypatch.setattr(test, "wait_for_server_ready", lambda: True, raising=False)
    monkeypatch.setattr(
        test, "_poll_video_completion", lambda **kw: f"/tmp/{kw['job_id']}.mp4"
    )

    pairs = load_i2v_eval_pairs(5)
    videos = test._generate_videos(
        prompts=[p["prompt"] for p in pairs],
        model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        server_url="http://x",
        num_inference_steps=40,
        conditioning=pairs,
    )

    assert [s["prompt"] for s in sent] == [p["prompt"] for p in pairs]
    assert [s["image_prompts"][0]["image"] for s in sent] == [
        p["image_b64"] for p in pairs
    ]
    assert [v["image"] for v in videos] == [p["image"] for p in pairs]


def test_every_pair_has_concepts():
    for pair in load_i2v_eval_pairs(5):
        assert 2 <= len(pair["concepts"]) <= 3


def test_concepts_are_scored_per_clip(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    from test_module._test_common import video_quality_metrics as vqm
    from test_module.eval_tests import video_generation_eval_test as mod
    from tests.test_module.video_clips import moving_frames, write_clip

    class FakeScorer:
        """Scores text by its length, so each concept gets a distinct, known value."""

        def score(self, *, frames, texts):
            frames = list(frames)
            return np.array([[len(t) for t in texts]] * len(frames), float), None

        def consecutive_image_similarity(self, features):
            return []

    monkeypatch.setattr(vqm, "BatchedCLIPScorer", FakeScorer)
    clip = str(write_clip(tmp_path / "c.mp4", moving_frames(16)))
    pairs = load_i2v_eval_pairs(2)
    videos = [
        {"prompt": p["prompt"], "video_path": clip, "image": p["image"]} for p in pairs
    ]
    test = mod.VideoGenerationEvalsTest.__new__(mod.VideoGenerationEvalsTest)

    results = test._score_concepts(videos, pairs)

    assert videos[0]["concept_scores"] == {
        c: float(len(c)) for c in pairs[0]["concepts"]
    }
    shortest = min((len(c), c) for p in pairs for c in p["concepts"])
    assert results["min_concept_score"] == shortest[0]
    assert results["weakest_concept"].startswith(shortest[1])


def test_concept_scoring_is_skipped_without_concepts():
    pytest.importorskip("torch")
    from test_module.eval_tests import video_generation_eval_test as mod

    test = mod.VideoGenerationEvalsTest.__new__(mod.VideoGenerationEvalsTest)
    assert test._score_concepts([], None) is None
