# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The video LLM judge, against a fake TT chat gateway."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from test_module._test_common import video_llm_judge as judge
from tests.test_module.video_clips import moving_frames, write_clip

pytest.importorskip("numpy")
pytest.importorskip("imageio_ffmpeg")
Image = pytest.importorskip("PIL.Image")

GOOD_REPLY = {
    "scores": {name: 4 for name in judge.CRITERIA},
    "issues": ["slight washout at the end"],
    "summary": "Follows the prompt.",
}


@pytest.mark.parametrize(
    "base,url",
    [
        ("https://gw.example", "https://gw.example/v1/chat/completions"),
        ("https://gw.example/", "https://gw.example/v1/chat/completions"),
        ("https://gw.example/v1", "https://gw.example/v1/chat/completions"),
    ],
)
def test_gateway_url_targets_the_v1_route(base, url):
    assert judge.gateway_url(base) == url


def test_parse_verdict_accepts_fenced_json():
    verdict = judge.parse_verdict("```json\n" + json.dumps(GOOD_REPLY) + "\n```")
    assert verdict["overall"] == 4
    assert verdict["issues"] == ["slight washout at the end"]


@pytest.mark.parametrize(
    "reply,match",
    [
        ("no json here", "no JSON"),
        (json.dumps({"scores": {"prompt_adherence": 4}}), "missing scores"),
        (
            json.dumps(
                {**GOOD_REPLY, "scores": {**GOOD_REPLY["scores"], "visual_quality": 9}}
            ),
            "not 1-5",
        ),
    ],
)
def test_parse_verdict_rejects_bad_replies(reply, match):
    with pytest.raises(ValueError, match=match):
        judge.parse_verdict(reply)


@pytest.fixture
def eval_dir(tmp_path):
    videos = tmp_path / "videos" / "evals"
    videos.mkdir(parents=True)
    write_clip(videos / "00_a.mp4", moving_frames(81))
    write_clip(videos / "01_b.mp4", moving_frames(81))
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 36), "blue").save(images / "a.jpg")
    (videos / "index.json").write_text(
        json.dumps(
            [
                {"file": "00_a.mp4", "prompt": "a bird", "conditioning_image": "a.jpg"},
                {"file": "01_b.mp4", "prompt": "a train"},
            ]
        )
    )
    return videos, images


def _fake_gateway(monkeypatch, replies):
    calls = []

    def post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        status, body = replies.pop(0)
        return SimpleNamespace(status_code=status, text=str(body), json=lambda: body)

    monkeypatch.setattr(judge.requests, "post", post)
    return calls


def _reply(content):
    return 200, {"choices": [{"message": {"content": content}}]}


def test_each_clip_is_sent_with_its_image_and_frames(monkeypatch, eval_dir):
    videos, images = eval_dir
    calls = _fake_gateway(
        monkeypatch, [_reply(json.dumps(GOOD_REPLY)), _reply(json.dumps(GOOD_REPLY))]
    )

    result = judge.judge_directory(
        videos, api_key="k", base_url="https://gw", model="m", images_dir=images
    )

    assert result["num_judged"] == 2 and result["mean_overall"] == 4
    first, second = calls
    assert first["url"] == "https://gw/v1/chat/completions"
    assert first["headers"] == {"Authorization": "Bearer k"}
    assert first["json"]["model"] == "m" and first["json"]["temperature"] == 0
    parts = first["json"]["messages"][0]["content"]
    images_sent = [p for p in parts if p["type"] == "image_url"]
    assert len(images_sent) == 1 + judge.NUM_FRAMES  # conditioning image + frames
    assert images_sent[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert {"type": "text", "text": "Prompt: a bird"} in parts
    # A clip without a conditioning image is sent its frames only.
    second_images = [
        p for p in second["json"]["messages"][0]["content"] if p["type"] == "image_url"
    ]
    assert len(second_images) == judge.NUM_FRAMES


def test_a_failed_clip_is_recorded_and_the_rest_still_run(monkeypatch, eval_dir):
    videos, images = eval_dir
    _fake_gateway(
        monkeypatch, [(500, {"error": "boom"}), _reply(json.dumps(GOOD_REPLY))]
    )

    result = judge.judge_directory(
        videos, api_key="k", base_url="https://gw", images_dir=images
    )

    first, second = result["clips"]
    assert "gateway returned 500" in first["error"]
    assert second["overall"] == 4
    assert result["num_judged"] == 1


def test_main_skips_without_a_key(monkeypatch, eval_dir, capsys):
    monkeypatch.delenv("TT_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("TT_CHAT_URL", raising=False)
    assert judge.main(["--videos-dir", str(eval_dir[0])]) == 0
    assert "skipped" in capsys.readouterr().out
    assert not (eval_dir[0] / "llm_judge.json").exists()


def test_main_writes_the_verdicts_and_a_summary(monkeypatch, eval_dir, tmp_path):
    videos, images = eval_dir
    monkeypatch.setenv("TT_CHAT_API_KEY", "k")
    monkeypatch.setenv("TT_CHAT_URL", "https://gw")
    _fake_gateway(monkeypatch, [_reply(json.dumps(GOOD_REPLY)), _reply("not json")])
    summary = tmp_path / "summary.md"

    code = judge.main(
        [
            "--videos-dir",
            str(videos),
            "--images-dir",
            str(images),
            "--summary-md",
            str(summary),
        ]
    )

    assert code == 0
    saved = json.loads((videos / "llm_judge.json").read_text())
    assert saved["num_judged"] == 1
    table = summary.read_text()
    assert (
        "Video LLM judge" in table
        and "slight washout" in table
        and "not judged" in table
    )


def test_main_fails_the_step_when_nothing_was_judged(monkeypatch, eval_dir):
    videos, images = eval_dir
    monkeypatch.setenv("TT_CHAT_API_KEY", "k")
    monkeypatch.setenv("TT_CHAT_URL", "https://gw")
    _fake_gateway(
        monkeypatch, [(400, "images not supported"), (400, "images not supported")]
    )
    assert judge.main(["--videos-dir", str(videos), "--images-dir", str(images)]) == 1
