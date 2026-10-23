# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image edit LLM judge, against a fake gateway."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from test_module._test_common import image_edit_llm_judge as judge

Image = pytest.importorskip("PIL.Image")

GOOD_REPLY = {
    "scores": {name: 4 for name in judge.CRITERIA},
    "issues": ["the road disappears under the snow"],
    "summary": "Snow added; framing kept.",
}


@pytest.fixture
def edits_dir(tmp_path):
    edits = tmp_path / "images" / "evals"
    edits.mkdir(parents=True)
    Image.new("RGB", (512, 512), "green").save(edits / "input.png")
    Image.new("RGB", (1024, 1024), "white").save(edits / "00_edit.jpg")
    Image.new("RGB", (1024, 1024), "navy").save(edits / "01_edit.jpg")
    (edits / "index.json").write_text(
        json.dumps(
            [
                {
                    "file": "00_edit.jpg",
                    "instruction": "Cover it with snow.",
                    "expected": "a snowy mountain",
                    "input_image": "input.png",
                    "problem": None,
                },
                {
                    "file": "01_edit.jpg",
                    "instruction": "Make it night.",
                    "expected": "a mountain at night",
                    "input_image": "input.png",
                    "problem": None,
                },
            ]
        )
    )
    return edits


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


def test_each_edit_is_sent_with_its_instruction_input_and_output(
    monkeypatch, edits_dir
):
    calls = _fake_gateway(
        monkeypatch, [_reply(json.dumps(GOOD_REPLY)), _reply(json.dumps(GOOD_REPLY))]
    )

    result = judge.judge_directory(
        edits_dir, api_key="k", base_url="https://gw", model="m"
    )

    assert result["num_judged"] == 2 and result["mean_overall"] == 4
    first = calls[0]
    assert first["url"] == "https://gw/v1/chat/completions"
    assert first["json"]["model"] == "m" and first["json"]["temperature"] == 0
    parts = first["json"]["messages"][0]["content"]
    assert {"type": "text", "text": "Instruction: Cover it with snow."} in parts
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 2  # input, then edited
    texts = [p["text"] for p in parts if p["type"] == "text"]
    assert texts.index("Input image:") < texts.index("Edited image:")


def test_reply_must_score_the_edit_rubric():
    with pytest.raises(ValueError, match="missing scores"):
        judge.parse_verdict(json.dumps({"scores": {"prompt_adherence": 4}}))
    verdict = judge.parse_verdict("```json\n" + json.dumps(GOOD_REPLY) + "\n```")
    assert set(verdict["scores"]) == set(judge.CRITERIA)


def test_a_failed_edit_is_recorded_and_the_rest_still_run(monkeypatch, edits_dir):
    _fake_gateway(monkeypatch, [(500, "boom"), _reply(json.dumps(GOOD_REPLY))])
    result = judge.judge_directory(edits_dir, api_key="k", base_url="https://gw")
    first, second = result["edits"]
    assert "gateway returned 500" in first["error"]
    assert second["overall"] == 4


def test_main_writes_verdicts_and_a_summary(monkeypatch, edits_dir, tmp_path):
    monkeypatch.setenv("TT_CHAT_API_KEY", "k")
    monkeypatch.setenv("TT_CHAT_URL", "https://gw")
    _fake_gateway(monkeypatch, [_reply(json.dumps(GOOD_REPLY)), _reply("not json")])
    summary = tmp_path / "summary.md"

    assert (
        judge.main(["--edits-dir", str(edits_dir), "--summary-md", str(summary)]) == 0
    )

    saved = json.loads((edits_dir / "llm_judge.json").read_text())
    assert saved["num_judged"] == 1
    table = summary.read_text()
    assert "Image edit LLM judge" in table
    assert "road disappears" in table and "not judged" in table


def test_main_skips_without_credentials(monkeypatch, edits_dir, capsys):
    monkeypatch.delenv("TT_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("TT_CHAT_URL", raising=False)
    assert judge.main(["--edits-dir", str(edits_dir)]) == 0
    assert "skipped" in capsys.readouterr().out


def test_main_fails_the_step_when_nothing_was_judged(monkeypatch, edits_dir):
    monkeypatch.setenv("TT_CHAT_API_KEY", "k")
    monkeypatch.setenv("TT_CHAT_URL", "https://gw")
    _fake_gateway(monkeypatch, [(400, "nope"), (400, "nope")])
    assert judge.main(["--edits-dir", str(edits_dir)]) == 1
