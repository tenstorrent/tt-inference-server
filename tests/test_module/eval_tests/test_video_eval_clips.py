# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Every video eval clip is kept in the run output, with its prompt."""

from __future__ import annotations

import json
from types import SimpleNamespace

from test_module.eval_tests.video_eval_tests import _save_eval_clips


def test_eval_clips_are_kept_with_their_prompts(tmp_path):
    clips = []
    for i in range(2):
        path = tmp_path / f"{i}.mp4"
        path.write_bytes(f"clip {i}".encode())
        clips.append(
            {"prompt": f"prompt {i}", "video_path": str(path), "job_id": f"j{i}"}
        )
    ctx = SimpleNamespace(output_path=str(tmp_path / "out"))

    _save_eval_clips(ctx, clips)

    evals = tmp_path / "out/videos/evals"
    assert (evals / "00_j0.mp4").read_bytes() == b"clip 0"
    assert (evals / "01_j1.mp4").read_bytes() == b"clip 1"
    assert json.loads((evals / "index.json").read_text()) == [
        {"file": "00_j0.mp4", "prompt": "prompt 0"},
        {"file": "01_j1.mp4", "prompt": "prompt 1"},
    ]


def test_each_clips_step_caching_is_kept_beside_the_clips(tmp_path):
    summary = [{"name": "high_noise", "executed_steps": 15, "cached_steps": [[3]]}]
    clips = []
    for i in range(2):
        path = tmp_path / f"{i}.mp4"
        path.write_bytes(b"clip")
        clips.append(
            {
                "prompt": "p",
                "video_path": str(path),
                "job_id": f"j{i}",
                "video_id": f"vbi2v_000{i}_832x480",
            }
        )
    # A server that does not report it leaves the clip out.
    clips[0]["cache_summary"] = summary

    _save_eval_clips(SimpleNamespace(output_path=str(tmp_path / "out")), clips)

    evals = tmp_path / "out/videos/evals"
    assert json.loads((evals / "cache_summary.json").read_text()) == [
        {
            "file": "00_vbi2v_0000_832x480.mp4",
            "video_id": "vbi2v_0000_832x480",
            "job_id": "j0",
            "cache_summary": summary,
        }
    ]
    # Kept for comparison only: not in the index the scorers read.
    assert all(
        "cache_summary" not in e for e in json.loads((evals / "index.json").read_text())
    )


def test_no_cache_summaries_write_no_file(tmp_path):
    path = tmp_path / "0.mp4"
    path.write_bytes(b"clip")
    _save_eval_clips(
        SimpleNamespace(output_path=str(tmp_path / "out")),
        [{"prompt": "p", "video_path": str(path), "job_id": "j0"}],
    )
    assert not (tmp_path / "out/videos/evals/cache_summary.json").exists()


def test_no_clips_writes_nothing(tmp_path):
    _save_eval_clips(SimpleNamespace(output_path=str(tmp_path)), [])
    assert not (tmp_path / "videos").exists()


def test_validation_clips_keep_their_request_and_rejects_go_with_failed(tmp_path):
    ok = tmp_path / "ok.mp4"
    ok.write_bytes(b"ok")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"bad")
    request = {
        "video_id": "vbi2v_0913_1280x720",
        "width": 1280,
        "height": 720,
        "seed": 913,
    }
    videos = [
        {
            "prompt": "camels",
            "video_path": str(ok),
            "image": "vbi2v_0913.jpg",
            **request,
        }
    ]
    rejected = [
        {
            "prompt": "neon",
            "video_path": str(bad),
            "video_id": "vbi2v_0048_480x832",
            "problem": "1280x720, expected 480x832",
        }
    ]
    _save_eval_clips(
        SimpleNamespace(output_path=str(tmp_path / "out")), videos, rejected
    )

    out = tmp_path / "out/videos"
    assert json.loads((out / "evals/index.json").read_text()) == [
        {
            "file": "00_vbi2v_0913_1280x720.mp4",
            "prompt": "camels",
            "conditioning_image": "vbi2v_0913.jpg",
            **request,
        }
    ]
    assert (out / "failed/evals/vbi2v_0048_480x832.mp4").read_bytes() == b"bad"
