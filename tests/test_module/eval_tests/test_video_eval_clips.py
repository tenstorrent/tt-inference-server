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


def test_no_clips_writes_nothing(tmp_path):
    _save_eval_clips(SimpleNamespace(output_path=str(tmp_path)), [])
    assert not (tmp_path / "videos").exists()
