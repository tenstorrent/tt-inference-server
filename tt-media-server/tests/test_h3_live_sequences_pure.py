# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tier-0 tests for the pure parts of h3_live_sequences (no server, no device)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("H3_LIVE_URL", "http://unit-test.invalid")  # h3_live_common reads it at import; never contacted
import h3_live_common as live  # noqa: E402
import h3_live_sequences as seq  # noqa: E402


@pytest.mark.parametrize(("aspect", "canvas"), [
    ("21:9", (1536, 672)), ("16:9", (1344, 768)), ("4:3", (1024, 768)),
    ("1:1", (768, 768)), ("3:4", (768, 1024)), ("9:16", (768, 1344)),
])
def test_expected_canvas_matches_the_served_canvases(aspect, canvas):
    # measured on quad1 2026-09-05 (worker log 't2va WxH' lines) for all six published ratios
    assert seq.expected_canvas(aspect) == canvas


@pytest.mark.parametrize(("aspect", "seconds", "rung"), [
    ("16:9", 4, 44032), ("16:9", 5, 44032), ("16:9", 6, 61440), ("16:9", 8, 61440), ("16:9", 9, 86016),
    ("16:9", 11, 86016), ("16:9", 12, 118784), ("16:9", 15, 118784),
    ("4:3", 5, 31744), ("1:1", 5, 22528), ("1:1", 15, 86016), ("21:9", 15, 118784),
])
def test_expected_rung_matches_the_pipeline_log(aspect, seconds, rung):
    # 'packed sequence N -> bucket R' lines from the 2026-09-05 sweep (t2va, ~37-token prompt)
    assert seq.expected_rung(aspect, seconds) == rung


def test_fl2va_one_keyframe_adds_a_canvas_of_rows_and_tokens():
    t2va = seq.packed_length_estimate("16:9", 5)
    fl2va = seq.packed_length_estimate("16:9", 5, task="fl2va", keyframes=1)
    assert fl2va - t2va == 2 * 1008 + 4            # cond rows + vision tokens (+ sentinels)
    assert seq.expected_rung("16:9", 11, task="fl2va", keyframes=1) == 86016   # 85620 measured


def test_spec_bodies():
    a = live.Assets(img="I", vid="V", aud="A", key_first="F", key_last="L")
    t = seq.Spec(task="t2va", aspect="9:16", seconds=9).body(a)
    assert t == {"prompt": live.PROMPT, "seed": 7, "aspect_ratio": "9:16", "duration_seconds": 9}
    f = seq.Spec(task="fl2va", aspect="4:3", seconds=5, keyframes=(0, -1)).body(a)
    assert f["image_prompts"] == [{"image": "F", "frame_pos": 0}, {"image": "L", "frame_pos": -1}]
    assert f["aspect_ratio"] == "4:3", "the runner derives the canvas from aspect_ratio, so fl2va must send it too"
    assert seq.Spec(task="fl2va", keyframes=(0,)).tag == "fl2va 16:9 5s+first"


def test_result_row_has_the_report_columns():
    r = seq.Result(spec=seq.Spec(), status="completed", wall_s=21.0, mp4_bytes=3, sha256="x")
    row = r.row()
    for k in ("combo", "task", "request", "status", "wall_s", "mp4_bytes", "rung", "captured", "content_ok"):
        assert k in row
