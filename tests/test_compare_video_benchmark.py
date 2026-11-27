# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the side by side table of two video benchmark reports."""

from __future__ import annotations

import json

import pytest

from scripts import compare_video_benchmark as cmp


def _report(**benchmarks):
    return {"sections": [{"kind": "benchmarks", "data": {"Benchmarks": benchmarks}}]}


def test_pick_follows_a_path_and_ignores_what_is_not_a_number():
    block = {"phases": {"submit_s": 0.2}, "name": "x"}
    assert cmp.pick(block, ("phases", "submit_s")) == 0.2
    assert cmp.pick(block, ("phases", "missing")) is None
    assert cmp.pick(block, ("name",)) is None


def test_change_is_relative_to_the_first_value():
    assert cmp.change(100.0, 102.0) == "+2.0 %"
    assert cmp.change(100.0, 98.0) == "-2.0 %"
    assert cmp.change(None, 1.0) == "-"
    assert cmp.change(0, 1.0) == "-"


def test_a_row_only_one_report_has_is_still_listed():
    python = _report(inference_steps_per_second=0.31)
    cpp = _report(
        inference_steps_per_second=0.32,
        server_side={"denoise_s": {"n": 2, "mean": 100.0}},
    )
    text = cmp.table(
        cmp.benchmarks_of(python), cmp.benchmarks_of(cpp), ["python", "cpp"], ["a", "b"]
    )
    assert "| steps per second | 0.31 | 0.32 | +3.2 % |" in text
    assert "| server: denoise s | - | 100.0 | - |" in text
    assert "burst" not in text  # neither report has it


def test_the_command_line_prints_the_table(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_report(inference_steps_per_second=0.3)))
    b.write_text(json.dumps(_report(inference_steps_per_second=0.33)))
    assert cmp.main([str(a), str(b), "--labels", "python", "cpp"]) == 0
    assert "| steps per second | 0.3 | 0.33 | +10.0 % |" in capsys.readouterr().out


def test_a_report_without_benchmarks_is_refused(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"sections": []}))
    with pytest.raises(SystemExit) as exc:
        raise SystemExit(cmp.main([str(a), str(a)]))
    assert exc.value.code == 1


def test_an_older_report_with_ttft_is_compared_with_a_newer_one():
    old = _report(ttft=127.9)
    new = _report(end_to_end_latency_s=125.5)
    text = cmp.table(
        cmp.benchmarks_of(old), cmp.benchmarks_of(new), ["python", "cpp"], ["a", "b"]
    )
    assert "| whole request s | 127.9 | 125.5 | -1.9 % |" in text
