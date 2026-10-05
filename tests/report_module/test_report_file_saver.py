# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.report_file_saver.ReportFileSaver``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report_module.report_file_saver import ReportFileSaver


def test_write_markdown_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "out.md"
    ReportFileSaver.write_markdown("# hello", path)
    assert path.read_text(encoding="utf-8") == "# hello"


def test_write_json_round_trips(tmp_path: Path):
    path = tmp_path / "data" / "out.json"
    ReportFileSaver.write_json({"a": 1, "b": [2, 3]}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}


def test_write_json_serializes_unknown_types_via_default_str(tmp_path: Path):
    path = tmp_path / "out.json"
    ReportFileSaver.write_json({"p": Path("/tmp/x")}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"p": "/tmp/x"}


def test_non_strict_swallows_errors(tmp_path: Path):
    # Target path is a directory: writing fails, but strict=False stays quiet.
    target = tmp_path / "adir"
    target.mkdir()
    ReportFileSaver.write_markdown("x", target, strict=False)  # no raise


def test_strict_reraises_on_failure(tmp_path: Path):
    target = tmp_path / "adir"
    target.mkdir()
    with pytest.raises(Exception):
        ReportFileSaver.write_markdown("x", target, strict=True)


@pytest.mark.parametrize("kind", ["json", "markdown"])
def test_interrupted_replacement_preserves_previous_report(tmp_path, monkeypatch, kind):
    path = tmp_path / "report"
    path.write_text("previous report")

    def interrupt_replace(source, destination):
        assert source.parent == path.parent
        assert source.read_text()  # The replacement is complete before publication.
        assert destination == path
        assert path.read_text() == "previous report"
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt):
        if kind == "json":
            ReportFileSaver.write_json({"new": "report"}, path, strict=True)
        else:
            ReportFileSaver.write_markdown("new report", path, strict=True)

    assert path.read_text() == "previous report"
    assert list(tmp_path.iterdir()) == [path]
