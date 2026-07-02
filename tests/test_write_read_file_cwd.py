"""Tests that write_file and read_file resolve relative paths against cwd."""
import os
import pytest
from orchestrator.tools import write_file, read_file, execute


def test_write_file_relative_path_uses_cwd(tmp_path):
    write_file("subdir/hello.txt", "content", cwd=str(tmp_path))
    assert (tmp_path / "subdir" / "hello.txt").read_text() == "content"


def test_write_file_absolute_path_ignores_cwd(tmp_path):
    target = tmp_path / "abs.txt"
    write_file(str(target), "abs-content", cwd="/some/other/dir")
    assert target.read_text() == "abs-content"


def test_write_file_no_cwd_uses_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("no_cwd.txt", "data")
    assert (tmp_path / "no_cwd.txt").read_text() == "data"


def test_read_file_relative_path_uses_cwd(tmp_path):
    (tmp_path / "greet.txt").write_text("hello")
    result = read_file("greet.txt", cwd=str(tmp_path))
    assert result == "hello"


def test_read_file_absolute_path_ignores_cwd(tmp_path):
    target = tmp_path / "abs_read.txt"
    target.write_text("world")
    result = read_file(str(target), cwd="/some/other/dir")
    assert result == "world"


def test_execute_write_file_passes_cwd(tmp_path):
    result = execute("write_file", {"path": "out.txt", "content": "via execute"}, cwd=str(tmp_path))
    assert "wrote" in result
    assert (tmp_path / "out.txt").read_text() == "via execute"


def test_execute_read_file_passes_cwd(tmp_path):
    (tmp_path / "in.txt").write_text("from execute")
    result = execute("read_file", {"path": "in.txt"}, cwd=str(tmp_path))
    assert result == "from execute"
