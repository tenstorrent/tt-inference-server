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


# -- path traversal rejection -------------------------------------------------

def test_write_file_traversal_rejected(tmp_path):
    result = write_file("../../etc/passwd", "evil", cwd=str(tmp_path))
    assert result.startswith("ERROR: path traversal outside cwd denied")
    assert not os.path.exists("/etc/passwd_evil")  # just a sanity guard


def test_read_file_traversal_rejected(tmp_path):
    result = read_file("../../etc/hostname", cwd=str(tmp_path))
    assert result.startswith("ERROR: path traversal outside cwd denied")


def test_write_file_traversal_via_nested_path_rejected(tmp_path):
    result = write_file("a/b/../../../etc/passwd", "evil", cwd=str(tmp_path))
    assert result.startswith("ERROR: path traversal outside cwd denied")


def test_write_file_subdirectory_allowed(tmp_path):
    result = write_file("a/b/c.txt", "ok", cwd=str(tmp_path))
    assert "wrote" in result
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "ok"


def test_write_file_dotdot_that_stays_inside_allowed(tmp_path):
    (tmp_path / "sub").mkdir()
    result = write_file("sub/../file.txt", "ok", cwd=str(tmp_path))
    assert "wrote" in result
    assert (tmp_path / "file.txt").read_text() == "ok"
