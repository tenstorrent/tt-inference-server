# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""NL2Repo-Bench task generation and grader hardening (reward hacks)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_DIR = (
    Path(__file__).resolve().parents[3]
    / "reference_config" / "evals" / "harbor_tasks" / "nl2repobench"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grade = _load("nl2repo_grade", _DIR / "task_files" / "grade.py")
gen = _load("nl2repo_gen_tasks", _DIR / "gen_tasks.py")


def test_counts_only_the_final_summary():
    log = (
        "tests/test_a.py ..F\n"
        "500 passed\n"
        "=========== 1 failed, 2 passed, 1 warning in 0.12s ===========\n"
    )
    result = grade.parse_pytest(log, exit_code=1)
    assert (result["passed"], result["failed"], result["fabricated"]) == (2, 1, False)
    assert grade.upstream_count(log) == 502


def test_quiet_and_legacy_summaries():
    assert grade.parse_pytest("3 passed, 2 errors in 1.50s (0:00:01)\n", 1)["passed"] == 3
    assert grade.parse_pytest("==== 4 passed in 0.12 seconds ====\n", 0)["passed"] == 4
    assert grade.parse_pytest("\x1b[32m==== 4 passed in 0.1s ====\x1b[0m\n", 0)["passed"] == 4
    assert grade.parse_pytest("==== no tests ran in 0.01s ====\n", 5)["passed"] == 0


def test_clean_summary_with_failing_exit_is_fabricated():
    # e.g. an agent conftest hook printing its own summary after a crash
    result = grade.parse_pytest("ERROR collecting\n==== 27 passed in 0.10s ====\n", 2)
    assert result["fabricated"] and result["passed"] == 0


def test_no_summary_counts_nothing():
    assert grade.parse_pytest("ImportError: no module named x\n", 4)["passed"] == 0


def test_test_paths_are_recognized():
    for rel in ("conftest.py", "a/conftest.py", "tests/helpers.py", "test_tzinfo.py",
                "pkg/foo_test.py", "test/__init__.py"):
        assert grade._is_test_path(Path(rel)), rel
    for rel in ("emoji/unicode_codes/__init__.py", "pkg/core.py", "utils/x.py"):
        assert not grade._is_test_path(Path(rel)), rel


def test_shadowing_modules_are_found_but_targets_kept(tmp_path):
    (tmp_path / "_pytest").mkdir()
    (tmp_path / "json.py").write_text("")
    (tmp_path / "pytest.py").write_text("")
    (tmp_path / "mypy.ini").write_text("")
    (tmp_path / "_pytest.cpython-312-x86_64-linux-gnu.so").write_text("")
    (tmp_path / "my_project_xyz").mkdir()
    hits = {p.name for p in grade._shadowed_names(tmp_path, keep={"json"})}
    assert hits == {"_pytest", "pytest.py", "_pytest.cpython-312-x86_64-linux-gnu.so"}


def test_bare_pytest_commands_get_hidden_test_paths():
    commands = ["pip install -e .", "pip install pytest", "pytest --continue-on-collection-errors -v"]
    assert gen.pin_test_paths("pytz", commands)[2] == (
        "pytest --continue-on-collection-errors -v test_docs.py test_lazy.py test_tzinfo.py"
    )
    assert gen.pin_test_paths("tinydb", ["pytest tests"]) == ["pytest tests"]
    with pytest.raises(SystemExit):
        gen.pin_test_paths("tinydb", ["pytest --continue-on-collection-errors"])
    with pytest.raises(SystemExit):
        gen.pin_test_paths("pytz", ["pytest tests"])


def test_lfs_pointer_spec_is_rejected(tmp_path):
    pointer = tmp_path / "start.md"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 41234\n"
    )
    with pytest.raises(SystemExit):
        gen.read_spec(pointer)
    pointer.write_text("# Spec\n" + "text " * 400)
    assert gen.read_spec(pointer).startswith(b"# Spec")


def test_target_modules_are_importable_spellings():
    assert gen.target_modules(["pytorch-grad-cam", "PyYAML", "sklearn"]) == [
        "PyYAML", "pytorch_grad_cam", "pyyaml", "sklearn",
    ]
