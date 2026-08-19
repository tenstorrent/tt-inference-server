# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for scripts/derive_merge_versions.py."""

import sys

import pytest

from scripts import derive_merge_versions as dmv


def _write_pyproject(tmp_path, deps):
    # Single-quoted TOML literals so deps containing '"' (env markers) stay valid.
    body = "[project]\ndependencies = [\n" + "".join(f"    '{d}',\n" for d in deps) + "]\n"
    path = tmp_path / "pyproject.toml"
    path.write_text(body)
    return str(path)


def test_transformers_spec_returns_pinned_dependency(tmp_path):
    path = _write_pyproject(tmp_path, ["torch>=2.7.1,<2.8", "transformers==4.55.0"])
    assert dmv.transformers_spec(path) == "transformers==4.55.0"


def test_transformers_spec_preserves_extras_and_markers(tmp_path):
    spec = 'transformers[torch]>=4.5,<5 ; python_version>="3.10"'
    path = _write_pyproject(tmp_path, [spec])
    assert dmv.transformers_spec(path) == spec


def test_transformers_spec_raises_when_absent(tmp_path):
    path = _write_pyproject(tmp_path, ["torch", "numpy"])
    with pytest.raises(ValueError, match="transformers"):
        dmv.transformers_spec(path)


def test_transformers_spec_raises_when_no_dependencies(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("[project]\nname = 'x'\n")
    with pytest.raises(ValueError):
        dmv.transformers_spec(str(path))


def test_installed_version_reads_real_package():
    version = dmv.installed_version(sys.executable, "pip")
    assert version and version[0].isdigit()


def test_installed_version_missing_package_raises():
    with pytest.raises(RuntimeError, match="version"):
        dmv.installed_version(sys.executable, "definitely-not-a-real-package-xyz")


def test_installed_version_missing_interpreter_raises():
    with pytest.raises(RuntimeError, match="Interpreter not found"):
        dmv.installed_version("/nonexistent/python", "pip")


def test_peft_spec_formats_installed_version(monkeypatch):
    monkeypatch.setattr(dmv, "installed_version", lambda py, pkg: "0.20.0")
    assert dmv.peft_spec("/whatever/python") == "peft==0.20.0"


def test_main_transformers_prints_spec(tmp_path, capsys):
    path = _write_pyproject(tmp_path, ["transformers==4.55.0"])
    dmv.main(["transformers", path])
    assert capsys.readouterr().out.strip() == "transformers==4.55.0"


def test_main_peft_prints_spec(monkeypatch, capsys):
    monkeypatch.setattr(dmv, "installed_version", lambda py, pkg: "0.20.0")
    dmv.main(["peft", "/whatever/python"])
    assert capsys.readouterr().out.strip() == "peft==0.20.0"
