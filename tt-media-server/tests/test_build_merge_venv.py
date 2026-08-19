# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for scripts/build_merge_venv.py."""

import os
import sys

import pytest

from scripts import build_merge_venv as bmv


def _write_pyproject(tmp_path, deps):
    # Single-quoted TOML literals so deps containing '"' (env markers) stay valid.
    body = (
        "[project]\ndependencies = [\n" + "".join(f"    '{d}',\n" for d in deps) + "]\n"
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(body)
    return str(path)


def test_transformers_spec_returns_pinned_dependency(tmp_path):
    path = _write_pyproject(tmp_path, ["torch>=2.7.1,<2.8", "transformers==4.55.0"])
    assert bmv.transformers_spec(path) == "transformers==4.55.0"


def test_transformers_spec_preserves_extras_and_markers(tmp_path):
    spec = 'transformers[torch]>=4.5,<5 ; python_version>="3.10"'
    path = _write_pyproject(tmp_path, [spec])
    assert bmv.transformers_spec(path) == spec


def test_transformers_spec_raises_when_absent(tmp_path):
    path = _write_pyproject(tmp_path, ["torch", "numpy"])
    with pytest.raises(ValueError, match="transformers"):
        bmv.transformers_spec(path)


def test_transformers_spec_raises_when_no_dependencies(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("[project]\nname = 'x'\n")
    with pytest.raises(ValueError):
        bmv.transformers_spec(str(path))


def test_peft_spec_pins_reported_version(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "0.20.0\n"
        stderr = ""

    monkeypatch.setattr(bmv.subprocess, "run", lambda *a, **k: _Result())
    assert bmv.peft_spec("/any/python") == "peft==0.20.0"


def test_peft_spec_missing_version_raises(monkeypatch):
    # An interpreter without peft (non-zero exit) surfaces a clear error.
    class _Result:
        returncode = 1
        stdout = ""
        stderr = "No package metadata was found for peft"

    monkeypatch.setattr(bmv.subprocess, "run", lambda *a, **k: _Result())
    with pytest.raises(RuntimeError, match="Could not read 'peft'"):
        bmv.peft_spec("/any/python")


def test_peft_spec_missing_interpreter_raises():
    with pytest.raises(RuntimeError, match="Interpreter not found"):
        bmv.peft_spec("/nonexistent/python")


def test_peft_spec_rejects_option_like_interpreter():
    with pytest.raises(ValueError, match="option-like interpreter"):
        bmv.peft_spec("--foo")


@pytest.mark.parametrize(
    "spec",
    [
        "transformers==4.55.0",
        "peft==0.20.0",
        "transformers[torch]==4.55.0",
        "transformers>=4.5,<5",
    ],
)
def test_validated_spec_accepts_plain_requirements(spec):
    assert bmv._validated_spec(spec) == spec


@pytest.mark.parametrize(
    "spec",
    [
        "--index-url=http://evil",  # pip option injection
        "transformers==4.55.0; rm -rf /",  # shell metacharacters
        "transformers 4.55.0",  # whitespace
        'transformers=="4.55.0"',  # quotes
        "transformers",  # unpinned (no specifier)
    ],
)
def test_validated_spec_rejects_unsafe_specs(spec):
    with pytest.raises(ValueError, match="unrecognized package spec"):
        bmv._validated_spec(spec)


@pytest.mark.parametrize("bad", ["-rf", "--foo", ""])
def test_validated_path_rejects_option_like(bad):
    with pytest.raises(ValueError, match="option-like"):
        bmv._validated_path(bad, "venv dir")


def test_build_venv_rejects_option_like_venv_dir(monkeypatch):
    monkeypatch.setattr(bmv.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="option-like venv dir"):
        bmv.build_venv("--evil", ["transformers==4.55.0"], requirements="/reqs.txt")


def test_build_venv_rejects_unsafe_spec(monkeypatch):
    monkeypatch.setattr(bmv.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="unrecognized package spec"):
        bmv.build_venv("/tmp/merge-venv", ["--index-url=http://evil"], requirements="/reqs.txt")


def test_default_forge_python_uses_python_env_dir(monkeypatch):
    monkeypatch.setenv("PYTHON_ENV_DIR", "/opt/venv-worker")
    assert bmv.default_forge_python() == "/opt/venv-worker/bin/python"


def test_default_forge_python_falls_back_next_to_script(monkeypatch):
    monkeypatch.delenv("PYTHON_ENV_DIR", raising=False)
    expected = os.path.join(bmv.SERVER_DIR, "venv-worker", "bin", "python")
    assert bmv.default_forge_python() == expected


def test_build_venv_runs_expected_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(bmv.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    bmv.build_venv(
        "/tmp/merge-venv",
        ["transformers==4.55.0", "peft==0.20.0"],
        requirements="/reqs.txt",
    )

    assert calls[0] == [sys.executable, "-m", "venv", "/tmp/merge-venv"]
    pip = "/tmp/merge-venv/bin/pip"
    assert calls[1] == [pip, "install", "--no-cache-dir", "--upgrade", "pip"]
    assert calls[2] == [
        pip,
        "install",
        "--no-cache-dir",
        "-r",
        "/reqs.txt",
        "transformers==4.55.0",
        "peft==0.20.0",
    ]


def test_main_derives_specs_and_builds(tmp_path, monkeypatch):
    pyproject = _write_pyproject(tmp_path, ["transformers==4.55.0"])
    monkeypatch.setattr(bmv, "peft_spec", lambda forge_python: "peft==0.20.0")

    captured = {}
    monkeypatch.setattr(
        bmv,
        "build_venv",
        lambda venv_dir, specs, requirements: captured.update(
            venv_dir=venv_dir, specs=specs, requirements=requirements
        ),
    )

    bmv.main([str(tmp_path / "merge-venv"), pyproject, "--forge-python", "/fp/python"])

    assert captured["venv_dir"] == str(tmp_path / "merge-venv")
    assert captured["specs"] == ["transformers==4.55.0", "peft==0.20.0"]
