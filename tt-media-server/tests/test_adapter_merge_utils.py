# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the adapter-merge subprocess launcher and its CLI."""

import sys
from types import SimpleNamespace

import pytest

import utils.adapter_merge_utils as amu


def _fake_run(store, returncode=0, stdout="", stderr=""):
    """Stub for `subprocess.run`: records the call and returns a fixed result."""

    def _run(cmd, *, cwd, env, capture_output, text):
        store["cmd"], store["cwd"], store["env"] = cmd, cwd, env
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return _run


def test_run_merge_subprocess_builds_command_and_isolates_env(monkeypatch):
    store = {}
    monkeypatch.setattr(amu.subprocess, "run", _fake_run(store))
    monkeypatch.setenv("ADAPTER_MERGE_PYTHON", "/merge-venv/bin/python")
    monkeypatch.setenv("TT_METAL_HOME", "/forge/tt-metal")  # must be dropped
    monkeypatch.setenv("HF_HOME", "/hf/cache")  # must be preserved

    amu.run_merge_subprocess("base/model", "/adapters/a", "/out")

    assert store["cmd"] == [
        "/merge-venv/bin/python",
        "-m",
        "utils.adapter_merge_utils",
        "--base-model",
        "base/model",
        "--adapter-path",
        "/adapters/a",
        "--output-dir",
        "/out",
        "--dtype",
        "torch.bfloat16",
    ]
    # The child gets the app dir on PYTHONPATH and the forge tt-metal path dropped,
    # while unrelated env (HF cache/token) is preserved.
    assert store["cwd"] == amu._app_root()
    assert store["env"]["PYTHONPATH"] == amu._app_root()
    assert "TT_METAL_HOME" not in store["env"]
    assert store["env"]["HF_HOME"] == "/hf/cache"


def test_run_merge_subprocess_respects_explicit_python_and_cwd(monkeypatch):
    store = {}
    monkeypatch.setattr(amu.subprocess, "run", _fake_run(store))

    amu.run_merge_subprocess(
        "b",
        "a",
        "o",
        python_executable="/x/py",
        cwd="/work",
        dtype_str="torch.float16",
    )

    assert store["cmd"][0] == "/x/py"
    assert store["cmd"][-2:] == ["--dtype", "torch.float16"]
    assert store["cwd"] == "/work"
    assert store["env"]["PYTHONPATH"] == "/work"


def test_run_merge_subprocess_raises_with_output_on_failure(monkeypatch):
    monkeypatch.setattr(
        amu.subprocess, "run", _fake_run({}, returncode=1, stdout="OUT", stderr="ERR")
    )
    with pytest.raises(RuntimeError) as exc:
        amu.run_merge_subprocess("b", "a", "o", python_executable="py", cwd="/w")
    message = str(exc.value)
    assert "exit 1" in message
    assert "OUT" in message and "ERR" in message


def test_cli_flags_from_launcher_parse_in_main(monkeypatch):
    """The flags run_merge_subprocess emits must be exactly what main() accepts."""
    store = {}
    monkeypatch.setattr(amu.subprocess, "run", _fake_run(store))
    amu.run_merge_subprocess(
        "base/model",
        "/adapters/a",
        "/out",
        python_executable="py",
        cwd="/w",
        dtype_str="torch.float32",
    )
    launcher_flags = store["cmd"][3:]  # drop [py, -m, utils.adapter_merge_utils]

    called = {}

    def fake_merge_adapter(
        base, adapter, output, dtype_str="torch.bfloat16", verify_load=True
    ):
        called.update(
            base=base,
            adapter=adapter,
            output=output,
            dtype=dtype_str,
            verify=verify_load,
        )

    monkeypatch.setattr(amu, "merge_adapter", fake_merge_adapter)
    monkeypatch.setattr(sys, "argv", ["utils.adapter_merge_utils", *launcher_flags])
    amu.main()

    assert called == {
        "base": "base/model",
        "adapter": "/adapters/a",
        "output": "/out",
        "dtype": "torch.float32",
        "verify": True,
    }


def test_cli_no_verify_load_flag(monkeypatch):
    called = {}
    monkeypatch.setattr(
        amu,
        "merge_adapter",
        lambda *args, verify_load=True, **kw: called.update(verify=verify_load),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--base-model",
            "b",
            "--adapter-path",
            "a",
            "--output-dir",
            "o",
            "--no-verify-load",
        ],
    )
    amu.main()

    assert called["verify"] is False
