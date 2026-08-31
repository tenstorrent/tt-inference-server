# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHERS = _REPO_ROOT / "launchers"


def _load_launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(_LAUNCHERS))
    spec = importlib.util.spec_from_file_location(
        "run_agentic_under_test", _LAUNCHERS / "run_agentic.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_agentic_launcher_bootstraps_uv_before_venv_setup(
    monkeypatch, tmp_path
):
    """A cold external-plan launch must not assume run.py already bootstrapped uv."""
    launcher = _load_launcher(monkeypatch)
    bootstrap_uv = tmp_path / ".workflow_venvs/.venv_bootstrap_uv/bin/uv"
    events: list[str] = []

    def fake_bootstrap_uv():
        events.append("bootstrap")
        bootstrap_uv.parent.mkdir(parents=True)
        bootstrap_uv.write_text("system uv installed here", encoding="utf-8")

    def fake_setup_venv_and_exec(venv_type, logger, label, model_spec=None):
        del venv_type, logger, label, model_spec
        events.append("setup")
        assert bootstrap_uv.is_file()
        return 0

    monkeypatch.setattr("workflows.bootstrap_uv.bootstrap_uv", fake_bootstrap_uv)
    monkeypatch.setattr(
        "workflows.model_spec.get_runtime_model_spec",
        lambda **kwargs: (SimpleNamespace(model_name=kwargs["model"]), None, None),
    )
    monkeypatch.setattr(launcher, "setup_venv_and_exec", fake_setup_venv_and_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agentic.py",
            "--model",
            "gpt-oss-120b",
            "--workflow",
            "agentic",
            "--device",
            "p300x2",
        ],
    )

    assert launcher.main() == 0
    assert events == ["bootstrap", "setup"]
