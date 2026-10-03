# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# Ported from #5234 (setuptools pin and NLTK preflight ordering).

from types import SimpleNamespace

import pytest

from workflows import workflow_venvs as wv


@pytest.mark.parametrize("cached", [False, True])
def test_meta_setuptools_is_pinned_before_install_or_cached_use(
    tmp_path, monkeypatch, cached
):
    cookbook = tmp_path / "llama-cookbook"
    meta_eval = (
        cookbook
        / "end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/work_dir_test"
    )
    if cached:
        meta_eval.mkdir(parents=True)
    calls = []

    def command(cmd, **kwargs):
        calls.append(cmd)
        if isinstance(cmd, str) and cmd.startswith("git clone"):
            meta_eval.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(wv, "run_command", command)
    monkeypatch.setattr(wv, "install_requirements", lambda *a: True)
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(venv_path=tmp_path, venv_python=tmp_path / "bin/python")
    spec = SimpleNamespace(model_type=wv.ModelType.LLM, model_name="test")
    wv.setup_evals_meta(config, spec)
    assert "setuptools>=77,<81" in str(calls[0])
    assert not any("-U pip setuptools" in str(cmd) for cmd in calls)
    scoring = next(i for i, cmd in enumerate(calls) if "setup_nltk_data.py" in str(cmd))
    assert scoring > 0
    if not cached:
        install = next(i for i, cmd in enumerate(calls) if "-e ." in str(cmd))
        assert install > 0


def test_meta_setuptools_failure_stops_before_install(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(wv, "run_command", lambda cmd, **kw: calls.append(cmd) or 1)
    config = SimpleNamespace(venv_path=tmp_path, venv_python=tmp_path / "bin/python")
    spec = SimpleNamespace(model_type=wv.ModelType.LLM, model_name="test")
    assert not wv.setup_evals_meta(config, spec)
    assert len(calls) == 1
    assert "setuptools>=77,<81" in str(calls[0])
