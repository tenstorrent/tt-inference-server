# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""TT_AGENTIC_TASK_NAMES overrides the configured agentic task selection."""

from types import SimpleNamespace

from llm_module.drivers.agentic import TASK_NAMES_ENV, resolve_task_names


def _task(task_names, task_names_map=None):
    cfg = SimpleNamespace(task_names=task_names, task_names_map=task_names_map or {})
    return SimpleNamespace(agentic_eval_config=cfg)


def test_default_returns_configured_names(monkeypatch):
    monkeypatch.delenv(TASK_NAMES_ENV, raising=False)
    assert resolve_task_names(_task(["a__*"])) == ["a__*"]


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(TASK_NAMES_ENV, "django__*, sympy__sympy-13551 ,")
    assert resolve_task_names(_task(["a__*"])) == ["django__*", "sympy__sympy-13551"]


def test_blank_env_is_ignored(monkeypatch):
    monkeypatch.setenv(TASK_NAMES_ENV, "  ")
    assert resolve_task_names(_task(["a__*"])) == ["a__*"]


def test_no_agentic_config(monkeypatch):
    monkeypatch.setenv(TASK_NAMES_ENV, "x")
    assert resolve_task_names(SimpleNamespace(agentic_eval_config=None)) == []


from llm_module.drivers.agentic import N_CONCURRENT_ENV, resolve_n_concurrent_trials


def test_n_concurrent_default(monkeypatch):
    monkeypatch.delenv(N_CONCURRENT_ENV, raising=False)
    assert resolve_n_concurrent_trials(SimpleNamespace(n_concurrent_trials=16)) == 16


def test_n_concurrent_override(monkeypatch):
    monkeypatch.setenv(N_CONCURRENT_ENV, "8")
    assert resolve_n_concurrent_trials(SimpleNamespace(n_concurrent_trials=16)) == 8


def test_n_concurrent_invalid(monkeypatch):
    import pytest
    monkeypatch.setenv(N_CONCURRENT_ENV, "0")
    with pytest.raises(ValueError):
        resolve_n_concurrent_trials(SimpleNamespace(n_concurrent_trials=16))
