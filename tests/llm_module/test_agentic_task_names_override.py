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
