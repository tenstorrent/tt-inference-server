# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Exit-code semantics for ``run_spec_tests``.

The contract: rc reflects *failures*, not *emptiness*. A model with no
matching suite, or a matched suite whose cases are all skipped, is a clean
no-op (rc=0, no block) — not a spurious rc=1. Real test failures (a raised
exception or a block that didn't report ``success=True``) drive rc=1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from report_module.schema import Block
from test_module import dispatch
from test_module.task_types import MediaTaskType


class _FakeTest:
    def __init__(self, block: Block):
        self._block = block

    def run_tests(self) -> Block:
        return self._block


def _ctx():
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name="speecht5_tts"),
        device=SimpleNamespace(name="N150"),
        spec_tests_num_prompts_cap=None,
    )


@pytest.fixture(autouse=True)
def _no_accumulator(monkeypatch):
    # Isolate from the sweep-level block accumulator side effect.
    monkeypatch.setattr(dispatch, "accept_blocks", lambda *a, **k: None)


def _suite(*cases: dict) -> dict:
    return {"id": "suite-1", "test_cases": list(cases)}


class TestResolveRunner:
    """The dispatch tables now hold function *names*, resolved lazily via the
    package __getattr__ so importing dispatch doesn't drag in every runner's
    optional deps. Only TTS is exercised here (image evals would need
    open_clip)."""

    def test_resolves_tts_eval_by_name(self):
        runner = dispatch._resolve_runner(MediaTaskType.EVALUATION, "TEXT_TO_SPEECH")
        assert callable(runner) and runner.__name__ == "run_tts_eval"

    def test_resolves_tts_benchmark_by_name(self):
        runner = dispatch._resolve_runner(MediaTaskType.BENCHMARK, "TEXT_TO_SPEECH")
        assert callable(runner) and runner.__name__ == "run_tts_benchmark"

    def test_unknown_model_type_returns_none(self):
        assert dispatch._resolve_runner(MediaTaskType.EVALUATION, "NOPE") is None


def test_empty_filter_is_rc0_noop(monkeypatch):
    monkeypatch.setattr(dispatch, "_resolve_spec_test_suites", lambda ctx: [])
    assert dispatch.run_spec_tests(_ctx()) == (0, None)


def test_matched_suite_all_disabled_is_rc0_noop(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "_resolve_spec_test_suites",
        lambda ctx: [
            _suite(
                {"name": "A", "module": "m", "enabled": False},
                {"name": "B", "module": "m", "enabled": False},
            )
        ],
    )
    # No case is instantiated, so a failure here would mean the guard leaked.
    monkeypatch.setattr(
        dispatch,
        "_instantiate_spec_test",
        lambda *a, **k: pytest.fail("disabled cases must not be instantiated"),
    )
    assert dispatch.run_spec_tests(_ctx()) == (0, None)


def test_matched_suite_all_malformed_is_rc0_noop(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "_resolve_spec_test_suites",
        lambda ctx: [_suite({"description": "missing name/module"})],
    )
    assert dispatch.run_spec_tests(_ctx()) == (0, None)


def test_passing_case_is_rc0_with_block(monkeypatch):
    block = Block(kind="spec_tests", title="A", data={"success": True})
    monkeypatch.setattr(
        dispatch,
        "_resolve_spec_test_suites",
        lambda ctx: [_suite({"name": "A", "module": "m", "enabled": True})],
    )
    monkeypatch.setattr(
        dispatch, "_instantiate_spec_test", lambda case, ctx: _FakeTest(block)
    )
    exit_code, returned = dispatch.run_spec_tests(_ctx())
    assert exit_code == 0
    assert returned is block


def test_block_without_success_is_rc1(monkeypatch):
    block = Block(kind="spec_tests", title="A", data={"success": False})
    monkeypatch.setattr(
        dispatch,
        "_resolve_spec_test_suites",
        lambda ctx: [_suite({"name": "A", "module": "m", "enabled": True})],
    )
    monkeypatch.setattr(
        dispatch, "_instantiate_spec_test", lambda case, ctx: _FakeTest(block)
    )
    exit_code, _ = dispatch.run_spec_tests(_ctx())
    assert exit_code == 1


def test_raising_case_is_rc1(monkeypatch):
    def _boom(case, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        dispatch,
        "_resolve_spec_test_suites",
        lambda ctx: [_suite({"name": "A", "module": "m", "enabled": True})],
    )
    monkeypatch.setattr(dispatch, "_instantiate_spec_test", _boom)
    # A case that raises is a real failure -> rc=1, and it emits a *visible*
    # error block
    rc, block = dispatch.run_spec_tests(_ctx())
    assert rc == 1
    assert block is not None
    assert block.data["status"] == "error"
    assert block.data["error"]["message"] == "kaboom"
