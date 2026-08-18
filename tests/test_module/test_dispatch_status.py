# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the status helpers + media-task wiring in ``test_module.dispatch``.

Covers the spec-test exit-code fix and its extension to evals/benchmarks:
  - ``_error_block`` / ``_media_outcome_block``: a crash or intentional skip
    still becomes a *visible* Block (closes the report/exit-code gap).
  - ``_block_status``: resolves a Block's TestStatus with a legacy fallback.
  - ``run_media_task``: skip -> non-blocking (rc=0), crash -> visible ERROR
    Block (rc=1).
"""

from __future__ import annotations

from types import SimpleNamespace

from report_module.schema import Block
from report_module.status import TestStatus
from test_module import dispatch
from test_module._test_common import NotApplicable, SkipTest
from test_module.dispatch import (
    _block_status,
    _error_block,
    _media_outcome_block,
    run_media_task,
)
from test_module.task_types import MediaTaskType
from workflows.workflow_types import ModelType


def _fake_ctx(model="speecht5_tts", device="N150", model_type=ModelType.AUDIO):
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name=model, model_type=model_type),
        device=SimpleNamespace(name=device),
    )


# --- spec-test helpers ----------------------------------------------------


def test_error_block_is_visible_error_block():
    case = {"name": "TestTTSServerHealth", "description": "load test", "targets": {}}
    block = _error_block(case, _fake_ctx(), AttributeError("no attribute"))
    assert block.kind == "spec_tests"
    assert block.data["status"] == TestStatus.ERROR.value
    assert block.data["success"] is False
    assert block.data["error"]["type"] == "AttributeError"
    assert block.data["error"]["message"] == "no attribute"
    assert block.id == "speecht5_tts_N150"


def test_block_status_prefers_explicit_status():
    block = Block(kind="spec_tests", data={"status": "skip", "success": False})
    assert _block_status(block) is TestStatus.SKIP


def test_block_status_falls_back_to_success_flag():
    assert _block_status(Block(kind="spec_tests", data={"success": True})) is (
        TestStatus.PASS
    )
    assert _block_status(Block(kind="spec_tests", data={"success": False})) is (
        TestStatus.FAIL
    )


def test_block_status_falls_back_to_skipped_flag():
    block = Block(kind="spec_tests", data={"success": False, "skipped": True})
    assert _block_status(block) is TestStatus.SKIP


# --- media outcome block (evals / benchmarks) -----------------------------


def test_media_outcome_block_kind_follows_task_type():
    skip = _media_outcome_block(
        _fake_ctx(), MediaTaskType.BENCHMARK, TestStatus.SKIP, reason="no board"
    )
    assert skip.kind == "benchmarks"
    assert skip.data["status"] == TestStatus.SKIP.value
    assert skip.data["reason"] == "no board"

    err = _media_outcome_block(
        _fake_ctx(), MediaTaskType.EVALUATION, TestStatus.ERROR, exc=RuntimeError("x")
    )
    assert err.kind == "evals"
    assert err.data["status"] == TestStatus.ERROR.value
    assert err.data["error"]["message"] == "x"


# --- run_media_task -------------------------------------------------------


def _patch_runner(monkeypatch, task_type, runner):

    monkeypatch.setattr(dispatch, "_resolve_runner", lambda tt, name: runner)
    monkeypatch.setattr(dispatch, "accept_blocks", lambda *a, **k: True)


def test_run_media_task_skip_is_non_blocking(monkeypatch):
    def runner(ctx):
        raise SkipTest("feature not enabled for this device")

    _patch_runner(monkeypatch, MediaTaskType.BENCHMARK, runner)
    rc, block = run_media_task(_fake_ctx(), MediaTaskType.BENCHMARK)
    assert rc == 0
    assert block.kind == "benchmarks"
    assert block.data["status"] == TestStatus.SKIP.value


def test_run_media_task_not_applicable_is_non_blocking(monkeypatch):
    def runner(ctx):
        raise NotApplicable("no reference dataset")

    _patch_runner(monkeypatch, MediaTaskType.EVALUATION, runner)
    rc, block = run_media_task(_fake_ctx(), MediaTaskType.EVALUATION)
    assert rc == 0
    assert block.data["status"] == TestStatus.NA.value


def test_run_media_task_crash_emits_visible_error_block(monkeypatch):
    def runner(ctx):
        raise ValueError("kaboom")

    _patch_runner(monkeypatch, MediaTaskType.EVALUATION, runner)
    rc, block = run_media_task(_fake_ctx(), MediaTaskType.EVALUATION)
    assert rc == 1
    assert block is not None and block.kind == "evals"
    assert block.data["status"] == TestStatus.ERROR.value
    assert block.data["error"]["message"] == "kaboom"


def test_run_media_task_no_runner_by_design_is_skip(monkeypatch):
    # A model type that is runnerless *by design* (_NO_MEDIA_RUNNER): previously
    # (1, None) — a silent failure. Now a non-blocking, visible SKIP block.
    monkeypatch.setattr(dispatch, "EVAL_DISPATCH", {})
    monkeypatch.setattr(dispatch, "accept_blocks", lambda *a, **k: True)
    rc, block = run_media_task(
        _fake_ctx(model_type=ModelType.VLM), MediaTaskType.EVALUATION
    )
    assert rc == 0
    assert block is not None and block.kind == "evals"
    assert block.data["status"] == TestStatus.SKIP.value
    assert "unsupported by design" in block.data["reason"]


def test_run_media_task_missing_expected_runner_is_error(monkeypatch):
    # A model type that *should* have a runner but doesn't: a real registration
    # bug -> blocking, visible ERROR block (not a silent (1, None)).
    monkeypatch.setattr(dispatch, "EVAL_DISPATCH", {})
    monkeypatch.setattr(dispatch, "accept_blocks", lambda *a, **k: True)
    rc, block = run_media_task(
        _fake_ctx(model_type=ModelType.AUDIO), MediaTaskType.EVALUATION
    )
    assert rc == 1
    assert block is not None and block.kind == "evals"
    assert block.data["status"] == TestStatus.ERROR.value
    assert "no evaluation runner registered" in block.data["error"]["message"]


def test_run_media_task_success_passthrough(monkeypatch):
    good = Block(kind="benchmarks", data={"target_checks": {}})

    def runner(ctx):
        return good

    _patch_runner(monkeypatch, MediaTaskType.BENCHMARK, runner)
    rc, block = run_media_task(_fake_ctx(), MediaTaskType.BENCHMARK)
    assert rc == 0 and block is good
