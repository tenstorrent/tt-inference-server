# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.commands`` command wrappers.

The workflow class lookup and summary generation are patched so the
command control-flow (return-code propagation, continue-on-failure,
error handling) is tested without running a real workflow.
"""

from __future__ import annotations

from types import SimpleNamespace

from workflow_module import workflows as workflows_mod
from workflow_module.commands import CommandResult, SummaryCommand, WorkflowCommand
from workflow_module.execution import OrchestratorMetadata, WorkflowResult


class TestCommandResult:
    def test_succeeded_on_zero(self):
        assert CommandResult("c", 0).succeeded is True
        assert CommandResult("c", 1, error="x").succeeded is False


def _patch_workflow(monkeypatch, *, return_code, error=None):
    class FakeWorkflow:
        def __init__(self, ctx, orchestrator_metadata=None):
            self.ctx = ctx

        def run(self):
            return WorkflowResult("fake", return_code=return_code, error=error)

    monkeypatch.setattr(workflows_mod, "get_workflow_class", lambda name: FakeWorkflow)


def _workflow_command(**kwargs) -> WorkflowCommand:
    return WorkflowCommand(
        ctx=SimpleNamespace(),
        workflow_name="benchmarks",
        orchestrator_metadata=OrchestratorMetadata(),
        **kwargs,
    )


class TestWorkflowCommand:
    def test_propagates_success(self, monkeypatch):
        _patch_workflow(monkeypatch, return_code=0)
        result = _workflow_command().execute()
        assert result.return_code == 0
        assert result.succeeded is True
        assert isinstance(result.payload, WorkflowResult)

    def test_propagates_failure(self, monkeypatch):
        _patch_workflow(monkeypatch, return_code=1, error="boom")
        result = _workflow_command().execute()
        assert result.return_code == 1
        assert result.error == "boom"

    def test_continue_on_failure_masks_nonzero_return(self, monkeypatch):
        _patch_workflow(monkeypatch, return_code=1, error="boom")
        result = _workflow_command(continue_on_failure=True).execute()
        # rc is forced to 0 so a --repeat sweep keeps going...
        assert result.return_code == 0
        # ...but the original error is still surfaced for logging.
        assert result.error == "boom"


class TestSummaryCommand:
    def test_success_returns_payload(self, monkeypatch, tmp_path):
        import workflow_module.summary_report as sr

        fake_result = SimpleNamespace(markdown_path=tmp_path / "summary.md")
        monkeypatch.setattr(sr, "summarize_container", lambda c, o: fake_result)
        cmd = SummaryCommand(container_dir=tmp_path, summary_output_dir=tmp_path / "s")
        result = cmd.execute()
        assert result.return_code == 0
        assert result.payload is fake_result

    def test_no_reports_returns_error(self, monkeypatch):
        import workflow_module.summary_report as sr

        monkeypatch.setattr(sr, "summarize_container", lambda c, o: None)
        cmd = SummaryCommand(container_dir="/x", summary_output_dir="/y")
        result = cmd.execute()
        assert result.return_code == 1
        assert result.error == "no_run_reports"

    def test_exception_is_caught(self, monkeypatch):
        import workflow_module.summary_report as sr

        def _boom(c, o):
            raise RuntimeError("disk full")

        monkeypatch.setattr(sr, "summarize_container", _boom)
        cmd = SummaryCommand(container_dir="/x", summary_output_dir="/y")
        result = cmd.execute()
        assert result.return_code == 1
        assert "disk full" in result.error
