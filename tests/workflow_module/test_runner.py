# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.runner.WorkflowRunner``.

The runner is command-agnostic, so fake commands exercise its control flow:
run each in order, collect results, stop at the first failure.
"""

from __future__ import annotations

from workflow_module.commands import Command, CommandResult
from workflow_module.runner import WorkflowRunner


class _FakeCommand(Command):
    def __init__(self, name: str, return_code: int, error: str | None = None) -> None:
        self.name = name
        self._return_code = return_code
        self._error = error
        self.executed = False

    def execute(self) -> CommandResult:
        self.executed = True
        return CommandResult(
            command_name=self.name,
            return_code=self._return_code,
            error=self._error,
        )


class TestWorkflowRunner:
    def test_all_succeed_returns_zero(self):
        cmds = [_FakeCommand("a", 0), _FakeCommand("b", 0)]
        runner = WorkflowRunner(cmds)
        assert runner.run() == 0
        assert [r.command_name for r in runner.results] == ["a", "b"]
        assert all(c.executed for c in cmds)

    def test_stops_at_first_failure(self):
        a = _FakeCommand("a", 0)
        b = _FakeCommand("b", 3, error="boom")
        c = _FakeCommand("c", 0)
        runner = WorkflowRunner([a, b, c])
        assert runner.run() == 3
        assert a.executed and b.executed
        assert not c.executed  # short-circuited after b failed
        assert len(runner.results) == 2

    def test_empty_command_list_returns_zero(self):
        assert WorkflowRunner([]).run() == 0
