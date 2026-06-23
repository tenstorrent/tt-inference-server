# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.execution`` result + options dataclasses."""

from __future__ import annotations

from workflow_module.execution import (
    TaskOutcome,
    WorkflowResult,
)


class TestTaskOutcome:
    def test_succeeded_when_exit_zero(self):
        o = TaskOutcome(
            task_type="benchmark",
            exit_code=0,
            elapsed_seconds=1.0,
            block_kind="benchmarks",
        )
        assert o.succeeded is True

    def test_not_succeeded_on_nonzero_exit(self):
        o = TaskOutcome(
            task_type="benchmark", exit_code=2, elapsed_seconds=1.0, block_kind=None
        )
        assert o.succeeded is False


class TestWorkflowResult:
    def test_succeeded_reflects_return_code(self):
        assert WorkflowResult("w", return_code=0).succeeded is True
        assert WorkflowResult("w", return_code=1, error="boom").succeeded is False
