# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.execution`` result + options dataclasses."""

from __future__ import annotations

from workflow_module.execution import (
    OrchestratorMetadata,
    PrefixCacheOptions,
    ServingBenchOptions,
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

    def test_defaults(self):
        r = WorkflowResult("w", return_code=0)
        assert r.task_outcomes == []
        assert r.markdown_path is None and r.json_path is None and r.error is None


class TestOptions:
    def test_prefix_cache_defaults(self):
        opts = PrefixCacheOptions()
        assert opts.preset == "full"
        assert opts.scenarios is None
        assert opts.auth_token == ""

    def test_serving_bench_default_runs_all_suites(self):
        assert ServingBenchOptions().suites is None

    def test_orchestrator_metadata_defaults(self):
        meta = OrchestratorMetadata()
        assert meta.server_mode is None
        assert meta.prefix_cache is None
        assert meta.serving_bench is None
