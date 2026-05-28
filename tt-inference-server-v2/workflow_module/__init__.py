# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from .blocks_sink import BlockAccumulator, accept_blocks, get_default_accumulator

_LAZY_FROM_EXECUTION = {
    "OrchestratorMetadata",
    "TaskOutcome",
    "WorkflowExecution",
    "WorkflowResult",
}

_LAZY_FROM_WORKFLOWS = {
    "BenchmarksWorkflow",
    "EvalsWorkflow",
    "ReleaseWorkflow",
    "SpecDecodeBenchmarksWorkflow",
    "SpecTestsWorkflow",
    "WORKFLOW_REGISTRY",
    "get_workflow_class",
}


def __getattr__(name):
    if name in _LAZY_FROM_EXECUTION:
        from . import execution

        return getattr(execution, name)
    if name in _LAZY_FROM_WORKFLOWS:
        from . import workflows

        return getattr(workflows, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BlockAccumulator",
    "accept_blocks",
    "get_default_accumulator",
    *sorted(_LAZY_FROM_EXECUTION),
    *sorted(_LAZY_FROM_WORKFLOWS),
]
