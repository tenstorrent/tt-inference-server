# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Concrete workflows + a name -> class registry.

Each leaf workflow just declares its ``task_types``. ``ReleaseWorkflow``
composes leaves by name so adding a new leaf to a release is a one-line
registry edit, not a structural change.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Sequence, Type

from test_module import MediaTaskType

from .execution import TaskOutcome, WorkflowExecution


class EvalsWorkflow(WorkflowExecution):
    name = "evals"
    task_types = (MediaTaskType.EVALUATION,)


class BenchmarksWorkflow(WorkflowExecution):
    name = "benchmarks"
    task_types = (MediaTaskType.BENCHMARK,)


class SpecTestsWorkflow(WorkflowExecution):
    """Pure test workflow — runs spec_tests only, no eval/benchmark."""

    name = "spec_tests"
    task_types = (MediaTaskType.SPEC_TESTS,)


class ReleaseWorkflow(WorkflowExecution):
    name = "release"
    children: ClassVar[Sequence[str]] = ("evals", "benchmarks", "spec_tests")

    def run_tasks(self) -> List[TaskOutcome]:
        outcomes: List[TaskOutcome] = []
        for child_name in self.children:
            child_cls = WORKFLOW_REGISTRY[child_name]
            self.logger.info("--- release: %s ---", child_name)
            for task_type in child_cls.task_types:
                outcomes.append(self._dispatch_task(task_type))
        return outcomes


WORKFLOW_REGISTRY: Dict[str, Type[WorkflowExecution]] = {
    EvalsWorkflow.name: EvalsWorkflow,
    BenchmarksWorkflow.name: BenchmarksWorkflow,
    SpecTestsWorkflow.name: SpecTestsWorkflow,
    ReleaseWorkflow.name: ReleaseWorkflow,
}


def get_workflow_class(name: str) -> Type[WorkflowExecution]:
    try:
        return WORKFLOW_REGISTRY[name]
    except KeyError as e:
        known = ", ".join(sorted(WORKFLOW_REGISTRY))
        raise KeyError(f"Unknown workflow {name!r}. Known: {known}") from e


__all__ = [
    "BenchmarksWorkflow",
    "EvalsWorkflow",
    "ReleaseWorkflow",
    "SpecTestsWorkflow",
    "WORKFLOW_REGISTRY",
    "get_workflow_class",
]
