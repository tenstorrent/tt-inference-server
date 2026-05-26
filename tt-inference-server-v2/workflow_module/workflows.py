# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Concrete workflows + a name -> class registry.

Each leaf workflow just declares its ``task_types``. ``ReleaseWorkflow``
composes leaves by name so adding a new leaf to a release is a one-line
registry edit, not a structural change.
"""

from __future__ import annotations

import time
from typing import ClassVar, Dict, List, Sequence, Type

from test_module.task_types import MediaTaskType

from .execution import PrefixCacheOptions, TaskOutcome, WorkflowExecution

# Synthetic task label used for the prefix-cache run in TaskOutcome /
# acceptance summary tables. Not a member of MediaTaskType because the
# sweep bypasses the media-task dispatcher.
_PREFIX_CACHE_TASK_LABEL = "prefix_cache"


class EvalsWorkflow(WorkflowExecution):
    name = "evals"
    task_types = (MediaTaskType.EVALUATION,)


class AgenticWorkflow(WorkflowExecution):
    """Agentic evals (Terminal-Bench-2, SWE-bench Verified).

    A flavor of evals — emits Block(kind="evals"), reuses _check_evals.
    Bypasses the media-task dispatcher because the agentic runner has
    its own multi-task loop (one Block per terminal_bench / swebench
    task in the model's EvalConfig).
    """

    name = "agentic"
    task_types = (MediaTaskType.EVALUATION,)

    def run_tasks(self) -> List[TaskOutcome]:
        from test_module.llm_tests.agentic_eval_tests import run_llm_agentic_eval

        self.logger.info("→ task=agentic")
        started = time.time()
        try:
            blocks = run_llm_agentic_eval(self.ctx)
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "✘ agentic raised after %.1fs: %s", elapsed, e
            )
            return [TaskOutcome("evaluation", 1, elapsed, None)]

        elapsed = time.time() - started
        if not blocks:
            self.logger.error("✘ agentic produced no blocks (%.1fs)", elapsed)
            return [TaskOutcome("evaluation", 1, elapsed, None)]

        self.logger.info(
            "✓ agentic blocks=%d kind=%s (%.1fs)",
            len(blocks),
            blocks[0].kind,
            elapsed,
        )
        return [TaskOutcome("evaluation", 0, elapsed, blocks[0].kind)]


class BenchmarksWorkflow(WorkflowExecution):
    name = "benchmarks"
    task_types = (MediaTaskType.BENCHMARK,)

    def run_tasks(self) -> List[TaskOutcome]:
        opts = self.orchestrator_metadata.prefix_cache
        if opts is None:
            return super().run_tasks()
        return [self._run_prefix_cache_task(opts)]

    def _run_prefix_cache_task(self, opts: PrefixCacheOptions) -> TaskOutcome:
        """Drive the AIPerf prefix-cache sweep in place of media benchmarks.

        Delegates to :func:`test_module.llm_tests.prefix_cache_tests.run_prefix_cache`,
        which builds the scenario plan, runs each AIPerf invocation, and
        forwards the resulting Blocks to the accumulator. We only need
        to translate its ``list[Block]`` return into a single
        :class:`TaskOutcome`.

        Imported from the leaf submodule (not ``test_module``) so the
        prefix-cache code path skips the audio/image/video/CNN/TTS/
        embedding runner imports that ``test_module/__init__.py`` would
        otherwise trigger.
        """
        from test_module.llm_tests.prefix_cache_tests import run_prefix_cache

        self.logger.info("→ task=%s preset=%s", _PREFIX_CACHE_TASK_LABEL, opts.preset)
        started = time.time()
        try:
            blocks = run_prefix_cache(
                self.ctx,
                preset=opts.preset,
                scenarios=opts.scenarios,
                arrival_pattern=opts.arrival_pattern,
                request_rate=opts.request_rate,
                scenarios_json=opts.scenarios_json,
                trace_path=opts.trace_path,
                auth_token=opts.auth_token,
            )
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "✘ task=%s raised after %.1fs: %s",
                _PREFIX_CACHE_TASK_LABEL,
                elapsed,
                e,
            )
            return TaskOutcome(_PREFIX_CACHE_TASK_LABEL, 1, elapsed, None)

        elapsed = time.time() - started
        if not blocks:
            self.logger.error(
                "✘ task=%s produced no blocks (%.1fs)",
                _PREFIX_CACHE_TASK_LABEL,
                elapsed,
            )
            return TaskOutcome(_PREFIX_CACHE_TASK_LABEL, 1, elapsed, None)

        block_kind = blocks[0].kind
        self.logger.info(
            "✓ task=%s blocks=%d kind=%s (%.1fs)",
            _PREFIX_CACHE_TASK_LABEL,
            len(blocks),
            block_kind,
            elapsed,
        )
        return TaskOutcome(_PREFIX_CACHE_TASK_LABEL, 0, elapsed, block_kind)


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
    AgenticWorkflow.name: AgenticWorkflow,
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
    "AgenticWorkflow",
    "BenchmarksWorkflow",
    "EvalsWorkflow",
    "ReleaseWorkflow",
    "SpecTestsWorkflow",
    "WORKFLOW_REGISTRY",
    "get_workflow_class",
]
