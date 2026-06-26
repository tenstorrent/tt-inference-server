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
from pathlib import Path
from typing import ClassVar, Dict, List, Sequence, Type

from test_module.task_types import MediaTaskType
from workflows.workflow_types import ModelType

from .execution import (
    LLMBenchOptions,
    PrefixCacheOptions,
    SpecDecodeOptions,
    TaskOutcome,
    WorkflowExecution,
)

# Synthetic task labels used for the prefix-cache / spec-decode runs in
# TaskOutcome / acceptance summary tables. Not members of MediaTaskType
# because these sweeps bypass the media-task dispatcher.
_PREFIX_CACHE_TASK_LABEL = "prefix_cache"
_LLM_BENCH_TASK_LABEL = "llm_benchmark"
_LLM_EVAL_TASK_LABEL = "llm_eval"
_SPEC_DECODE_TASK_LABEL = "spec_decode"


class EvalsWorkflow(WorkflowExecution):
    name = "evals"
    task_types = (MediaTaskType.EVALUATION,)

    def run_tasks(self) -> List[TaskOutcome]:
        if self.ctx.model_spec.model_type == ModelType.LLM:
            return [self._run_llm_eval_task()]
        return super().run_tasks()

    def _run_llm_eval_task(self) -> TaskOutcome:
        """Drive the standard (lm-eval / lmms-eval) sweep for an LLM model.

        Delegates to :func:`test_module.llm_tests.llm_eval_tests.run_llm_eval`,
        which gates on server health, runs each task, scores the results, and
        forwards Blocks to the accumulator. Agentic evals are a separate
        workflow. Imported from the leaf submodule so the media runner imports
        stay untouched.
        """
        from test_module.llm_tests.llm_eval_tests import run_llm_eval

        opts = self.orchestrator_metadata.llm_eval
        auth_token = opts.auth_token if opts is not None else ""
        self.logger.info("→ task=%s", _LLM_EVAL_TASK_LABEL)
        started = time.time()
        try:
            blocks = run_llm_eval(self.ctx, auth_token=auth_token)
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "❌ %s raised after %.1fs: %s", _LLM_EVAL_TASK_LABEL, elapsed, e
            )
            return TaskOutcome("evaluation", 1, elapsed, None)

        elapsed = time.time() - started
        if not blocks:
            # No standard eval tasks configured (e.g. an agentic-only model) —
            # a clean no-op, not a failure. Acceptance still runs on whatever
            # other workflows accumulated.
            self.logger.info(
                "%s: model has no standard eval tasks (%.1fs)",
                _LLM_EVAL_TASK_LABEL,
                elapsed,
            )
            return TaskOutcome("evaluation", 0, elapsed, None)

        self.logger.info(
            "✅ %s blocks=%d kind=%s (%.1fs)",
            _LLM_EVAL_TASK_LABEL,
            len(blocks),
            blocks[0].kind,
            elapsed,
        )
        return TaskOutcome("evaluation", 0, elapsed, blocks[0].kind)


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
            self.logger.exception("❌ agentic raised after %.1fs: %s", elapsed, e)
            return [TaskOutcome("evaluation", 1, elapsed, None)]

        elapsed = time.time() - started
        if not blocks:
            self.logger.error("❌ agentic produced no blocks (%.1fs)", elapsed)
            return [TaskOutcome("evaluation", 1, elapsed, None)]

        self.logger.info(
            "✅ agentic blocks=%d kind=%s (%.1fs)",
            len(blocks),
            blocks[0].kind,
            elapsed,
        )
        return [TaskOutcome("evaluation", 0, elapsed, blocks[0].kind)]


class ServingBenchWorkflow(WorkflowExecution):
    """Serving benchmark suites (agentic_bench, benchmark).

    Bypasses the media-task dispatcher: each suite under
    ``test_module/serving_bench`` is a self-contained shell harness driven
    by ``run_test.sh``, which manages its own per-suite uv venv. Emits one
    Block(kind="serving_bench") per suite.
    """

    name = "serving_bench"
    task_types = ()

    def run_tasks(self) -> List[TaskOutcome]:
        from test_module.serving_bench.runner import run_serving_bench

        opts = self.orchestrator_metadata.serving_bench
        suites = opts.suites if opts is not None else None
        self.logger.info("→ task=serving_bench suites=%s", suites or "all")
        started = time.time()
        try:
            results = run_serving_bench(self.ctx, suites=suites)
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception("❌ serving_bench raised after %.1fs: %s", elapsed, e)
            return [TaskOutcome("serving_bench", 1, elapsed, None)]

        if not results:
            elapsed = time.time() - started
            self.logger.error("❌ serving_bench ran no suites (%.1fs)", elapsed)
            return [TaskOutcome("serving_bench", 1, elapsed, None)]

        for r in results:
            mark = "✅" if r.return_code == 0 else "❌"
            self.logger.info(
                "%s serving_bench:%s rc=%d (%.1fs)",
                mark,
                r.suite,
                r.return_code,
                r.elapsed_seconds,
            )
        return [
            TaskOutcome(
                f"serving_bench:{r.suite}",
                r.return_code,
                r.elapsed_seconds,
                "serving_bench",
            )
            for r in results
        ]


class BenchmarksWorkflow(WorkflowExecution):
    name = "benchmarks"
    task_types = (MediaTaskType.BENCHMARK,)

    def run_tasks(self) -> List[TaskOutcome]:
        prefix_cache_opts = self.orchestrator_metadata.prefix_cache
        if prefix_cache_opts is not None:
            return [self._run_prefix_cache_task(prefix_cache_opts)]
        spec_decode_opts = self.orchestrator_metadata.spec_decode
        if spec_decode_opts is not None:
            return [self._run_spec_decode_task(spec_decode_opts)]
        if self.ctx.model_spec.model_type == ModelType.LLM:
            opts = self.orchestrator_metadata.llm_bench or LLMBenchOptions()
            return [self._run_llm_bench_task(opts)]
        return super().run_tasks()

    def _run_llm_bench_task(self, opts: LLMBenchOptions) -> TaskOutcome:
        """Drive the LLM performance sweep in place of media benchmarks.

        Delegates to :func:`test_module.llm_tests.llm_benchmark_tests.run_llm_bench`,
        which selects the perf-tool driver from ``opts.tools``, builds the
        ``BENCHMARK_CONFIGS`` sweep, runs it, and forwards the resulting
        Blocks to the accumulator. Imported from the leaf submodule so the
        media runner imports stay untouched.
        """
        from test_module.llm_tests.llm_benchmark_tests import run_llm_bench

        self.logger.info("→ task=%s tools=%s", _LLM_BENCH_TASK_LABEL, opts.tools)
        venv_python = Path(opts.venv_python) if opts.venv_python else None
        return self._run_bench_task(
            _LLM_BENCH_TASK_LABEL,
            lambda: run_llm_bench(
                self.ctx,
                tools=opts.tools,
                auth_token=opts.auth_token,
                venv_python=venv_python,
            ),
        )

    def _run_prefix_cache_task(self, opts: PrefixCacheOptions) -> TaskOutcome:
        """Drive the AIPerf prefix-cache sweep in place of media benchmarks.

        Delegates to :func:`test_module.llm_tests.prefix_cache_tests.run_prefix_cache`,
        which builds the scenario plan, runs each AIPerf invocation, and
        forwards the resulting Blocks to the accumulator.

        Imported from the leaf submodule (not ``test_module``) so the
        prefix-cache code path skips the audio/image/video/CNN/TTS/
        embedding runner imports that ``test_module/__init__.py`` would
        otherwise trigger.
        """
        from test_module.llm_tests.prefix_cache_tests import run_prefix_cache

        self.logger.info("→ task=%s preset=%s", _PREFIX_CACHE_TASK_LABEL, opts.preset)
        return self._run_bench_task(
            _PREFIX_CACHE_TASK_LABEL,
            lambda: run_prefix_cache(
                self.ctx,
                preset=opts.preset,
                scenarios=opts.scenarios,
                arrival_pattern=opts.arrival_pattern,
                request_rate=opts.request_rate,
                scenarios_json=opts.scenarios_json,
                trace_path=opts.trace_path,
                auth_token=opts.auth_token,
                metrics_urls=opts.metrics_urls,
            ),
        )

    def _run_bench_task(self, label: str, run_sweep) -> TaskOutcome:
        """Run an LLM sweep callable and map its ``RunnerResult`` to a TaskOutcome.

        ``run_sweep`` returns a :class:`llm_module.runner.RunnerResult`. A
        non-zero return code on *any* sweep point (``result.ok`` is False)
        fails the task even when some Blocks were produced — a partial sweep
        failure must not report success.
        """
        started = time.time()
        try:
            result = run_sweep()
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "❌ task=%s raised after %.1fs: %s", label, elapsed, e
            )
            return TaskOutcome(label, 1, elapsed, None)

        elapsed = time.time() - started
        if not result.blocks:
            self.logger.error("❌ task=%s produced no blocks (%.1fs)", label, elapsed)
            return TaskOutcome(label, 1, elapsed, None)

        block_kind = result.blocks[0].kind
        if not result.ok:
            self.logger.error(
                "❌ task=%s partial failure: %d block(s) but return_codes=%s (%.1fs)",
                label,
                len(result.blocks),
                result.return_codes,
                elapsed,
            )
            return TaskOutcome(label, 1, elapsed, block_kind)

        self.logger.info(
            "✅ task=%s blocks=%d kind=%s (%.1fs)",
            label,
            len(result.blocks),
            block_kind,
            elapsed,
        )
        return TaskOutcome(label, 0, elapsed, block_kind)

    def _run_spec_decode_task(self, opts: SpecDecodeOptions) -> TaskOutcome:
        """Drive one spec-decode phase in place of media benchmarks.

        Delegates to :func:`test_module.llm_tests.spec_decode_tests.run_spec_decode`,
        which builds the sweep plan, runs each AIPerf invocation, and
        forwards the resulting Blocks to the accumulator. We only need
        to translate its ``list[Block]`` return into a single
        :class:`TaskOutcome`.

        Imported from the leaf submodule (not ``test_module``) so the
        spec-decode code path skips the audio/image/video/CNN/TTS/
        embedding runner imports that ``test_module/__init__.py`` would
        otherwise trigger.
        """
        from test_module.llm_tests.spec_decode_tests import run_spec_decode

        self.logger.info("→ task=%s preset=%s", _SPEC_DECODE_TASK_LABEL, opts.preset)
        started = time.time()
        try:
            blocks = run_spec_decode(
                self.ctx,
                preset=opts.preset,
                warmup_requests=opts.warmup_requests,
                auth_token=opts.auth_token,
            )
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "✘ task=%s raised after %.1fs: %s",
                _SPEC_DECODE_TASK_LABEL,
                elapsed,
                e,
            )
            return TaskOutcome(_SPEC_DECODE_TASK_LABEL, 1, elapsed, None)

        elapsed = time.time() - started
        if not blocks:
            self.logger.error(
                "✘ task=%s produced no blocks (%.1fs)",
                _SPEC_DECODE_TASK_LABEL,
                elapsed,
            )
            return TaskOutcome(_SPEC_DECODE_TASK_LABEL, 1, elapsed, None)

        block_kind = blocks[0].kind
        self.logger.info(
            "✓ task=%s blocks=%d kind=%s (%.1fs)",
            _SPEC_DECODE_TASK_LABEL,
            len(blocks),
            block_kind,
            elapsed,
        )
        return TaskOutcome(_SPEC_DECODE_TASK_LABEL, 0, elapsed, block_kind)


class SpecTestsWorkflow(WorkflowExecution):
    """Pure test workflow — runs spec_tests only, no eval/benchmark."""

    name = "spec_tests"
    task_types = (MediaTaskType.SPEC_TESTS,)


class ReleaseWorkflow(WorkflowExecution):
    name = "release"
    children: ClassVar[Sequence[str]] = ("evals", "benchmarks", "spec_tests")
    llm_children: ClassVar[Sequence[str]] = ("evals", "benchmarks")

    def run_tasks(self) -> List[TaskOutcome]:
        children = (
            self.llm_children
            if self.ctx.model_spec.model_type == ModelType.LLM
            else self.children
        )
        outcomes: List[TaskOutcome] = []
        for child_name in children:
            child_cls = WORKFLOW_REGISTRY[child_name]
            self.logger.info("--- release: %s ---", child_name)
            child = child_cls(
                self.ctx,
                accumulator=self.accumulator,
                orchestrator_metadata=self.orchestrator_metadata,
            )
            outcomes.extend(child.run_tasks())
        return outcomes


WORKFLOW_REGISTRY: Dict[str, Type[WorkflowExecution]] = {
    EvalsWorkflow.name: EvalsWorkflow,
    AgenticWorkflow.name: AgenticWorkflow,
    BenchmarksWorkflow.name: BenchmarksWorkflow,
    ServingBenchWorkflow.name: ServingBenchWorkflow,
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
    "ServingBenchWorkflow",
    "ReleaseWorkflow",
    "SpecTestsWorkflow",
    "WORKFLOW_REGISTRY",
    "get_workflow_class",
]
