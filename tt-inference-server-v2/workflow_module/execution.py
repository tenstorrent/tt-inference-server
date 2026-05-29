# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Core workflow execution scaffolding.

A workflow  is a small decorator pipeline around the test_module
dispatcher:

    prepare -> run_tasks -> format_results -> acceptance -> metadata -> report
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence, Tuple

from report_module import (
    GenerateResult,
    ReportGenerator,
    ReportSchema,
    acceptance_criteria_check,
    format_acceptance_summary_markdown,
)
from test_module import MediaContext, MediaTaskType, run_media_task

from .blocks_sink import BlockAccumulator, get_default_accumulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskOutcome:
    """Result of a single ``run_media_task`` invocation inside a workflow."""

    task_type: str
    exit_code: int
    elapsed_seconds: float
    block_kind: Optional[str]

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class WorkflowResult:
    workflow_name: str
    return_code: int
    task_outcomes: List[TaskOutcome] = field(default_factory=list)
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0


@dataclass(frozen=True)
class OrchestratorMetadata:
    """Top-level metadata the per-task runners can't see themselves.

    Populated by the CLI entry point (``run.py``) and threaded into the
    schema by :meth:`WorkflowExecution.inject_metadata`.
    """

    server_mode: Optional[str] = None
    run_command: Optional[str] = None
    runtime_model_spec_json: Optional[str] = None


class WorkflowExecution(ABC):
    """Template-method base for every workflow.

    Subclasses set ``name`` and ``task_types``. The default ``run()``
    invokes each task in order, formats the accumulated Blocks into a
    ``ReportSchema``, evaluates acceptance criteria, injects orchestrator
    metadata, and generates the report."""

    name: ClassVar[str] = ""
    task_types: ClassVar[Sequence[MediaTaskType]] = ()

    def __init__(
        self,
        ctx: MediaContext,
        *,
        accumulator: Optional[BlockAccumulator] = None,
        orchestrator_metadata: Optional[OrchestratorMetadata] = None,
    ) -> None:
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty class-level `name`."
            )
        self.ctx = ctx
        self.accumulator = accumulator or get_default_accumulator()
        self.orchestrator_metadata = orchestrator_metadata or OrchestratorMetadata()
        self.logger = logging.getLogger(f"workflow.{self.name}")

    def run(self) -> WorkflowResult:
        self.logger.info(
            "=== Workflow start: %s | model=%s device=%s port=%d output=%s ===",
            self.name,
            self.ctx.model_spec.model_name,
            self.ctx.device.name,
            self.ctx.service_port,
            self.ctx.output_path,
        )
        try:
            self.prepare()
            task_outcomes = self.run_tasks()
        except Exception as e:
            self.logger.exception(
                "Workflow %s aborted during task phase: %s", self.name, e
            )
            return WorkflowResult(
                workflow_name=self.name,
                return_code=1,
                task_outcomes=[],
                error=str(e),
            )

        schema = self.format_results()
        if schema is None:
            self.logger.error("No blocks accumulated — cannot generate report.")
            return WorkflowResult(
                workflow_name=self.name,
                return_code=1,
                task_outcomes=task_outcomes,
                error="no_blocks",
            )

        try:
            self.apply_acceptance_criteria(schema)
            self.inject_metadata(schema)
            gen = self.generate_report(schema)
        except Exception as e:
            self.logger.exception(
                "Workflow %s aborted during report phase: %s", self.name, e
            )
            return WorkflowResult(
                workflow_name=self.name,
                return_code=1,
                task_outcomes=task_outcomes,
                error=str(e),
            )

        self.logger.info("=== Workflow done: %s (rc=0) ===", self.name)
        return WorkflowResult(
            workflow_name=self.name,
            return_code=0,
            task_outcomes=task_outcomes,
            markdown_path=gen.markdown_path,
            json_path=gen.json_path,
        )

    def prepare(self) -> None:
        # v2 runs against an external inference server, so there's no venv
        # / library install step to do here. Subclasses that need
        # environment prep (warmup pings, dataset downloads, etc.) override
        # this hook.
        self.logger.debug("prepare(): no-op")

    def run_tasks(self) -> List[TaskOutcome]:
        return [self._dispatch_task(t) for t in self.task_types]

    def _dispatch_task(self, task_type: MediaTaskType) -> TaskOutcome:
        self.logger.info("→ task=%s", task_type.value)
        started = time.time()
        try:
            exit_code, block = run_media_task(self.ctx, task_type)
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "❌ task=%s raised after %.1fs: %s", task_type.value, elapsed, e
            )
            return TaskOutcome(task_type.value, 1, elapsed, None)

        elapsed = time.time() - started
        block_kind = block.kind if block is not None else None
        if exit_code != 0 or block is None:
            self.logger.error(
                "❌ task=%s rc=%d block=%s (%.1fs)",
                task_type.value,
                exit_code,
                block_kind,
                elapsed,
            )
        else:
            self.logger.info(
                "✅ task=%s block=%s (%.1fs)", task_type.value, block_kind, elapsed
            )
        return TaskOutcome(task_type.value, exit_code, elapsed, block_kind)

    def format_results(self) -> Optional[ReportSchema]:
        if not self.accumulator.blocks:
            return None
        return self.accumulator.build_schema()

    def apply_acceptance_criteria(self, schema: ReportSchema) -> Tuple[bool, list]:
        accepted, blockers, categories = acceptance_criteria_check(schema)
        schema.metadata["acceptance_summary_markdown"] = (
            format_acceptance_summary_markdown(accepted, blockers, categories)
        )
        schema.metadata["acceptance_criteria"] = {
            "accepted": accepted,
            "blockers": blockers,
            "categories": [c.to_dict() for c in categories],
        }
        self.logger.info(
            "Acceptance: %s (%d blocker(s))",
            "PASS" if accepted else "FAIL",
            len(blockers),
        )
        return accepted, blockers

    def inject_metadata(self, schema: ReportSchema) -> None:
        meta = schema.metadata
        meta["workflow"] = self.name
        m = self.orchestrator_metadata
        if m.server_mode is not None:
            meta["server_mode"] = m.server_mode
        if m.run_command is not None:
            meta["run_command"] = m.run_command
        if m.runtime_model_spec_json is not None:
            meta["runtime_model_spec_json"] = m.runtime_model_spec_json

    def generate_report(self, schema: ReportSchema) -> GenerateResult:
        report_dir = Path(self.ctx.output_path).parent
        result = ReportGenerator().generate(schema, report_dir)
        self.logger.info("Wrote markdown: %s", result.markdown_path)
        self.logger.info("Wrote json:     %s", result.json_path)
        return result


__all__ = [
    "OrchestratorMetadata",
    "TaskOutcome",
    "WorkflowExecution",
    "WorkflowResult",
]
