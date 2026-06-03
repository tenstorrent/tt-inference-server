# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from test_module import MediaContext

from .execution import OrchestratorMetadata, WorkflowResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    command_name: str
    return_code: int
    error: Optional[str] = None
    payload: Optional[Any] = None

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0


class Command(ABC):
    name: str = ""

    @abstractmethod
    def execute(self) -> CommandResult: ...


class WorkflowCommand(Command):
    name = "workflow"

    def __init__(
        self,
        ctx: MediaContext,
        *,
        workflow_name: str,
        orchestrator_metadata: OrchestratorMetadata,
        num_prompts: Optional[int] = None,
        continue_on_failure: bool = False,
    ) -> None:
        self.ctx = ctx
        self.workflow_name = workflow_name
        self.orchestrator_metadata = orchestrator_metadata
        self.num_prompts = num_prompts
        self.continue_on_failure = continue_on_failure

    def execute(self) -> CommandResult:
        from .blocks_sink import get_default_accumulator
        from .workflows import get_workflow_class

        self._apply_num_prompts_override()
        get_default_accumulator().clear()
        workflow_cls = get_workflow_class(self.workflow_name)
        workflow = workflow_cls(
            self.ctx,
            orchestrator_metadata=self.orchestrator_metadata,
        )
        result: WorkflowResult = workflow.run()
        return_code = result.return_code
        if return_code != 0 and self.continue_on_failure:
            logger.warning(
                "Workflow run failed (rc=%d, error=%s) but continuing because "
                "--repeat is active; this run is excluded from the summary.",
                return_code,
                result.error,
            )
            return_code = 0
        return CommandResult(
            command_name=self.name,
            return_code=return_code,
            error=result.error,
            payload=result,
        )

    def _apply_num_prompts_override(self) -> None:
        if self.num_prompts is None:
            return
        from test_module.benchmark_tests import image_benchmark_tests as _ibt

        _ibt.SDXL_BENCHMARK_NUM_PROMPTS = self.num_prompts
        _ibt.SDXL_SD35_BENCHMARK_NUM_PROMPTS = self.num_prompts
        logger.info(
            "Overriding image benchmark + spec_tests prompt count to %d",
            self.num_prompts,
        )


class SummaryCommand(Command):
    """Aggregate every per-run report under a container into one summary report."""

    name = "benchmark_summary"

    def __init__(self, *, container_dir: Path, summary_output_dir: Path) -> None:
        self.container_dir = container_dir
        self.summary_output_dir = summary_output_dir

    def execute(self) -> CommandResult:
        from .summary_report import summarize_container

        try:
            result = summarize_container(self.container_dir, self.summary_output_dir)
        except Exception as e:
            logger.exception("Benchmark summary failed: %s", e)
            return CommandResult(command_name=self.name, return_code=1, error=str(e))

        if result is None:
            logger.error(
                "No benchmark run reports found under %s — nothing to summarize.",
                self.container_dir,
            )
            return CommandResult(
                command_name=self.name, return_code=1, error="no_run_reports"
            )

        logger.info("Wrote benchmark summary: %s", result.markdown_path)
        return CommandResult(command_name=self.name, return_code=0, payload=result)


__all__ = [
    "Command",
    "CommandResult",
    "SummaryCommand",
    "WorkflowCommand",
]
