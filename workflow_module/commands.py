# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    # Annotation-only imports (``from __future__ import annotations`` keeps them
    # unevaluated at runtime). Kept out of the import path so lightweight callers
    # — e.g. run.py constructing a ServerCommand — need not pull the heavy
    # test_module.context / report_module stack.
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


class ServerMode(str, Enum):
    """How :class:`ServerCommand` brings up the inference server."""

    DOCKER = "docker"
    LOCAL = "local"


@dataclass(frozen=True)
class ServerLaunchSpec:
    """Everything :class:`ServerCommand` needs to bring up an inference server.

    Carries the launcher-side objects (``model_spec``, ``runtime_config``,
    ``setup_config``) that :func:`workflows.run_docker_server.run_docker_server`
    and :func:`workflows.run_local_server.run_local_server` expect. They are
    typed ``Any`` here so the command model stays free of a hard import
    dependency on the launcher stack.

    ``mode`` selects the launcher (:class:`ServerMode`). ``json_fpath`` is the
    runtime model-spec JSON path the launchers persist / read (the docker
    launcher only forwards it in ``--dev-mode``).
    """

    mode: ServerMode
    model_spec: Any
    runtime_config: Any
    setup_config: Any
    json_fpath: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, ServerMode):
            return
        try:
            object.__setattr__(self, "mode", ServerMode(self.mode))
        except ValueError as e:
            raise ValueError(f"unknown server mode: {self.mode!r}") from e


class ServerCommand(Command):
    """Bring up the inference server as the first step of a run.

    Wraps ``workflows.run_docker_server`` / ``run_local_server`` so server
    bring-up is a command in the same list the :class:`WorkflowRunner` executes.
    """

    name = "server"

    def __init__(self, launch: ServerLaunchSpec) -> None:
        self.launch = launch

    def execute(self) -> CommandResult:
        from workflows.run_docker_server import run_docker_server
        from workflows.run_local_server import run_local_server

        spec = self.launch
        try:
            if spec.mode is ServerMode.DOCKER:
                payload = run_docker_server(
                    spec.model_spec,
                    spec.runtime_config,
                    spec.setup_config,
                    spec.json_fpath,
                )
            elif spec.mode is ServerMode.LOCAL:
                payload = run_local_server(
                    spec.model_spec,
                    spec.runtime_config,
                    spec.json_fpath,
                    spec.setup_config,
                )
            else:  # pragma: no cover - ServerLaunchSpec rejects unknown modes
                return CommandResult(
                    command_name=self.name,
                    return_code=1,
                    error=f"unknown server mode: {spec.mode!r}",
                )
        except Exception as e:
            logger.exception("Server bring-up failed: %s", e)
            return CommandResult(command_name=self.name, return_code=1, error=str(e))

        return CommandResult(command_name=self.name, return_code=0, payload=payload)


class VenvCommand(Command):
    """Run an argv as a subprocess, optionally inside a declared workflow venv."""

    def __init__(
        self,
        venv_type: Any,
        argv: Sequence[str],
        *,
        model_spec: Any = None,
        env: Optional[Mapping[str, str]] = None,
        label: Optional[str] = None,
        dependency_venvs: Sequence[Any] = (),
    ) -> None:
        self.venv_type = venv_type
        self.argv = list(argv)
        self.model_spec = model_spec
        self.env = dict(env) if env is not None else None
        self.dependency_venvs = list(dependency_venvs)
        if label:
            self.name = label
        elif venv_type is None:
            self.name = "venv[current]"
        else:
            self.name = f"venv[{getattr(venv_type, 'name', venv_type)}]"

    def execute(self) -> CommandResult:
        import os
        import sys

        from workflows.utils import run_command

        if self.venv_type is None:
            python = sys.executable
        else:
            from workflows.workflow_venvs import VENV_CONFIGS

            # Provision the primary venv plus any dependency venvs the workflow
            # needs before running.
            for venv_type in [self.venv_type, *self.dependency_venvs]:
                try:
                    venv_config = VENV_CONFIGS[venv_type]
                except KeyError:
                    return CommandResult(
                        command_name=self.name,
                        return_code=1,
                        error=f"no venv config for {venv_type!r}",
                    )
                if not venv_config.setup(model_spec=self.model_spec):
                    return CommandResult(
                        command_name=self.name,
                        return_code=1,
                        error=(
                            f"failed to provision venv "
                            f"{getattr(venv_type, 'name', venv_type)}"
                        ),
                    )
            python = str(VENV_CONFIGS[self.venv_type].venv_python)

        cmd = [python, *[str(a) for a in self.argv]]
        env = {**os.environ, **self.env} if self.env else None
        try:
            return_code = run_command(cmd, logger=logger, env=env)
        except Exception as e:
            logger.exception("venv command failed: %s", e)
            return CommandResult(command_name=self.name, return_code=1, error=str(e))

        return CommandResult(
            command_name=self.name,
            return_code=return_code,
            error=None if return_code == 0 else f"exit code {return_code}",
        )


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
    "ServerCommand",
    "ServerLaunchSpec",
    "ServerMode",
    "SummaryCommand",
    "VenvCommand",
    "WorkflowCommand",
]
