# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Out-of-process workflow execution.

The orchestrator
(``workflow_runner.py`` / ``CommandFactory``) stays minimal, and each
workflow's heavy client deps (torch / open-clip / scipy / aiohttp /
future vllm-client / lm-eval) live in a dedicated venv that this command
bootstraps and shells out to.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from workflows.model_spec import get_runtime_model_spec
from workflows.utils import run_command
from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

from .commands import Command, CommandResult

logger = logging.getLogger(__name__)

_V2_ROOT = Path(__file__).resolve().parent.parent
_WORKER = _V2_ROOT / "workflow_worker.py"

# name -> venv that workflow's worker runs in. This is the V2 analogue of
# V1's WorkflowConfig.workflow_run_script_venv_type: a config lookup that
# does NOT import the workload, so reading it stays cheap.
#
# Each workflow gets its own venv (requirements/v2-{evals,benchmarks,
# spec-tests}.txt) so conflicting pins never share an interpreter and a
# benchmarks run doesn't install the eval accuracy stack. `release`
# dispatches its children in one process, so it needs the union venv.
WORKFLOW_VENV_TYPES: Dict[str, WorkflowVenvType] = {
    "evals": WorkflowVenvType.V2_EVALS,
    "benchmarks": WorkflowVenvType.V2_BENCHMARKS,
    "spec_tests": WorkflowVenvType.V2_SPEC_TESTS,
    "release": WorkflowVenvType.V2_RELEASE,
}


class SubprocessWorkflowCommand(Command):
    """Runs one workflow in its own venv, in a child process.

    ``execute()`` mirrors V1's ``WorkflowSetup.run_workflow_script``:
    materialize the venv (idempotent), then ``[venv_python, worker, ...]``.
    The worker re-derives the MediaContext from these flags + the runtime
    spec json and runs the workflow in-process. Results land on disk
    (markdown + json); we surface only the return code.
    """

    name = "workflow"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workflow_name = args.workflow

    def execute(self) -> CommandResult:
        try:
            venv_python = self._ensure_venv()
        except Exception as e:
            logger.exception("venv bootstrap failed for %s", self.workflow_name)
            return CommandResult(self.name, return_code=1, error=str(e))

        cmd = self._build_argv(venv_python)
        logger.info("→ %s in venv=%s", self.workflow_name, venv_python)
        rc = run_command(cmd, logger=logger)
        return CommandResult(
            self.name,
            return_code=rc,
            error=None if rc == 0 else f"worker exited rc={rc}",
        )

    def _ensure_venv(self) -> Path:
        venv_type = WORKFLOW_VENV_TYPES.get(self.workflow_name)
        if venv_type is None:
            raise KeyError(
                f"No venv mapped for workflow {self.workflow_name!r}. "
                f"Known: {sorted(WORKFLOW_VENV_TYPES)}"
            )
        venv_config = VENV_CONFIGS[venv_type]
        # model_spec only feeds setup_function hooks (HF prep etc.); the
        # base setup is a no-op once the venv exists, so this is cheap to
        # call on every dispatch.
        model_spec, _, _ = get_runtime_model_spec(
            model=self.args.model, device=self.args.device
        )
        ok = venv_config.setup(model_spec=model_spec)
        assert ok, f"Failed to setup venv: {venv_type.name}"
        return venv_config.venv_python

    def _build_argv(self, venv_python: Path) -> list[str]:
        a = self.args
        # fmt: off
        argv = [
            str(venv_python), str(_WORKER),
            "--model", a.model,
            "--workflow", a.workflow,
            "--device", a.device,
            "--service-port", str(a.service_port),
            "--output-dir", str(a.output_dir),
            "--log-level", a.log_level,
        ]
        # fmt: on
        if a.runtime_model_spec_json:
            argv += ["--runtime-model-spec-json", a.runtime_model_spec_json]
        if a.docker_server:
            argv.append("--docker-server")
        if a.num_prompts is not None:
            argv += ["--num-prompts", str(a.num_prompts)]
        return argv


__all__ = ["SubprocessWorkflowCommand", "WORKFLOW_VENV_TYPES"]
