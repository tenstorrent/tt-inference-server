# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""GuideLLM driver.

Two run modes, selected by the config type:

- :class:`GuideLLMScenario` — a dataset-driven scenario (multi_turn_chat,
  custom_dataset, omni_modal).
- :class:`LLMRunConfig` — the synthetic ISL/OSL sweep point, shaped like
  the other drivers (``prompt_tokens=N,output_tokens=M``, concurrent rate).

Both invoke the ``guidellm`` CLI from a venv (or PATH-resolved) and write
a single benchmarks JSON the :class:`GuideLLMParser` consumes.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..guidellm_scenarios import GuideLLMScenario
from ..parsers.guidellm import GuideLLMParser
from ._subprocess import load_json, run_command, safe_filename_part
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class GuideLLMDriver(LLMDriver):
    name = "guidellm"
    _parser = GuideLLMParser()

    def __init__(
        self,
        venv_python: Optional[Path] = None,
        guidellm_binary: Optional[str] = None,
    ) -> None:
        self.venv_python = Path(venv_python) if venv_python else Path(sys.executable)
        self.guidellm_binary = guidellm_binary or shutil.which("guidellm")

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(config, GuideLLMScenario):
            return self._run_scenario(config, server, context)
        return self._run_sweep(config, server, context)

    def _base_cmd(self) -> List[str]:
        if self.guidellm_binary:
            return [self.guidellm_binary, "benchmark"]
        return [str(self.venv_python), "-m", "guidellm", "benchmark"]

    def _execute(
        self,
        cmd: List[str],
        server: ServerConnection,
        context: DriverContext,
        out_path: Path,
    ) -> DriverResult:
        env = dict(context.extra_env)
        if server.auth_token:
            env["GUIDELLM__OPENAI__API_KEY"] = server.auth_token
            env.setdefault("OPENAI_API_KEY", server.auth_token)
        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        raw = load_json(out_path) if rc == 0 else None
        return DriverResult(return_code=rc, raw=raw, raw_path=out_path)

    def _run_sweep(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = context.output_dir / (
            f"guidellm_{safe_filename_part(server.model)}_{run_ts}"
            f"_isl-{config.isl}_osl-{config.osl}"
            f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
        )
        cmd = self._base_cmd()
        cmd.extend(
            [
                "--target",
                server.url_with_port,
                "--model",
                server.model,
                "--data",
                f"prompt_tokens={config.isl},output_tokens={config.osl}",
                "--rate-type",
                "concurrent",
                "--rate",
                str(config.max_concurrency),
                "--max-requests",
                str(config.num_prompts),
                "--output-path",
                str(out_path),
            ]
        )
        return self._execute(cmd, server, context, out_path)

    def _run_scenario(
        self,
        scenario: GuideLLMScenario,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = context.output_dir / (
            f"guidellm_{safe_filename_part(server.model)}_{scenario.name}_{run_ts}.json"
        )
        # guidellm appends the OpenAI route to --target; v1 passed the /v1 base.
        target = f"{server.url_with_port}/v1"
        cmd = self._base_cmd()
        cmd.extend(
            [
                "--disable-console-interactive",
                "--target",
                target,
                "--model",
                server.model,
                "--profile",
                scenario.profile,
                "--backend-type",
                scenario.backend_type,
                "--output-path",
                str(out_path),
                "--data",
                scenario.data,
            ]
        )
        if scenario.request_type:
            cmd.extend(["--request-type", scenario.request_type])
        if scenario.max_requests is not None:
            cmd.extend(["--max-requests", str(scenario.max_requests)])
        if scenario.max_seconds is not None:
            cmd.extend(["--max-seconds", str(scenario.max_seconds)])
        if scenario.data_args:
            cmd.extend(["--data-args", scenario.data_args])
        if scenario.data_column_mapper:
            cmd.extend(["--data-column-mapper", scenario.data_column_mapper])
        if scenario.data_preprocessors:
            cmd.extend(["--data-preprocessors", scenario.data_preprocessors])
        if scenario.backend_kwargs:
            cmd.extend(["--backend-kwargs", scenario.backend_kwargs])
        if scenario.extra_args:
            cmd.extend(scenario.extra_args)
        logger.info("Running GuideLLM scenario: %s", scenario.name)
        return self._execute(cmd, server, context, out_path)
