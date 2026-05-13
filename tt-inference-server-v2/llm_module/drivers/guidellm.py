# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""GuideLLM driver.

No v1 reference — fresh runner using the same shape as the other
drivers: invokes the ``guidellm`` CLI from a venv (or PATH-resolved),
writes the benchmarks JSON into the sweep output dir.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ._subprocess import load_json, run_command, safe_filename_part
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class GuideLLMDriver(LLMDriver):
    name = "guidellm"

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
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = context.output_dir / (
            f"guidellm_{safe_filename_part(server.model)}_{run_ts}"
            f"_isl-{config.isl}_osl-{config.osl}"
            f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
        )

        target = server.url_with_port
        data_spec = f"prompt_tokens={config.isl},output_tokens={config.osl}"

        if self.guidellm_binary:
            cmd = [self.guidellm_binary, "benchmark"]
        else:
            cmd = [str(self.venv_python), "-m", "guidellm", "benchmark"]

        cmd.extend(
            [
                "--target",
                target,
                "--model",
                server.model,
                "--data",
                data_spec,
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

        env = dict(context.extra_env)
        if server.auth_token:
            env["GUIDELLM__OPENAI__API_KEY"] = server.auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        raw = load_json(out_path) if rc == 0 else None
        return DriverResult(return_code=rc, raw=raw, raw_path=out_path)
