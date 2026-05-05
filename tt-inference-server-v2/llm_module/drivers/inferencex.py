# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""InferenceMax driver.

No v1 reference — fresh runner that invokes
``benchmark_serving.py`` from an InferenceMax checkout. The script
path is configurable so callers can point at their local clone.
The result file matches v1 ``vllm bench serve``'s flat shape (the
InferenceMax fork is descended from it), parsed by
:class:`llm_module.parsers.InferenceMaxParser`.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ._subprocess import load_json, run_command
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class InferenceMaxDriver(LLMDriver):
    name = "inferencex"

    def __init__(
        self,
        benchmark_script: Path,
        venv_python: Optional[Path] = None,
    ) -> None:
        self.benchmark_script = Path(benchmark_script)
        self.venv_python = (
            Path(venv_python) if venv_python else Path(sys.executable)
        )

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_filename = (
            context.output_dir
            / (
                f"inferencex_{_safe(server.model)}_{run_ts}"
                f"_isl-{config.isl}_osl-{config.osl}"
                f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
            )
        )

        cmd = [
            str(self.venv_python),
            str(self.benchmark_script),
            "--backend", "openai-chat",
            "--endpoint", "/v1/chat/completions",
            "--model", server.model,
            "--port", str(server.service_port),
            "--dataset-name", "random",
            "--max-concurrency", str(config.max_concurrency),
            "--num-prompts", str(config.num_prompts),
            "--random-input-len", str(config.isl),
            "--random-output-len", str(config.osl),
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-filename", str(result_filename),
        ]

        env = dict(context.extra_env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token

        rc = run_command(cmd, env=env)
        raw = load_json(result_filename) if rc == 0 else None
        return DriverResult(return_code=rc, raw=raw, raw_path=result_filename)


def _safe(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")
