# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""``vllm bench serve`` driver.

Self-contained port of the v1 ``benchmarking/run_benchmarks.py`` command
build + invocation: assumes the ``vllm`` CLI is available on PATH (or
provided via ``vllm_binary``) and writes ``--save-result`` JSON into the
sweep output dir.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ._subprocess import load_json, run_command
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class VLLMBenchDriver(LLMDriver):
    name = "vllm"

    def __init__(self, vllm_binary: Optional[str] = None) -> None:
        self.vllm_binary = vllm_binary or shutil.which("vllm") or "vllm"

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
                f"benchmark_{_safe(server.model)}_{run_ts}"
                f"_isl-{config.isl}_osl-{config.osl}"
                f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
            )
        )

        cmd = [
            self.vllm_binary,
            "bench",
            "serve",
            "--backend", "openai-chat",
            "--endpoint", "/v1/chat/completions",
            "--extra-body", json.dumps({"truncate_prompt_tokens": str(config.isl)}),
            "--model", server.model,
            "--port", str(server.service_port),
            "--dataset-name", "random",
            "--max-concurrency", str(config.max_concurrency),
            "--num-prompts", str(config.num_prompts),
            "--random-input-len", str(config.isl),
            "--random-output-len", str(config.osl),
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result",
            "--save-detailed",
            "--result-filename", str(result_filename),
        ]

        env = {}
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token
        env.update(context.extra_env)

        rc = run_command(cmd, env=env)
        raw = load_json(result_filename) if rc == 0 else None
        return DriverResult(return_code=rc, raw=raw, raw_path=result_filename)


def _safe(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")
