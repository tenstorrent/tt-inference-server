# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""InferenceMax driver.

Runs the vLLM serve benchmark via the ``vllm bench serve`` CLI(``benchmark_script = venv/bin/vllm`` in
``benchmarking/run_benchmarks.py``). The standalone ``benchmark_serving.py``
is just the un-packaged form of this subcommand and emits the identical flat
JSON the :class:`InferenceMaxParser` reads. Kept distinct from the ``vllm``
driver only by its ``inferencex`` kind and ``inferencex_*.json`` filename.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..parsers.inferencex import InferenceMaxParser
from ._subprocess import load_json, run_command, safe_filename_part
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class InferenceMaxDriver(LLMDriver):
    name = "inferencex"
    _parser = InferenceMaxParser()

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
        result_filename = context.output_dir / (
            f"inferencex_{safe_filename_part(server.model)}_{run_ts}"
            f"_isl-{config.isl}_osl-{config.osl}"
            f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
        )

        cmd = [
            str(self.vllm_binary),
            "bench",
            "serve",
            "--backend",
            "openai-chat",
            "--endpoint",
            "/v1/chat/completions",
            "--extra-body",
            json.dumps(
                {
                    "truncate_prompt_tokens": str(config.isl),
                    "max_tokens": int(config.osl),
                }
            ),
            "--model",
            server.model,
            "--host",
            server.host,
            "--port",
            str(server.service_port),
            "--dataset-name",
            "random",
            "--max-concurrency",
            str(config.max_concurrency),
            "--num-prompts",
            str(config.num_prompts),
            "--random-input-len",
            str(config.isl),
            "--random-output-len",
            str(config.osl),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-filename",
            str(result_filename),
        ]

        env = dict(context.extra_env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        raw = load_json(result_filename) if rc == 0 else None
        return DriverResult(return_code=rc, raw=raw, raw_path=result_filename)
