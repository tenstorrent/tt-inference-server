# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Driver for vLLM's structured-output benchmark.

Runs ``benchmark_serving_structured_output.py``, fetched from the pinned vLLM
tag into the LLM_VLLM venv's ``work_dir`` at venv setup, through
``llm_module/structured_output_client.py`` (which renders reasoning models'
prompts with thinking off and checks that the grammar applied). One run per
``StructuredOutputRunConfig``; v1 ran the same command from
``benchmarking/run_benchmarks.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from utils.url_helpers import uses_remote_base_url

from ..config import DriverContext, ServerConnection, StructuredOutputRunConfig
from ..parsers.vllm_structured_output import VLLMStructuredOutputParser
from ._subprocess import load_json, run_command, safe_filename_part
from .base import DriverResult, LLMDriver
from .vllm import _resolve_auth_token

logger = logging.getLogger(__name__)

CLIENT = Path(__file__).parents[1] / "structured_output_client.py"


def build_structured_output_argv(
    *,
    venv_python: Path,
    script: Path,
    config: StructuredOutputRunConfig,
    server: ServerConnection,
    result_path: Path,
) -> Tuple[List[str], str]:
    """The client argv for one run, and the auth token to export for it."""
    cmd: List[str] = [
        str(venv_python),
        str(CLIENT),
        "--script",
        str(script),
        "--backend",
        "vllm",
        "--model",
        server.model,
        "--tokenizer",
        server.tokenizer or server.model,
        "--dataset",
        config.dataset,
        "--max-concurrency",
        str(config.max_concurrency),
        "--num-prompts",
        str(config.num_prompts),
        "--output-len",
        str(config.osl),
        "--request-rate",
        "inf",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--disable-tqdm",
        "--save-results",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    if config.structured_output_ratio is None:
        cmd.append("--no-structured-output")
    else:
        cmd.extend(["--structured-output-ratio", str(config.structured_output_ratio)])
    if config.chat_template_kwargs is not None:
        cmd.extend(["--chat-template-kwargs", json.dumps(config.chat_template_kwargs)])
    if server.tokenizer_trust_remote_code:
        cmd.append("--trust-remote-code")
    if uses_remote_base_url(server.url_with_port, server.is_remote):
        cmd.extend(["--base-url", server.url_with_port])
    else:
        cmd.extend(["--host", server.host, "--port", str(server.service_port)])
    return cmd, _resolve_auth_token(server)


class VLLMStructuredOutputDriver(LLMDriver):
    name = "vllm_structured_output"
    _parser = VLLMStructuredOutputParser()

    def __init__(self, venv_python: Path) -> None:
        from workflows.workflow_venvs import STRUCTURED_OUTPUT_SCRIPT_NAME

        self.venv_python = Path(venv_python)
        # <venv>/bin/python -> <venv>/work_dir (fetch_structured_output_scripts).
        self.script = (
            self.venv_python.parent.parent / "work_dir" / STRUCTURED_OUTPUT_SCRIPT_NAME
        )

    def run(
        self,
        config: StructuredOutputRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        if not self.script.is_file():
            raise FileNotFoundError(
                f"vLLM's structured-output benchmark is missing at {self.script}; "
                "the LLM_VLLM venv setup fetches it (fetch_structured_output_scripts)"
            )
        context.output_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        ratio_tag = (
            "no-so"
            if config.structured_output_ratio is None
            else f"so-{config.structured_output_ratio}"
        )
        result_path = context.output_dir / (
            f"benchmark_structured_{safe_filename_part(server.model)}_{run_ts}"
            f"_dataset-{config.dataset}_{ratio_tag}_osl-{config.osl}"
            f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
        )
        cmd, auth_token = build_structured_output_argv(
            venv_python=self.venv_python,
            script=self.script,
            config=config,
            server=server,
            result_path=result_path,
        )
        env = dict(context.extra_env)
        if auth_token:
            env["OPENAI_API_KEY"] = auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        raw = load_json(result_path) if rc == 0 else None
        if raw is not None:
            raw.update(
                tt_dataset=config.dataset,
                tt_structured_output_ratio=config.structured_output_ratio,
                tt_osl=config.osl,
            )
            active = (raw.get("tt_structured_output") or {}).get("grammar_active_rate")
            if active is not None and active < 1:
                logger.error(
                    "Structured-output run %s ratio %s: only %.0f%% of the "
                    "structured outputs start as a JSON object, so the grammar did "
                    "not constrain the rest (chat_template_kwargs=%s). Its timings "
                    "do not measure structured output.",
                    config.dataset,
                    config.structured_output_ratio,
                    100 * active,
                    config.chat_template_kwargs,
                )
        return DriverResult(return_code=rc, raw=raw, raw_path=result_path)


__all__ = ["VLLMStructuredOutputDriver", "build_structured_output_argv"]
