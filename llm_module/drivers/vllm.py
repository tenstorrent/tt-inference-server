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
import os
import shutil
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from utils.url_helpers import uses_remote_base_url

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..parsers.vllm import VLLMBenchParser
from ._subprocess import load_json, run_command, safe_filename_part
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


def _resolve_auth_token(server: ServerConnection) -> str:
    return (
        server.auth_token or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    )


@lru_cache(maxsize=None)
def _chat_template_defined(name: str) -> bool:
    """Whether ``name``'s tokenizer defines a chat template, from local files only.

    Mirrors the check ``utils.prompt_generation.template_prompt`` already makes. A
    template may live in ``tokenizer_config.json`` or, since transformers v4.43,
    in a standalone ``chat_template.jinja``; loading the tokenizer resolves either.

    ``local_files_only=True`` keeps this off the network: the served checkpoint is
    already on the host (the server just loaded it), and a probe that downloads a
    tokenizer merely to inspect it could stall or fail for remote-endpoint runs.
    Any failure returns ``True`` so unknown capability keeps the historical
    chat-endpoint behaviour instead of silently moving every model to
    ``/v1/completions``.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            name, local_files_only=True, trust_remote_code=True
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "unknown"
        logger.warning(
            "Could not load tokenizer %s locally to check for a chat template (%s); "
            "assuming one exists and using the chat endpoint.",
            name,
            exc,
        )
        return True
    return bool(getattr(tokenizer, "chat_template", None))


def tokenizer_defines_chat_template(server: ServerConnection) -> bool:
    """Whether the served checkpoint's tokenizer defines a chat template."""
    name = server.tokenizer or server.model
    if not name:
        return True
    return _chat_template_defined(name)


def _endpoint_args(server: ServerConnection) -> List[str]:
    """Pick the benchmark endpoint from the tokenizer's chat capability.

    ``vllm bench serve`` fails its pre-flight probe with ``Bad Request`` when it
    posts to ``/v1/chat/completions`` against a server whose tokenizer defines no
    chat template -- the server raises ``ChatTemplateResolutionError`` because, as
    of transformers v4.44, there is no implicit default template. Base
    checkpoints are the models this affects; they are also the ones for which
    ``/v1/completions`` is the semantically correct endpoint, since wrapping a
    completion model in an invented chat template changes the measured prompt.
    """
    if tokenizer_defines_chat_template(server):
        return ["--backend", "openai-chat", "--endpoint", "/v1/chat/completions"]

    logger.info(
        "Tokenizer for %s defines no chat template; benchmarking against "
        "/v1/completions instead of /v1/chat/completions.",
        server.tokenizer or server.model,
    )
    return ["--backend", "vllm", "--endpoint", "/v1/completions"]


def build_vllm_bench_serve_argv(
    *,
    vllm_binary: str,
    config: LLMRunConfig,
    server: ServerConnection,
    result_filename: Path,
) -> Tuple[List[str], str]:
    """Build the ``vllm bench serve`` argv list.

    Local servers use ``--host``/``--port`` and vLLM-specific ``extra_body``.
    Remote OpenAI-compatible endpoints (e.g. the Tenstorrent console) need
    ``--base-url`` with TLS, explicit auth headers, and vLLM's internal ready
    check disabled after ``RemoteOpenAIController`` has already probed
    ``/v1/models``.
    """
    auth_token = _resolve_auth_token(server)
    headers = ["Accept-Encoding=identity"]

    cmd: List[str] = [
        vllm_binary,
        "bench",
        "serve",
        *_endpoint_args(server),
        "--model",
        server.model,
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
        "--save-detailed",
        "--result-filename",
        str(result_filename),
    ]

    if uses_remote_base_url(server.url_with_port, server.is_remote):
        cmd.extend(["--base-url", server.url_with_port])
        cmd.extend(["--ready-check-timeout-sec", "0"])
        cmd.extend(["--trust-remote-code"])
        if auth_token:
            headers.append(f"Authorization=Bearer {auth_token}")
    else:
        cmd.extend(["--host", server.host, "--port", str(server.service_port)])
        cmd.extend(
            [
                "--extra-body",
                json.dumps({"truncate_prompt_tokens": str(config.isl)}),
            ]
        )

    # vllm bench serve defines --header with nargs="*"; pass all headers on one flag.
    cmd.extend(["--header", *headers])
    return cmd, auth_token


class VLLMBenchDriver(LLMDriver):
    name = "vllm"
    _parser = VLLMBenchParser()

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
            f"benchmark_{safe_filename_part(server.model)}_{run_ts}"
            f"_isl-{config.isl}_osl-{config.osl}"
            f"_maxcon-{config.max_concurrency}_n-{config.num_prompts}.json"
        )

        cmd, auth_token = build_vllm_bench_serve_argv(
            vllm_binary=self.vllm_binary,
            config=config,
            server=server,
            result_filename=result_filename,
        )

        env = dict(context.extra_env)
        if auth_token:
            env["OPENAI_API_KEY"] = auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        raw = load_json(result_filename) if rc == 0 else None
        return DriverResult(return_code=rc, raw=raw, raw_path=result_filename)


__all__ = ["VLLMBenchDriver", "build_vllm_bench_serve_argv"]
