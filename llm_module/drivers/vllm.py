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


def build_vllm_bench_serve_argv(
    *,
    vllm_binary: str,
    config: LLMRunConfig,
    server: ServerConnection,
    result_filename: Path,
    custom_dataset_path: Optional[Path] = None,
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
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        server.model,
        "--max-concurrency",
        str(config.max_concurrency),
        "--num-prompts",
        str(config.num_prompts),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--save-detailed",
        "--result-filename",
        str(result_filename),
    ]

    if custom_dataset_path is not None:
        # Pre-built exact-ISL prompts (SPEED-Bench text for block-granular
        # models): real language reaches the entropy halt, so the sweep
        # reports serving behaviour instead of the 48-step worst case that
        # random token salad pins every block to.
        cmd.extend(
            [
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(custom_dataset_path),
                "--custom-output-len",
                str(config.osl),
                "--disable-shuffle",
            ]
        )
    else:
        cmd.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(config.isl),
                "--random-output-len",
                str(config.osl),
            ]
        )

    is_remote_base_url = uses_remote_base_url(
        server.url_with_port,
        server.is_remote,
    )
    if server.tokenizer_trust_remote_code or is_remote_base_url:
        cmd.append("--trust-remote-code")

    if is_remote_base_url:
        cmd.extend(["--base-url", server.url_with_port])
        cmd.extend(["--ready-check-timeout-sec", "0"])
        if auth_token:
            headers.append(f"Authorization=Bearer {auth_token}")
    else:
        cmd.extend(["--host", server.host, "--port", str(server.service_port)])
        cmd.extend(
            [
                "--extra-body",
                json.dumps({"truncate_prompt_tokens": config.isl}),
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

        custom_dataset_path = self._maybe_speed_bench_prompts(config, server, context)
        cmd, auth_token = build_vllm_bench_serve_argv(
            vllm_binary=self.vllm_binary,
            config=config,
            server=server,
            result_filename=result_filename,
            custom_dataset_path=custom_dataset_path,
        )

        env = dict(context.extra_env)
        if auth_token:
            env["OPENAI_API_KEY"] = auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        raw = load_json(result_filename) if rc == 0 else None
        if raw is not None and server.output_block_size > 1:
            raw["tt_output_block_size"] = server.output_block_size
        return DriverResult(return_code=rc, raw=raw, raw_path=result_filename)

    @staticmethod
    def _maybe_speed_bench_prompts(
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> Optional[Path]:
        """Exact-ISL SPEED-Bench prompt file for block-granular sweeps.

        DiffusionGemma denoises whole 256-token canvases and halts on entropy;
        random-token prompts never halt, so the sweep would only ever measure
        the 48-step cap. Falls back to the random dataset with a loud warning
        when the prompts cannot be built (offline host, gated tokenizer), so a
        benchmark run still produces rows.
        """
        if "diffusiongemma" not in server.model.lower():
            return None
        from ..speed_bench_prompts import write_speed_bench_prompt_file

        prompts_path = context.output_dir / (
            f"speed_bench_prompts_isl-{config.isl}_n-{config.num_prompts}.jsonl"
        )
        try:
            return write_speed_bench_prompt_file(
                output_path=prompts_path,
                model=server.model,
                target_isl=config.isl,
                num_prompts=config.num_prompts,
                trust_remote_code=server.tokenizer_trust_remote_code,
            )
        except Exception as build_error:  # noqa: BLE001 - fail open to random
            logger.warning(
                "falling back to --dataset-name random for %s isl=%d: "
                "SPEED-Bench prompt construction failed: %r",
                server.model,
                config.isl,
                build_error,
            )
            return None


__all__ = ["VLLMBenchDriver", "build_vllm_bench_serve_argv"]
