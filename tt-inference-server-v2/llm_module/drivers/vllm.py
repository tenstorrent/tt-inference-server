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

# sonnet's shared-prefix budget. It is subtracted from --sonnet-input-len, and the dataset raises
# when the remainder cannot hold the base prompt, so points shorter than this get no prefix.
SONNET_PREFIX_LEN = 200
SONNET_MIN_ISL_FOR_PREFIX = 512


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

    # Prompt source. The default is `random`, which fabricates token ids: fine for a model whose
    # per-token cost does not depend on what the tokens say, and WRONG for a block-diffusion model
    # like DiffusionGemma, whose canvas never settles on gibberish -- measured on QB2, random ids
    # give 48/48 denoise steps and a 4.3% halt rate versus K~15-18 and 100% on real text, i.e. a
    # ~3x understatement of tok/s that no real traffic ever produces. `random` also flips
    # ignore_eos on inside vllm bench serve, so nothing can stop early either.
    #
    # A model opts into real text via spec metadata (benchmark_dataset_name/path). `sonnet` is the
    # option that keeps this sweep comparable: it assembles real English from a text file to hit
    # --sonnet-input-len, so the six (ISL, OSL) points keep their meaning. It ASSEMBLES WHOLE LINES,
    # so it undershoots the target by up to ~1 line (measured: 8192 -> 8125 raw, aligning to 8160),
    # and truncate_prompt_tokens can only cut from above -- the model's warmed prefill lengths must
    # therefore cover a band BELOW each ISL or up-front capture rejects the request and returns an
    # empty answer with a plausible-looking throughput attached.
    if server.benchmark_dataset_name == "sonnet":
        if not server.benchmark_dataset_path:
            raise ValueError(
                "benchmark_dataset_name=sonnet requires benchmark_dataset_path"
            )
        # prefix_len is a shared prefix taken out of the input budget; sonnet raises when it does
        # not leave room, so short points get none.
        prefix_len = 0 if config.isl < SONNET_MIN_ISL_FOR_PREFIX else SONNET_PREFIX_LEN
        cmd.extend(
            [
                "--dataset-name",
                "sonnet",
                "--dataset-path",
                str(server.benchmark_dataset_path),
                "--sonnet-input-len",
                str(config.isl),
                "--sonnet-output-len",
                str(config.osl),
                "--sonnet-prefix-len",
                str(prefix_len),
            ]
        )
    elif server.benchmark_dataset_name:
        raise ValueError(
            f"unsupported benchmark_dataset_name {server.benchmark_dataset_name!r}; "
            "supported: 'sonnet' (or empty for the default random dataset)"
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

    # ``vllm bench serve`` loads the tokenizer itself -- the random/sonnet
    # datasets are token-length-driven, so prompts are encoded client-side. A
    # model whose HF repo ships custom tokenizer code (e.g.
    # google/diffusiongemma-26B-A4B-it, whose spec metadata sets
    # tokenizer_trust_remote_code) then cannot be loaded without this flag, and
    # the failure is an AutoTokenizer raise before the first request, i.e. the
    # whole sweep. The flag was previously tied to the remote-console branch, so
    # a local server benchmarking such a model never received it.
    is_remote_base_url = uses_remote_base_url(server.url_with_port, server.is_remote)
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
