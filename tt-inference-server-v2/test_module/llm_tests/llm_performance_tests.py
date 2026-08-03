# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""LLM performance test caller.

Bridges ``test_module`` to ``llm_module``: builds an
``LLMPerformanceRunner`` from a (driver, server_controller) pair,
executes the sweep defined by ``configs``, and forwards the resulting
``list[Block]`` to ``workflow_module`` for downstream processing
(report rendering, artifact upload, etc.). The driver carries its own
parser, so command-build, execute, and parse stay selected as one unit.

The caller is the only place in test_module that knows about
llm_module's internals; everything else (drivers, runner
orchestration) stays inside llm_module.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from llm_module import (
    DriverContext,
    HttpServerController,
    LLMDriver,
    LLMPerformanceRunner,
    LLMRunConfig,
    RemoteOpenAIController,
    ServerConnection,
    ServerController,
)
from llm_module.runner import RunnerResult
from workflow_module import accept_blocks

from ..context import MediaContext

logger = logging.getLogger(__name__)

# <repo>/tt-inference-server-v2/test_module/llm_tests/this_file.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]


def run_llm_performance(
    ctx: MediaContext,
    *,
    driver: LLMDriver,
    configs: Sequence[LLMRunConfig],
    server_controller: Optional[ServerController] = None,
    output_subdir: str = "llm",
    auth_token: str = "",
    goodput: Optional[str] = None,
) -> RunnerResult:
    """Run an LLM perf sweep and forward the Blocks to workflow_module.

    Returns the :class:`RunnerResult` so callers see per-sweep-point exit
    codes (``return_codes``/``ok``), not just the Blocks — a partial sweep
    failure must not read as success.

    ``auth_token`` is sent to the inference server (e.g. a minted JWT
    exported as the bearer token); empty string disables auth.
    """
    server_base_url = ctx.server_url if ctx.remote_server else ctx.server_host
    # Same opt-in as the prefix-cache / spec-decode paths: the perf drivers
    # tokenize client-side, so a model whose HF repo ships a custom tokenizer
    # (Kimi, DiffusionGemma) must declare tokenizer_trust_remote_code in its
    # spec metadata rather than us executing Hub code for every model.
    tokenizer_trust_remote_code = bool(
        getattr(ctx.model_spec, "metadata", {}).get(
            "tokenizer_trust_remote_code", False
        )
    )
    # Real-text prompts, opt-in per model. See ServerConnection: the default `random` dataset
    # fabricates token ids, which for a block-diffusion model measures a regime real traffic never
    # produces. The path is resolved against the repo so it ships with the image.
    metadata = getattr(ctx.model_spec, "metadata", {}) or {}
    benchmark_dataset_name = str(metadata.get("benchmark_dataset_name", "") or "")
    benchmark_dataset_path = str(metadata.get("benchmark_dataset_path", "") or "")
    if benchmark_dataset_name and not benchmark_dataset_path:
        raise ValueError(
            f"model {ctx.model_spec.model_name!r} sets benchmark_dataset_name="
            f"{benchmark_dataset_name!r} without benchmark_dataset_path"
        )
    if benchmark_dataset_path and not Path(benchmark_dataset_path).is_absolute():
        benchmark_dataset_path = str((REPO_ROOT / benchmark_dataset_path).resolve())
    if benchmark_dataset_path and not Path(benchmark_dataset_path).exists():
        raise FileNotFoundError(
            f"benchmark_dataset_path {benchmark_dataset_path!r} does not exist; the sweep would "
            "fail on its first request after the server is already up"
        )
    server = ServerConnection(
        base_url=server_base_url,
        service_port=ctx.server_port,
        model=ctx.model_spec.hf_model_repo,
        auth_token=auth_token,
        is_remote=ctx.remote_server,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
        benchmark_dataset_name=benchmark_dataset_name,
        benchmark_dataset_path=benchmark_dataset_path,
    )
    output_dir = Path(ctx.output_path) / output_subdir
    device_label = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    context = DriverContext(output_dir=output_dir, device=device_label, goodput=goodput)

    if server_controller is None:
        if ctx.remote_server:
            server_controller = RemoteOpenAIController(
                base_url=server.url_with_port,
                auth_token=auth_token,
            )
        else:
            server_controller = HttpServerController(
                base_url=ctx.server_host,
                service_port=ctx.server_port,
                auth_token=auth_token,
            )

    runner = LLMPerformanceRunner(
        driver=driver,
        server_controller=server_controller,
    )
    result = runner.run(configs, server, context)

    if result.return_codes and not result.ok:
        logger.warning(
            "LLM sweep finished with non-zero exits: %s", result.return_codes
        )
    else:
        logger.info(
            "LLM sweep produced %d Block(s) over %d sweep point(s)",
            len(result.blocks),
            len(result.return_codes),
        )

    accept_blocks(
        result.blocks,
        envelope={
            "model_name": server.model,
            "device": device_label,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return result


__all__ = ["run_llm_performance"]
