# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""LLM performance test caller.

Bridges ``test_module`` to ``llm_module``: builds an
``LLMPerformanceRunner`` from a (driver, server_controller) pair,
executes the sweep defined by ``configs``, and forwards each resulting
``Block`` to ``workflow_module`` as it is produced -- each accept
checkpoints the report (``WorkflowExecution``'s accumulator hook), so a
sweep killed mid-flight still leaves a report for the points that finished. The driver carries its own parser, so
command-build, execute, and parse stay selected as one unit.

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

from .._test_common import report_model_fields
from ..context import MediaContext

logger = logging.getLogger(__name__)


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
    metadata = getattr(ctx.model_spec, "metadata", {}) or {}
    tokenizer_trust_remote_code = bool(
        metadata.get("tokenizer_trust_remote_code", False)
    )

    server = ServerConnection(
        base_url=server_base_url,
        service_port=ctx.server_port,
        model=ctx.model_spec.hf_model_repo,
        auth_token=auth_token,
        is_remote=ctx.remote_server,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
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

    # Built before the sweep: `generated_at` is recorded once and synthesises
    # the report_id that names the report files, so the checkpoints and the
    # final report overwrite one another.
    envelope = {
        **report_model_fields(ctx.model_spec),
        "device": device_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    def _persist(block) -> None:
        accept_blocks([block], envelope=envelope)

    result = runner.run(configs, server, context, on_block=_persist)

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

    return result


__all__ = ["run_llm_performance"]
