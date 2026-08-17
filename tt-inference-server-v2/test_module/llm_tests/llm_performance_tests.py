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
from typing import List, Optional, Sequence

from llm_module import (
    DriverContext,
    LLMDriver,
    LLMPerformanceRunner,
    LLMRunConfig,
    ServerConnection,
    ServerController,
)
from report_module.schema import Block
from workflow_module import accept_blocks

from ..context import MediaContext

logger = logging.getLogger(__name__)


def run_llm_performance(
    ctx: MediaContext,
    *,
    driver: LLMDriver,
    configs: Sequence[LLMRunConfig],
    server_controller: Optional[ServerController] = None,
    output_subdir: str = "llm",
) -> List[Block]:
    """Run an LLM perf sweep and forward the Blocks to workflow_module.

    Returns the list of Blocks produced by the runner. The same list
    is also handed to ``workflow_module.accept_blocks`` so the
    downstream workflow can act on it. Callers can ignore the return
    if they don't need the in-memory copy.
    """
    server = ServerConnection(
        base_url="http://localhost",
        service_port=ctx.service_port,
        model=ctx.model_spec.hf_model_repo,
    )
    output_dir = Path(ctx.output_path) / output_subdir
    device_label = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    context = DriverContext(output_dir=output_dir, device=device_label)

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
    return result.blocks


__all__ = ["run_llm_performance"]
