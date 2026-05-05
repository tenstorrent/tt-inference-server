# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""LLMPerformanceRunner.

Mirrors v1 ``benchmarking/run_benchmarks.py`` orchestration:

1. ``server.wait_for_healthy()`` — block until the inference server is up.
2. ``server.capture_traces(unique (isl,osl) pairs)`` — warm trace cache.
3. For each ``LLMRunConfig`` in the sweep: health-check, sleep 2 s,
   ``driver.run()``, ``parser.parse(raw)`` → ``Block``.

Returns the list of Blocks plus any nonzero driver exit codes the
caller should surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import requests

from report_module.schema import Block

from .config import DriverContext, LLMRunConfig, ServerConnection
from .drivers.base import LLMDriver
from .parsers.base import LLMResultParser
from .server_control import ServerController

logger = logging.getLogger(__name__)


@dataclass
class RunnerResult:
    blocks: List[Block] = field(default_factory=list)
    return_codes: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.return_codes) and all(rc == 0 for rc in self.return_codes)


class LLMPerformanceRunner:
    """Orchestrate one tool's run over a sweep of configs."""

    def __init__(
        self,
        driver: LLMDriver,
        parser: LLMResultParser,
        server_controller: Optional[ServerController] = None,
        *,
        inter_run_sleep_s: float = 2.0,
        capture_trace_timeout_s: float = 1200.0,
        wait_healthy_timeout_s: float = 1200.0,
    ) -> None:
        self.driver = driver
        self.parser = parser
        self.server_controller = server_controller
        self.inter_run_sleep_s = inter_run_sleep_s
        self.capture_trace_timeout_s = capture_trace_timeout_s
        self.wait_healthy_timeout_s = wait_healthy_timeout_s

    def run(
        self,
        configs: Sequence[LLMRunConfig],
        server: ServerConnection,
        context: DriverContext,
        *,
        skip_trace_capture: bool = False,
    ) -> RunnerResult:
        result = RunnerResult()
        if not configs:
            logger.warning("LLMPerformanceRunner.run called with zero configs")
            return result

        if self.server_controller is not None:
            if not self.server_controller.wait_for_healthy(
                timeout=self.wait_healthy_timeout_s
            ):
                logger.error("Inference server not healthy; aborting sweep.")
                result.return_codes.append(1)
                return result

            if not skip_trace_capture:
                unique_lens = sorted({(c.isl, c.osl) for c in configs})
                try:
                    self.server_controller.capture_traces(
                        context_lens=unique_lens,
                        timeout=self.capture_trace_timeout_s,
                    )
                except Exception as exc:
                    logger.warning("Trace capture failed (continuing): %s", exc)

        total = len(configs)
        for i, cfg in enumerate(configs, 1):
            if self.server_controller is not None:
                try:
                    health = self.server_controller.get_health()
                    if getattr(health, "status_code", 200) != 200:
                        logger.error(
                            "Server unhealthy mid-sweep (status %s); aborting.",
                            getattr(health, "status_code", "?"),
                        )
                        result.return_codes.append(1)
                        break
                except requests.exceptions.RequestException as exc:
                    logger.error("Health check raised: %s — aborting sweep.", exc)
                    result.return_codes.append(1)
                    break

            logger.info(
                "Running %s sweep point %d/%d  isl=%d osl=%d max_conc=%d n=%d",
                self.driver.name,
                i,
                total,
                cfg.isl,
                cfg.osl,
                cfg.max_concurrency,
                cfg.num_prompts,
            )
            if i > 1 and self.inter_run_sleep_s:
                time.sleep(self.inter_run_sleep_s)

            outcome = self.driver.run(cfg, server, context)
            result.return_codes.append(outcome.return_code)
            if outcome.return_code != 0:
                logger.error(
                    "%s exited %d on sweep point %d/%d",
                    self.driver.name,
                    outcome.return_code,
                    i,
                    total,
                )
                continue
            if outcome.raw is None:
                logger.error(
                    "%s sweep point %d/%d produced no parseable raw result",
                    self.driver.name,
                    i,
                    total,
                )
                continue

            block = self.parser.parse(outcome.raw, device=context.device)
            result.blocks.append(block)

        return result
