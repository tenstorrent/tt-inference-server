# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""LLMPerformanceRunner.

Mirrors v1 ``benchmarking/run_benchmarks.py`` orchestration:

1. ``server.wait_for_healthy()`` — block until the inference server is up.
2. ``server.capture_traces(unique (isl,osl) pairs)`` — warm trace cache.
3. For each ``LLMRunConfig`` in the sweep: health-check, sleep 2 s,
   ``driver.run()``, ``driver.parse(raw)`` → ``Block``, then grade the
   Block against the sweep point's perf targets
   (:func:`llm_module.target_checks.apply_target_checks`).

Returns the list of Blocks plus any nonzero driver exit codes the
caller should surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import requests

from report_module.schema import Block

from .benchmark_configs import ensure_custom_dataset
from .config import DriverContext, LLMRunConfig, ServerConnection
from .drivers.base import LLMDriver
from .server_control import ServerController
from .target_checks import apply_target_checks

logger = logging.getLogger(__name__)


@dataclass
class RunnerResult:
    blocks: List[Block] = field(default_factory=list)
    return_codes: List[int] = field(default_factory=list)
    parse_failures: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            bool(self.return_codes)
            and all(rc == 0 for rc in self.return_codes)
            and not self.parse_failures
        )


class LLMPerformanceRunner:
    """Orchestrate one tool's run over a sweep of configs."""

    def __init__(
        self,
        driver: LLMDriver,
        server_controller: Optional[ServerController] = None,
        *,
        inter_run_sleep_s: float = 2.0,
        capture_trace_timeout_s: float = 1200.0,
        wait_healthy_timeout_s: float = 1200.0,
    ) -> None:
        self.driver = driver
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

            unique_lens = sorted({(c.isl, c.osl) for c in configs if c.isl and c.osl})
            if not skip_trace_capture and unique_lens:
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

            if self.driver.name == "vllm":
                cfg = ensure_custom_dataset(cfg, server, context.output_dir)
            repetitions = getattr(cfg, "repetitions", 1)
            full_warmup = getattr(cfg, "full_workload_warmup", False)
            if repetitions < 1:
                raise ValueError("Benchmark repetitions must be positive")
            phases = (["warmup"] if full_warmup else []) + [
                f"rep{rep}" for rep in range(1, repetitions + 1)
            ]
            for phase in phases:
                run_context = context
                if full_warmup or repetitions > 1:
                    run_context = replace(
                        context,
                        output_dir=context.output_dir / f"point-{i}" / phase,
                    )
                outcome = self.driver.run(cfg, server, run_context)
                result.return_codes.append(outcome.return_code)
                if outcome.return_code != 0 or outcome.raw is None:
                    logger.error(
                        "%s sweep point %d/%d %s failed (exit %d, raw=%s)",
                        self.driver.name,
                        i,
                        total,
                        phase,
                        outcome.return_code,
                        outcome.raw is not None,
                    )
                    if outcome.return_code == 0:
                        result.parse_failures.append(i)
                    if phase == "warmup":
                        # No measured repetition is valid without its full warmup.
                        break
                    continue
                if phase == "warmup":
                    continue

                block = self.driver.parse(outcome.raw, device=context.device)
                block = apply_target_checks(block, cfg)
                if repetitions > 1:
                    block = replace(
                        block,
                        id=f"{block.id or 'benchmark'}-point-{i}-{phase}",
                        title=f"{block.title or 'Benchmark'} — {phase}",
                        data={
                            **block.data,
                            "repetition": int(phase[3:]),
                            "raw_result": str(outcome.raw_path),
                        },
                    )
                result.blocks.append(block)

        return result
