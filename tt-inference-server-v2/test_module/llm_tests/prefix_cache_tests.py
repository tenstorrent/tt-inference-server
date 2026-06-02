# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Orchestrator for the AIPerf prefix-cache benchmark sweep.

Bridges ``test_module`` to ``llm_module``: builds a scenario plan from
``llm_module.prefix_cache``, runs one
:class:`AIPerfPrefixCacheDriver` invocation per
:class:`PrefixCacheRun`, converts each driver payload into a
:class:`report_module.schema.Block` via
:class:`AIPerfPrefixCacheParser`, and forwards the Blocks to
``workflow_module.accept_blocks`` so the unified report generator picks
them up alongside any other workflow output.

Orchestrates the full prefix-cache scenario sweep for v2 benchmarks.
loop, minus the v1-specific PromptClient bootstrap -- v2's
:class:`ServerController` protocol replaces it.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from llm_module import ServerConnection
from llm_module.config import DriverContext
from llm_module.drivers.aiperf_prefix_cache import (
    AIPerfPrefixCacheDriver,
    PrefixCacheDriverResult,
)
from llm_module.parsers.aiperf_prefix_cache import AIPerfPrefixCacheParser
from llm_module.prefix_cache import (
    PrefixCacheRun,
    build_runs as build_prefix_cache_runs,
    summarize_runs as summarize_prefix_cache_runs,
)
from llm_module.server_control import ServerController
from report_module.schema import Block
from workflow_module import accept_blocks

from ..context import MediaContext

logger = logging.getLogger(__name__)

# Scenarios run in this order so reuse scenarios benefit from a fresh
# but warm starting point. Trace-driven mooncake_trace is heaviest, runs
# last.
_SCENARIO_ORDER = {
    "baseline": 0,
    "shared_system": 1,
    "prefix_pool": 2,
    "multi_turn": 3,
    "mooncake_trace": 4,
}


def run_prefix_cache(
    ctx: MediaContext,
    *,
    preset: str = "full",
    scenarios: Optional[str] = None,
    arrival_pattern: Optional[str] = None,
    request_rate: Optional[float] = None,
    scenarios_json: Optional[str] = None,
    trace_path: Optional[str] = None,
    auth_token: str = "",
    venv_python: Optional[Path] = None,
    server_controller: Optional[ServerController] = None,
    output_subdir: str = "prefix_cache",
    inter_run_sleep_s: float = 2.0,
) -> List[Block]:
    """Run the prefix-cache sweep end-to-end.

    Parameters
    ----------
    ctx:
        v2 :class:`MediaContext` (same one ``run.py`` uses for the
        media workflows). Provides ``model_spec``, ``device``,
        ``service_port``, and ``output_path``.
    preset / scenarios / arrival_pattern / request_rate / scenarios_json / trace_path:
        Forwarded to :func:`llm_module.prefix_cache.build_runs`. See its
        docstring for the contract.
    auth_token:
        Bearer token sent to the inference server (JWT, OPENAI_API_KEY).
        Empty string disables auth.
    venv_python:
        Python interpreter that has ``aiperf`` installed. Falls back to
        ``sys.executable``.
    server_controller:
        Optional server-health protocol. When supplied the orchestrator
        polls health before each run and aborts on an unhealthy status.
    output_subdir:
        Sub-directory of ``ctx.output_path`` where the per-run JSONs and
        AIPerf artifacts go.
    inter_run_sleep_s:
        Seconds to sleep between runs (matches v1 behavior).

    Returns
    -------
    list[Block]
        One Block per successful run (kind ``aiperf_prefix_cache``).
        The same Blocks are also forwarded to
        :func:`workflow_module.accept_blocks` so the unified report
        generator picks them up.
    """
    runs = build_prefix_cache_runs(
        preset=preset,
        scenarios=scenarios,
        arrival_pattern=arrival_pattern,
        request_rate=request_rate,
        manifest_path=Path(scenarios_json) if scenarios_json else None,
        trace_path_override=trace_path,
    )
    if not runs:
        logger.error(
            "No prefix-cache runs produced by preset=%s scenarios=%s",
            preset,
            scenarios,
        )
        return []

    logger.info(summarize_prefix_cache_runs(runs))

    output_root = Path(ctx.output_path) / output_subdir
    artifact_root = output_root / "aiperf_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    spec = ctx.model_spec
    model_repo = getattr(spec, "hf_model_repo", "") or ""
    model_id = getattr(spec, "model_id", "") or model_repo
    device_label = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)

    driver = AIPerfPrefixCacheDriver(
        venv_python=Path(venv_python) if venv_python else Path(sys.executable),
        artifact_root=artifact_root,
        model_repo=model_repo,
        model_id=model_id,
        tokenizer=model_repo,
        output_dir=output_root,
    )

    # One analyze-trace per unique (trace, block_size) before the loop.
    driver.prepare_trace_analyses(runs)

    server = ServerConnection(
        base_url="http://localhost",
        service_port=ctx.service_port,
        model=model_repo,
        tokenizer=model_repo,
        auth_token=auth_token,
    )
    context = DriverContext(output_dir=output_root, device=device_label)

    if server_controller is not None and not server_controller.wait_for_healthy():
        logger.error("Inference server not healthy; aborting prefix-cache sweep.")
        return []

    parser = AIPerfPrefixCacheParser()
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            _SCENARIO_ORDER.get(r.scenario, 99),
            r.scenario,
            r.concurrency,
            r.label,
        ),
    )

    blocks: List[Block] = []
    for i, run in enumerate(runs_sorted, 1):
        if server_controller is not None:
            try:
                health = server_controller.get_health()
                if getattr(health, "status_code", 200) != 200:
                    logger.error(
                        "Server unhealthy mid-sweep (status %s); aborting.",
                        getattr(health, "status_code", "?"),
                    )
                    break
            except Exception as exc:  # noqa: BLE001 -- log and abort
                logger.error("Health check raised: %s -- aborting sweep.", exc)
                break

        logger.info(
            "[prefix-cache] Running %d/%d: %s/%s",
            i,
            len(runs_sorted),
            run.scenario,
            run.label,
        )
        if i > 1 and inter_run_sleep_s:
            time.sleep(inter_run_sleep_s)

        outcome: PrefixCacheDriverResult = driver.run(run, server, context)
        if outcome.return_code != 0 or outcome.payload is None:
            logger.error(
                "[prefix-cache] %s/%s failed (rc=%d); continuing.",
                run.scenario,
                run.label,
                outcome.return_code,
            )
            continue

        blocks.append(parser.parse(outcome.payload, device=device_label))

    if not blocks:
        logger.error("[prefix-cache] No blocks produced -- sweep had zero successes.")
        return []

    accept_blocks(
        blocks,
        envelope={
            "model_name": getattr(spec, "model_name", "") or model_repo,
            "device": device_label,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    logger.info(
        "[prefix-cache] Sweep complete: %d successful run(s) / %d planned",
        len(blocks),
        len(runs_sorted),
    )
    return blocks


__all__ = ["run_prefix_cache"]


# Type-check the orchestrator builds against ``Sequence[PrefixCacheRun]``
# at static-analysis time. Suppressed at runtime to avoid importing
# something heavy just for a hint.
_: Optional[Sequence[PrefixCacheRun]] = None
