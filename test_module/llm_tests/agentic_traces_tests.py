# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Orchestrator for the agentic trace-replay benchmark.

Bridges ``test_module`` to ``llm_module``: resolves the per-ModelSpec config,
expands it into runs for the requested :class:`AgenticTracesMode`, executes one
:class:`AIPerfAgenticTracesDriver` invocation per run, converts each payload into
a :class:`report_module.schema.Block` via
:class:`AIPerfAgenticTracesParser`, and forwards the Blocks to
``workflow_module.accept_blocks`` so the unified report generator picks them up.

Modeled on :mod:`test_module.llm_tests.prefix_cache_tests`. The one structural
difference is the timeout: a full-length agentic run profiles for an hour after
warming the cache, which comfortably exceeds the 2h default in
:class:`llm_module.config.DriverContext`, so the per-run timeout is derived from
the plan instead of left at the default.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from llm_module import ServerConnection
from llm_module.agentic_traces import (
    build_runs,
    summarize_runs,
    total_planned_seconds,
)
from llm_module.config import DriverContext
from llm_module.drivers.aiperf_agentic_traces import (
    AgenticTracesDriverResult,
    AIPerfAgenticTracesDriver,
)
from llm_module.parsers.aiperf_agentic_traces import AIPerfAgenticTracesParser
from llm_module.runner import RunnerResult
from llm_module.server_control import ServerController
from reference_config.agentic_traces.agentic_traces_config import (
    TraceSource,
    get_agentic_traces_config,
    resolve_run_specs,
)
from workflow_module import accept_blocks
from workflows.workflow_types import AgenticTracesMode

from .._test_common import report_model_fields
from ..context import MediaContext

logger = logging.getLogger(__name__)

# Head-room over the planned warmup + profiling window. Dataset download,
# tokenization of ~400 long traces, and the scenario's own configuration phase
# all happen before the clock starts, and AIPerf needs time to drain in-flight
# trajectories after it stops.
_TIMEOUT_HEADROOM_SECONDS = 3600


def run_agentic_traces(
    ctx: MediaContext,
    *,
    mode: str = "full",
    trace_sources: Optional[str] = None,
    duration_override: Optional[int] = None,
    git_ref_override: Optional[str] = None,
    metrics_urls: Sequence[str] = (),
    auth_token: str = "",
    venv_python: Optional[Path] = None,
    server_controller: Optional[ServerController] = None,
    output_subdir: str = "agentic_traces",
    inter_run_sleep_s: float = 2.0,
) -> RunnerResult:
    """Run the agentic trace-replay sweep end-to-end.

    Parameters
    ----------
    ctx:
        Workflow-engine :class:`MediaContext`. Provides ``model_spec``,
        ``device``, the server URL/port, and ``output_path``.
    mode:
        :class:`AgenticTracesMode` name (``full`` / ``ci``) selecting the
        duration profile from the model's config.
    trace_sources:
        Comma-separated :class:`TraceSource` names narrowing which configured
        runs execute. ``None`` runs every configured run.
    duration_override / git_ref_override:
        Ad-hoc overrides for the mode's ``benchmark_duration`` and the config's
        pinned InferenceX revision.
    metrics_urls:
        Extra Prometheus ``/metrics`` endpoints holding the prefix-cache
        counters, for a deployment whose load target does not expose them.
        AIPerf scrapes the load target's own ``/metrics`` either way.
    auth_token:
        Bearer token sent to the inference server. Empty disables auth.
    venv_python:
        Interpreter whose environment holds the InferenceX AIPerf fork. Falls
        back to ``sys.executable`` (correct for the launcher path, which already
        re-exec'd into the AGENTIC_TRACES venv).
    server_controller:
        Optional health protocol; polled before the sweep and between runs.

    Returns
    -------
    RunnerResult
        ``blocks`` holds one Block per successful run (kind ``agentic_traces``);
        ``return_codes`` records every planned run's exit code so a partial
        failure does not read as success.
    """
    result = RunnerResult()
    spec = ctx.model_spec

    config = get_agentic_traces_config(spec)
    if config is None:
        logger.error(
            "No agentic-traces config registered for model_id=%s. Add an entry to "
            "reference_config/agentic_traces/agentic_traces_config.py.",
            getattr(spec, "model_id", "<unknown>"),
        )
        result.return_codes.append(1)
        return result

    try:
        traces_mode = AgenticTracesMode.from_string(mode) or AgenticTracesMode.FULL
        selected_sources = _parse_trace_sources(trace_sources)
        effective_config, run_specs = resolve_run_specs(
            config,
            trace_sources=selected_sources,
            git_ref_override=git_ref_override,
        )
        runs = build_runs(
            effective_config,
            spec,
            mode=traces_mode,
            run_specs=run_specs,
            duration_override=duration_override,
        )
    except (ValueError, NotImplementedError) as exc:
        logger.error("[agentic-traces] Could not plan the sweep: %s", exc)
        result.return_codes.append(1)
        return result

    if not runs:
        logger.error(
            "[agentic-traces] No runs planned for model_id=%s mode=%s",
            getattr(spec, "model_id", "<unknown>"),
            mode,
        )
        result.return_codes.append(1)
        return result

    logger.info(summarize_runs(runs))
    logger.info(
        "[agentic-traces] InferenceX client pinned at %s",
        effective_config.inferencex_git_ref,
    )

    output_root = Path(ctx.output_path) / output_subdir
    artifact_root = output_root / "aiperf_artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_repo = getattr(spec, "hf_model_repo", "") or ""
    model_id = getattr(spec, "model_id", "") or model_repo
    device_label = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)

    driver = AIPerfAgenticTracesDriver(
        venv_python=Path(venv_python) if venv_python else Path(sys.executable),
        artifact_root=artifact_root,
        model_repo=model_repo,
        model_id=model_id,
        output_dir=output_root,
    )

    server = ServerConnection(
        base_url=ctx.server_host,
        service_port=ctx.server_port,
        model=model_repo,
        tokenizer=model_repo,
        auth_token=auth_token,
        # The scenario's own --tokenizer-trust-remote-code flag covers AIPerf,
        # but ServerConnection carries it too so the field stays consistent with
        # the other AIPerf drivers.
        tokenizer_trust_remote_code=any(
            run.tokenizer_trust_remote_code for run in runs
        ),
        prefix_cache_metrics_urls=tuple(metrics_urls or ()),
    )
    context = DriverContext(
        output_dir=output_root,
        device=device_label,
        # The request-bounded warmup has no wall-clock cap, so warmup_grace_period
        # is the allowance for it rather than a guarantee; the timeout backstops.
        per_run_timeout_s=float(
            max(run.benchmark_duration for run in runs)
            + max(run.warmup_grace_period for run in runs)
            + _TIMEOUT_HEADROOM_SECONDS
        ),
    )
    logger.info(
        "[agentic-traces] Per-run subprocess timeout: %.0fs (planned sweep "
        "wall-clock: %ds)",
        context.per_run_timeout_s,
        total_planned_seconds(runs),
    )

    if server_controller is not None and not server_controller.wait_for_healthy():
        logger.error("Inference server not healthy; aborting agentic-traces sweep.")
        result.return_codes.append(1)
        return result

    parser = AIPerfAgenticTracesParser()
    for i, run in enumerate(runs, 1):
        if server_controller is not None and not _server_still_healthy(
            server_controller, result
        ):
            break

        logger.info("[agentic-traces] Running %d/%d: %s", i, len(runs), run.label)
        if i > 1 and inter_run_sleep_s:
            time.sleep(inter_run_sleep_s)

        outcome: AgenticTracesDriverResult = driver.run(run, server, context)
        result.return_codes.append(outcome.return_code)
        if outcome.return_code != 0 or outcome.payload is None:
            logger.error(
                "[agentic-traces] %s failed (rc=%d); continuing.",
                run.label,
                outcome.return_code,
            )
            continue

        result.blocks.append(parser.parse(outcome.payload, device=device_label))

    if not result.blocks:
        logger.error("[agentic-traces] No blocks produced -- sweep had zero successes.")
        return result

    accept_blocks(
        result.blocks,
        envelope={
            **report_model_fields(spec),
            "device": device_label,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    logger.info(
        "[agentic-traces] Sweep complete: %d successful run(s) / %d planned (ok=%s)",
        len(result.blocks),
        len(runs),
        result.ok,
    )
    return result


def _parse_trace_sources(raw: Optional[str]):
    """Parse a comma-separated ``--agentic-traces-sources`` value."""
    if not raw:
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        return None
    return tuple(TraceSource.from_string(name) for name in names)


def _server_still_healthy(
    server_controller: ServerController, result: RunnerResult
) -> bool:
    """Health gate between runs; records a failure code when unhealthy."""
    try:
        health = server_controller.get_health()
    except Exception as exc:  # noqa: BLE001 -- log and abort
        logger.error("Health check raised: %s -- aborting sweep.", exc)
        result.return_codes.append(1)
        return False
    if getattr(health, "status_code", 200) != 200:
        logger.error(
            "Server unhealthy mid-sweep (status %s); aborting.",
            getattr(health, "status_code", "?"),
        )
        result.return_codes.append(1)
        return False
    return True


__all__ = ["run_agentic_traces"]
