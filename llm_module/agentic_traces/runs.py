# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Agentic trace-replay run expansion.

Resolves a per-ModelSpec :class:`AgenticTracesConfig` plus a mode into concrete
:class:`AgenticTracesRun` objects, each of which is exactly one ``aiperf
profile`` invocation. Values that the model catalog already knows
(``max_context_length``, ``tokenizer_trust_remote_code``) are filled in from the
``ModelSpec`` here rather than duplicated in the config.

The driver stays dumb about modes and trace sources: by the time it sees a run,
every knob is a concrete value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reference_config.agentic_traces.agentic_traces_config import (
    AgenticTracesConfig,
    AgenticTracesRunSpec,
    TraceSource,
)
from workflows.workflow_types import AgenticTracesMode

# Trace sources with a working client. SWARMONE is a registered config value
# with no harness yet, so it fails loudly at plan time instead of producing an
# empty (silently passing) sweep.
SUPPORTED_TRACE_SOURCES: Tuple[TraceSource, ...] = (TraceSource.INFERENCEX_AGENTX,)


@dataclass(frozen=True)
class AgenticTracesRun:
    """One fully-resolved ``aiperf profile`` agentic-trace invocation."""

    trace_source: TraceSource
    label: str
    scenario: str
    public_dataset: str
    endpoint: str
    endpoint_type: str
    streaming: bool
    concurrency: int
    benchmark_duration: int
    warmup_requests_per_lane: int
    warmup_grace_period: int
    num_dataset_entries: int
    random_seed: int
    failed_request_threshold: float
    trajectory_start_min_ratio: float
    trajectory_start_max_ratio: float
    slice_duration: float
    max_context_length: int
    tokenizer_trust_remote_code: bool
    use_server_token_count: bool
    gpu_telemetry: bool
    mode: AgenticTracesMode
    env: Dict[str, str] = field(default_factory=dict)
    # Free-form provenance echoed into the result payload for the report.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def filesafe_label(self) -> str:
        return self.label.replace("/", "_").replace(" ", "_")


def _resolve_max_context_length(run_spec: AgenticTracesRunSpec, model_spec) -> int:
    """Explicit override, else the spec's served context window."""
    if run_spec.max_context_length is not None:
        return run_spec.max_context_length
    device_model_spec = getattr(model_spec, "device_model_spec", None)
    max_context = getattr(device_model_spec, "max_context", None)
    if not max_context:
        raise ValueError(
            "Cannot resolve max_context_length for agentic traces: the run spec "
            "leaves it unset and the ModelSpec has no device_model_spec."
            "max_context. Set max_context_length explicitly in the config."
        )
    return int(max_context)


def _resolve_tokenizer_trust_remote_code(
    run_spec: AgenticTracesRunSpec, model_spec
) -> bool:
    """Explicit override, else the catalog's per-weight metadata flag."""
    if run_spec.tokenizer_trust_remote_code is not None:
        return run_spec.tokenizer_trust_remote_code
    metadata = getattr(model_spec, "metadata", None) or {}
    return bool(metadata.get("tokenizer_trust_remote_code", False))


def build_runs(
    config: AgenticTracesConfig,
    model_spec,
    *,
    mode: AgenticTracesMode = AgenticTracesMode.FULL,
    run_specs: Optional[Sequence[AgenticTracesRunSpec]] = None,
    duration_override: Optional[int] = None,
) -> List[AgenticTracesRun]:
    """Expand ``config`` into concrete runs for ``mode``.

    ``run_specs`` narrows to a subset (e.g. after a ``--agentic-traces-sources``
    filter); it defaults to every run in the config. ``duration_override``
    replaces the mode's ``benchmark_duration`` for ad-hoc runs.

    Raises ``NotImplementedError`` for a configured-but-unimplemented trace
    source so an unsupported selection can never look like a clean sweep.
    """
    settings = config.settings_for_mode(mode)
    specs = tuple(run_specs) if run_specs is not None else config.runs

    unsupported = sorted(
        {
            spec.trace_source.value
            for spec in specs
            if spec.trace_source not in SUPPORTED_TRACE_SOURCES
        }
    )
    if unsupported:
        raise NotImplementedError(
            f"Agentic trace source(s) {unsupported} are registered in the config "
            f"schema but have no client integration yet. Supported today: "
            f"{sorted(s.value for s in SUPPORTED_TRACE_SOURCES)}."
        )

    benchmark_duration = (
        duration_override
        if duration_override is not None
        else settings.benchmark_duration
    )

    runs: List[AgenticTracesRun] = []
    for spec in specs:
        concurrency = settings.concurrency or spec.concurrency
        runs.append(
            AgenticTracesRun(
                trace_source=spec.trace_source,
                label=f"{spec.label}_{mode.to_string()}",
                scenario=spec.scenario,
                public_dataset=spec.public_dataset,
                endpoint=spec.endpoint,
                endpoint_type=spec.endpoint_type,
                streaming=spec.streaming,
                concurrency=concurrency,
                benchmark_duration=benchmark_duration,
                warmup_requests_per_lane=settings.warmup_requests_per_lane,
                warmup_grace_period=settings.warmup_grace_period,
                num_dataset_entries=settings.num_dataset_entries,
                random_seed=spec.random_seed,
                failed_request_threshold=spec.failed_request_threshold,
                trajectory_start_min_ratio=spec.trajectory_start_min_ratio,
                trajectory_start_max_ratio=spec.trajectory_start_max_ratio,
                slice_duration=spec.slice_duration,
                max_context_length=_resolve_max_context_length(spec, model_spec),
                tokenizer_trust_remote_code=_resolve_tokenizer_trust_remote_code(
                    spec, model_spec
                ),
                use_server_token_count=spec.use_server_token_count,
                gpu_telemetry=spec.gpu_telemetry,
                mode=mode,
                env=dict(spec.env),
                metadata={
                    "model_id": getattr(model_spec, "model_id", ""),
                    "inferencex_git_ref": config.inferencex_git_ref,
                    "mode": mode.to_string(),
                },
            )
        )
    return runs


def summarize_runs(runs: Sequence[AgenticTracesRun]) -> str:
    """One-line-per-run plan summary for the run log."""
    if not runs:
        return "[agentic-traces] No runs planned."
    lines = [f"[agentic-traces] Planned {len(runs)} run(s):"]
    for run in runs:
        lines.append(
            f"  - {run.label}: source={run.trace_source.value} "
            f"scenario={run.scenario} dataset={run.public_dataset} "
            f"concurrency={run.concurrency} duration={run.benchmark_duration}s "
            f"warmup={run.warmup_requests_per_lane}req/lane "
            f"entries={run.num_dataset_entries} "
            f"max_context={run.max_context_length}"
        )
    return "\n".join(lines)


def total_planned_seconds(runs: Sequence[AgenticTracesRun]) -> int:
    """Sum of every run's profiling window plus its warmup allowance.

    Used to size the per-run subprocess timeout: the default 2h driver timeout
    is shorter than a single full-length agentic run plus its warmup.

    A request-bounded warmup has no wall-clock cap -- ``warmup_requests_per_lane``
    requests take as long as the server needs -- so this is an allowance, not a
    bound. ``warmup_grace_period`` supplies it: at 1800s it is roughly 3x the
    583.7s the validated run spent warming up, and the subprocess timeout is the
    backstop if a degraded server exceeds even that.
    """
    return sum(run.benchmark_duration + run.warmup_grace_period for run in runs)


__all__ = [
    "SUPPORTED_TRACE_SOURCES",
    "AgenticTracesRun",
    "build_runs",
    "summarize_runs",
    "total_planned_seconds",
]
