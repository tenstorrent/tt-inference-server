# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Per-ModelSpec configuration for the ``agentic_traces`` workflow.

Mirrors the authoring style of :mod:`reference_config.evals.eval_config`, with
one deliberate difference: entries are keyed by ``ModelSpec.model_id`` rather
than ``hf_model_repo``. Agentic trace replay is sensitive to the served context
window and to the impl/device the weights run on, so two specs for the same
weights can legitimately need different datasets, concurrency, or a different
pinned InferenceX revision.

Unlike ``EVAL_CONFIGS`` this registry is NOT intersected with ``MODEL_SPECS``:
the catalog that gets loaded depends on ``MODEL_SPECS_ENV`` (``--dev-mode``),
and filtering would silently drop configs for dev-only models. Typos surface
instead at lookup time, where the workflow refuses to run for a model with no
config, and in ``tests/reference_config/test_agentic_traces_config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Tuple

from workflows.utils import map_configs_by_attr
from workflows.workflow_types import AgenticTracesMode

# The InferenceX ``inferencex-agentx-mvp`` scenario rejects a profiling window
# shorter than this (see its scenario definition in the InferenceX repo:
# utils/aiperf/src/aiperf/common/scenario/inferencex_agentx_mvp.py). It is the
# floor for CI mode too -- there is no shorter "smoke" shape available.
AGENTIC_TRACES_MIN_PROFILE_SECONDS = 900

# Scenarios known to enforce AGENTIC_TRACES_MIN_PROFILE_SECONDS.
_MIN_DURATION_SCENARIOS = frozenset({"inferencex-agentx-mvp"})


class TraceSource(Enum):
    """Where a run's agentic traces come from.

    ``INFERENCEX_AGENTX`` replays the SemiAnalysis Weka coding traces through
    the AIPerf fork vendored in the InferenceX repo. ``SWARMONE`` is registered
    so configs and CLI selection can name it, but building runs for it raises
    ``NotImplementedError`` until its harness is integrated.
    """

    INFERENCEX_AGENTX = "inferencex_agentx"
    SWARMONE = "swarmone"

    @classmethod
    def from_string(cls, name: str) -> "TraceSource":
        key = name.strip().upper().replace("-", "_")
        try:
            return cls[key]
        except KeyError:
            valid = ", ".join(sorted(m.value for m in cls))
            raise ValueError(f"Invalid TraceSource: {name!r}. Valid: {valid}")


@dataclass(frozen=True)
class AgenticTracesRunSpec:
    """One ``aiperf profile`` invocation, minus the duration knobs.

    Everything that does not change between a CI run and a full run lives here;
    the wall-clock knobs live in :class:`AgenticTracesModeSettings` so a single
    run spec can be replayed at either length.

    ``max_context_length`` and ``tokenizer_trust_remote_code`` default to
    ``None`` meaning "derive from the ModelSpec" (``device_model_spec.max_context``
    and ``metadata.tokenizer_trust_remote_code`` respectively), so they cannot
    drift away from the catalog. Set them explicitly only to override.
    """

    trace_source: TraceSource = TraceSource.INFERENCEX_AGENTX
    scenario: str = "inferencex-agentx-mvp"
    public_dataset: str = "semianalysis_cc_traces_weka_062126_256k"
    endpoint: str = "/v1/chat/completions"
    endpoint_type: str = "chat"
    streaming: bool = True
    concurrency: int = 8
    random_seed: int = 42
    failed_request_threshold: float = 0.10
    trajectory_start_min_ratio: float = 0.25
    trajectory_start_max_ratio: float = 0.75
    slice_duration: float = 1.0
    max_context_length: Optional[int] = None
    tokenizer_trust_remote_code: Optional[bool] = None
    use_server_token_count: bool = True
    gpu_telemetry: bool = False
    # AIPERF_* process env. These are read by AIPerf itself, not passed as
    # flags: the two timeouts cover dataset download plus per-service profile
    # configuration for the ~400-trace Weka datasets, the WEKA_LIVE toggle
    # replays recorded assistant turns instead of re-generating them, and the
    # TCP user timeout keeps long idle agentic turns from being reaped.
    env: Dict[str, str] = field(
        default_factory=lambda: {
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
            "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES": "0",
            "AIPERF_HTTP_TCP_USER_TIMEOUT": "900000",
        }
    )

    def __post_init__(self) -> None:
        if not self.scenario.strip():
            raise ValueError("AgenticTracesRunSpec.scenario must not be empty")
        if self.concurrency < 1:
            raise ValueError(
                f"AgenticTracesRunSpec.concurrency must be >= 1, got {self.concurrency}"
            )
        if not 0.0 <= self.failed_request_threshold <= 1.0:
            raise ValueError(
                "AgenticTracesRunSpec.failed_request_threshold must be a "
                f"fraction in [0, 1], got {self.failed_request_threshold}"
            )
        if not (
            0.0
            <= self.trajectory_start_min_ratio
            <= self.trajectory_start_max_ratio
            <= 1.0
        ):
            raise ValueError(
                "AgenticTracesRunSpec trajectory ratios must satisfy "
                "0 <= min <= max <= 1, got min="
                f"{self.trajectory_start_min_ratio} max="
                f"{self.trajectory_start_max_ratio}"
            )

    @property
    def label(self) -> str:
        """Short identifier used for artifact dirs and report rows."""
        return f"{self.trace_source.value}_{self.public_dataset}_c{self.concurrency}"

    @property
    def enforces_min_duration(self) -> bool:
        return self.scenario in _MIN_DURATION_SCENARIOS


@dataclass(frozen=True)
class AgenticTracesModeSettings:
    """Wall-clock knobs for one :class:`AgenticTracesMode`.

    ``concurrency`` overrides the run spec's value when set, so a CI run can
    also shrink the load, not just the duration.
    """

    benchmark_duration: int
    agentic_cache_warmup_duration: int
    warmup_grace_period: int
    num_dataset_entries: int
    concurrency: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "benchmark_duration",
            "agentic_cache_warmup_duration",
            "warmup_grace_period",
            "num_dataset_entries",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"AgenticTracesModeSettings.{name} must be >= 0, "
                    f"got {getattr(self, name)}"
                )
        if self.num_dataset_entries < 1:
            raise ValueError(
                "AgenticTracesModeSettings.num_dataset_entries must be >= 1, "
                f"got {self.num_dataset_entries}"
            )
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError(
                "AgenticTracesModeSettings.concurrency must be >= 1 when set, "
                f"got {self.concurrency}"
            )


# Reference full-length run: the shape validated by hand before this workflow
# existed (1h profiling, 10min cache warmup, all 393 eligible traces).
FULL_MODE_SETTINGS = AgenticTracesModeSettings(
    benchmark_duration=3600,
    agentic_cache_warmup_duration=600,
    warmup_grace_period=1800,
    num_dataset_entries=393,
)

# Shortest run the scenario permits. Cache warmup and the trace pool shrink
# proportionally so a CI run is dominated by profiling rather than warmup.
CI_MODE_SETTINGS = AgenticTracesModeSettings(
    benchmark_duration=AGENTIC_TRACES_MIN_PROFILE_SECONDS,
    agentic_cache_warmup_duration=120,
    warmup_grace_period=600,
    num_dataset_entries=32,
)

DEFAULT_MODE_SETTINGS: Dict[AgenticTracesMode, AgenticTracesModeSettings] = {
    AgenticTracesMode.FULL: FULL_MODE_SETTINGS,
    AgenticTracesMode.CI: CI_MODE_SETTINGS,
}


@dataclass(frozen=True)
class AgenticTracesConfig:
    """Agentic-trace benchmark configuration for a single ``ModelSpec``.

    ``inferencex_git_ref`` pins the InferenceX revision cloned into the
    AGENTIC_TRACES venv. It is per-spec on purpose: the client, the scenario
    definition, and the dataset loaders all live in that repo, so reproducing a
    number means reproducing the client that produced it.
    """

    model_id: str
    inferencex_git_ref: str
    runs: Tuple[AgenticTracesRunSpec, ...] = (AgenticTracesRunSpec(),)
    mode_settings: Dict[AgenticTracesMode, AgenticTracesModeSettings] = field(
        default_factory=lambda: dict(DEFAULT_MODE_SETTINGS)
    )

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("AgenticTracesConfig.model_id must not be empty")
        if not self.inferencex_git_ref.strip():
            raise ValueError(
                f"AgenticTracesConfig.inferencex_git_ref must not be empty "
                f"for model_id={self.model_id!r}"
            )
        if not self.runs:
            raise ValueError(
                f"AgenticTracesConfig.runs must not be empty for "
                f"model_id={self.model_id!r}"
            )
        missing = [m.name for m in AgenticTracesMode if m not in self.mode_settings]
        if missing:
            raise ValueError(
                f"AgenticTracesConfig for model_id={self.model_id!r} is missing "
                f"mode_settings for: {', '.join(missing)}"
            )
        # Fail at import rather than 15 minutes into a run that the scenario
        # would reject for being too short.
        if any(run.enforces_min_duration for run in self.runs):
            for mode, settings in self.mode_settings.items():
                if settings.benchmark_duration < AGENTIC_TRACES_MIN_PROFILE_SECONDS:
                    raise ValueError(
                        f"AgenticTracesConfig for model_id={self.model_id!r} sets "
                        f"{mode.name} benchmark_duration="
                        f"{settings.benchmark_duration}s, below the "
                        f"{AGENTIC_TRACES_MIN_PROFILE_SECONDS}s minimum enforced "
                        f"by scenario(s) "
                        f"{sorted(_MIN_DURATION_SCENARIOS)}"
                    )

    def settings_for_mode(self, mode: AgenticTracesMode) -> AgenticTracesModeSettings:
        return self.mode_settings[mode]

    def trace_sources(self) -> Tuple[TraceSource, ...]:
        """Distinct trace sources referenced by this config, in run order."""
        seen: List[TraceSource] = []
        for run in self.runs:
            if run.trace_source not in seen:
                seen.append(run.trace_source)
        return tuple(seen)


def for_model_ids(model_ids: List[str], **kwargs) -> List[AgenticTracesConfig]:
    """Fan one parameter set out across several ``model_id``s.

    Keeps a model whose weights run on multiple devices/impls from needing a
    duplicated literal block per spec.
    """
    return [AgenticTracesConfig(model_id=mid, **kwargs) for mid in model_ids]


_agentic_traces_config_list: List[AgenticTracesConfig] = [
    # Kimi K2.7-Code on SUPER_CLUSTER (dev catalog). 256k dataset variant to
    # match the spec's 262144 max_context
    #
    # Pin commits reachable from InferenceX ``main``. The repo force-pushes and
    # deletes its agent/* and PR branches, so a sha taken from one stops
    # resolving ("reference is not a tree") once that branch is gone, and the
    # server then refuses it outright as "not our ref".
    AgenticTracesConfig(
        model_id="id_tt-transformers_Kimi-K2.7-Code_super_cluster",
        inferencex_git_ref="e2dcfa91c86936cc011e3be0668eb3b1ca17288f",
        runs=(
            AgenticTracesRunSpec(
                trace_source=TraceSource.INFERENCEX_AGENTX,
                public_dataset="semianalysis_cc_traces_weka_062126_256k",
                concurrency=8,
            ),
        ),
    ),
]

AGENTIC_TRACES_CONFIGS: Dict[str, AgenticTracesConfig] = map_configs_by_attr(
    config_list=_agentic_traces_config_list, attr="model_id"
)


def get_agentic_traces_config(model_spec) -> Optional[AgenticTracesConfig]:
    """Config for ``model_spec``, or ``None`` when the model has none.

    Callers treat ``None`` as "this model is not onboarded to agentic traces"
    and refuse to run, the same way an eval workflow refuses a model missing
    from ``EVAL_CONFIGS``.
    """
    model_id = getattr(model_spec, "model_id", None)
    if not model_id:
        return None
    return AGENTIC_TRACES_CONFIGS.get(model_id)


def resolve_run_specs(
    config: AgenticTracesConfig,
    *,
    trace_sources: Optional[Tuple[TraceSource, ...]] = None,
    git_ref_override: Optional[str] = None,
) -> Tuple[AgenticTracesConfig, Tuple[AgenticTracesRunSpec, ...]]:
    """Apply CLI-level narrowing/overrides to a config.

    Returns the (possibly ref-overridden) config alongside the selected run
    specs so the caller can pass both to the run builder.
    """
    effective = config
    if git_ref_override:
        effective = replace(config, inferencex_git_ref=git_ref_override)
    runs = effective.runs
    if trace_sources:
        wanted = set(trace_sources)
        runs = tuple(run for run in runs if run.trace_source in wanted)
        if not runs:
            raise ValueError(
                f"No agentic-trace runs for model_id={config.model_id!r} match "
                f"trace source(s) {sorted(s.value for s in wanted)}. "
                f"Configured: {sorted(s.value for s in config.trace_sources())}"
            )
    return effective, runs


__all__ = [
    "AGENTIC_TRACES_CONFIGS",
    "AGENTIC_TRACES_MIN_PROFILE_SECONDS",
    "AgenticTracesConfig",
    "AgenticTracesModeSettings",
    "AgenticTracesRunSpec",
    "CI_MODE_SETTINGS",
    "DEFAULT_MODE_SETTINGS",
    "FULL_MODE_SETTINGS",
    "TraceSource",
    "for_model_ids",
    "get_agentic_traces_config",
    "resolve_run_specs",
]
