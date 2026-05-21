# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prefix-caching benchmark scenario generation for `--tools aiperf --prefix-cache`.

A scenario manifest (JSON) describes per-preset (`ci`, `full`) a set of
scenarios, each of which expands into multiple AIPerf runs by taking the
cartesian product of:

- ``isl_profiles``       (input sequence length distributions)
- ``concurrencies``      (concurrency factors)
- ``arrival_patterns``   (constant | poisson | gamma | concurrency_burst)
- scenario-specific knobs (e.g. ``reuse_ratios`` for ``prefix_pool``)

Each expanded :class:`PrefixCacheRun` carries every parameter the AIPerf
runner needs; the runner is intentionally kept dumb about the scenario
shapes so we can extend / override scenarios entirely from the JSON
manifest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_MANIFEST_PATH = (
    Path(__file__).parent / "benchmark_targets" / "prefix_cache_scenarios.json"
)

ALL_SCENARIOS = (
    "shared_system",
    "prefix_pool",
    "multi_turn",
    "baseline",
    # Trace-driven scenario from AIPerf's prefix-synthesis pipeline.
    # See: https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md
    "mooncake_trace",
)
# Arrival patterns supported by AIPerf 0.5.0 `--arrival-pattern`.
# `gamma` with smoothness <1 produces bursty/clustered traffic (this is how
# AIPerf models "concurrency burst" workloads, per `--arrival-smoothness`
# documentation).
ARRIVAL_PATTERNS = ("constant", "poisson", "gamma")


@dataclass
class PrefixCacheRun:
    """A single AIPerf invocation that exercises one prefix-caching scenario."""

    scenario: str
    label: str
    isl_mean: int
    isl_stddev: int
    osl_mean: int
    osl_stddev: int
    concurrency: int
    request_count: int
    arrival_pattern: str
    arrival_smoothness: Optional[float] = None
    request_rate: Optional[float] = None

    # Mutually exclusive prefix knobs.
    shared_system_prompt_length: Optional[int] = None
    num_prefix_prompts: Optional[int] = None
    prefix_prompt_length: Optional[int] = None

    # Multi-turn knobs.
    conversation_num: Optional[int] = None
    conversation_turn_mean: Optional[int] = None
    conversation_turn_stddev: Optional[int] = None
    conversation_turn_delay_mean_ms: Optional[int] = None

    # Trace-driven scenario (mooncake_trace) knobs. When `trace_input_file` is
    # set the runner switches aiperf to `--custom-dataset-type mooncake-trace
    # --input-file <trace>` and ignores the synthetic ISL/OSL flags.
    trace_input_file: Optional[str] = None
    custom_dataset_type: Optional[str] = None  # e.g. "mooncake-trace"
    synthesis_speedup_ratio: Optional[float] = None
    synthesis_prefix_len_multiplier: Optional[float] = None
    synthesis_prefix_root_multiplier: Optional[int] = None
    synthesis_prompt_len_multiplier: Optional[float] = None
    synthesis_max_isl: Optional[int] = None
    synthesis_max_osl: Optional[int] = None
    fixed_schedule: bool = False  # use absolute timestamps from the trace
    block_size: Optional[int] = None  # KV cache block size for prefix grouping

    # Free-form provenance: derived/source field values for reporting.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uses_trace(self) -> bool:
        return bool(self.trace_input_file)

    def filesafe_label(self) -> str:
        return self.label.replace("/", "_").replace(" ", "_")


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a prefix-cache scenarios manifest from disk.

    *path* defaults to :data:`DEFAULT_MANIFEST_PATH` which is the in-tree
    auditable scenario manifest under ``benchmarking/benchmark_targets/``.
    """
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prefix-cache scenarios manifest not found: {manifest_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "presets" not in data:
        raise ValueError(
            f"Invalid prefix-cache scenarios manifest (no 'presets'): {manifest_path}"
        )
    return data


def _normalize_scenario_filter(scenarios: Optional[str]) -> List[str]:
    if not scenarios:
        return list(ALL_SCENARIOS)
    selected = [s.strip() for s in scenarios.split(",") if s.strip()]
    bad = [s for s in selected if s not in ALL_SCENARIOS]
    if bad:
        raise ValueError(
            f"Unknown prefix-cache scenarios: {bad}. "
            f"Valid choices: {list(ALL_SCENARIOS)}"
        )
    return selected


def _coerce_lengths(
    value_single: Optional[int], values_list: Optional[Sequence[int]], fallback: int
) -> List[int]:
    """Resolve a prompt-length list from JSON keys.

    Either a singular ``"_length"`` key (one length) or a plural
    ``"_lengths"`` array (multiple lengths). Always returns a non-empty list.
    """
    if values_list:
        return [int(v) for v in values_list]
    if value_single is not None:
        return [int(value_single)]
    return [int(fallback)]


def _build_arrival_overrides(
    arrival_pattern: Optional[str],
    request_rate: Optional[float],
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if arrival_pattern:
        if arrival_pattern not in ARRIVAL_PATTERNS:
            raise ValueError(
                f"Invalid arrival pattern: {arrival_pattern}. "
                f"Valid choices: {list(ARRIVAL_PATTERNS)}"
            )
        overrides["arrival_pattern"] = arrival_pattern
    if request_rate is not None:
        overrides["request_rate"] = float(request_rate)
    return overrides


def _expand_shared_system(
    cfg: Dict[str, Any],
    *,
    preset_request_count: int,
    isl_profiles: Sequence[Dict[str, Any]],
    osl_mean: int,
    osl_stddev: int,
    concurrencies: Sequence[int],
    arrival_overrides: Dict[str, Any],
) -> List[PrefixCacheRun]:
    lengths = _coerce_lengths(
        cfg.get("shared_system_prompt_length"),
        cfg.get("shared_system_prompt_lengths"),
        fallback=512,
    )
    arrivals = arrival_overrides.get("arrival_pattern_overrides") or cfg.get(
        "arrival_patterns"
    ) or [{"pattern": "constant"}]
    runs: List[PrefixCacheRun] = []
    request_count = int(cfg.get("request_count", preset_request_count))
    for isl_profile in isl_profiles:
        for length in lengths:
            for arrival in arrivals:
                for concurrency in concurrencies:
                    pattern = arrival_overrides.get("arrival_pattern") or arrival.get(
                        "pattern", "constant"
                    )
                    smoothness = arrival.get("smoothness")
                    request_rate = arrival_overrides.get("request_rate")
                    isl_name = isl_profile.get("name") or f"isl{isl_profile['isl_mean']}"
                    label = (
                        f"shared_sys{length}_{isl_name}"
                        f"_c{concurrency}_{pattern}"
                    )
                    runs.append(
                        PrefixCacheRun(
                            scenario="shared_system",
                            label=label,
                            isl_mean=int(isl_profile["isl_mean"]),
                            isl_stddev=int(isl_profile.get("isl_stddev", 0)),
                            osl_mean=osl_mean,
                            osl_stddev=osl_stddev,
                            concurrency=concurrency,
                            request_count=request_count,
                            arrival_pattern=pattern,
                            arrival_smoothness=smoothness,
                            request_rate=request_rate,
                            shared_system_prompt_length=int(length),
                            metadata={
                                "expected_reuse": "100%",
                                "isl_profile": isl_profile.get("name"),
                                "shared_system_prompt_length": int(length),
                            },
                        )
                    )
    return runs


def _expand_prefix_pool(
    cfg: Dict[str, Any],
    *,
    preset_request_count: int,
    isl_profiles: Sequence[Dict[str, Any]],
    osl_mean: int,
    osl_stddev: int,
    concurrencies: Sequence[int],
    arrival_overrides: Dict[str, Any],
) -> List[PrefixCacheRun]:
    lengths = _coerce_lengths(
        cfg.get("prefix_prompt_length"),
        cfg.get("prefix_prompt_lengths"),
        fallback=512,
    )
    reuse_ratios = [int(r) for r in cfg.get("reuse_ratios", [4])]
    arrivals = cfg.get("arrival_patterns") or [{"pattern": "poisson"}]
    runs: List[PrefixCacheRun] = []
    request_count = int(cfg.get("request_count", preset_request_count))
    for isl_profile in isl_profiles:
        for length in lengths:
            for reuse in reuse_ratios:
                # reuse_ratio = average requests per prefix => pool_size = ceil(N / reuse)
                pool_size = max(1, math.ceil(request_count / max(reuse, 1)))
                for arrival in arrivals:
                    for concurrency in concurrencies:
                        pattern = arrival_overrides.get(
                            "arrival_pattern"
                        ) or arrival.get("pattern", "poisson")
                        smoothness = arrival.get("smoothness")
                        request_rate = arrival_overrides.get("request_rate")
                        isl_name = (
                            isl_profile.get("name") or f"isl{isl_profile['isl_mean']}"
                        )
                        label = (
                            f"pool_reuse{reuse}_plen{length}_{isl_name}"
                            f"_c{concurrency}_{pattern}"
                        )
                        runs.append(
                            PrefixCacheRun(
                                scenario="prefix_pool",
                                label=label,
                                isl_mean=int(isl_profile["isl_mean"]),
                                isl_stddev=int(isl_profile.get("isl_stddev", 0)),
                                osl_mean=osl_mean,
                                osl_stddev=osl_stddev,
                                concurrency=concurrency,
                                request_count=request_count,
                                arrival_pattern=pattern,
                                arrival_smoothness=smoothness,
                                request_rate=request_rate,
                                num_prefix_prompts=pool_size,
                                prefix_prompt_length=int(length),
                                metadata={
                                    "reuse_ratio": reuse,
                                    "pool_size": pool_size,
                                    "isl_profile": isl_profile.get("name"),
                                    "prefix_prompt_length": int(length),
                                    "expected_reuse": f"~{(1 - 1 / max(reuse, 1)) * 100:.0f}%",
                                },
                            )
                        )
    return runs


def _expand_multi_turn(
    cfg: Dict[str, Any],
    *,
    preset_request_count: int,
    isl_profiles: Sequence[Dict[str, Any]],
    osl_mean: int,
    osl_stddev: int,
    concurrencies: Sequence[int],
    arrival_overrides: Dict[str, Any],
) -> List[PrefixCacheRun]:
    conversation_num = int(cfg.get("conversation_num", 64))
    turn_mean = int(cfg.get("conversation_turn_mean", 6))
    turn_stddev = int(cfg.get("conversation_turn_stddev", 2))
    delay_mean = int(cfg.get("conversation_turn_delay_mean_ms", 1500))
    arrivals = cfg.get("arrival_patterns") or [{"pattern": "poisson"}]
    # For multi-turn, total requests ~= conversation_num * turn_mean; we still pass
    # an explicit request_count to AIPerf so it stops at a predictable point.
    request_count = int(
        cfg.get("request_count", max(preset_request_count, conversation_num * turn_mean))
    )
    runs: List[PrefixCacheRun] = []
    for isl_profile in isl_profiles:
        for arrival in arrivals:
            for concurrency in concurrencies:
                pattern = arrival_overrides.get("arrival_pattern") or arrival.get(
                    "pattern", "poisson"
                )
                smoothness = arrival.get("smoothness")
                request_rate = arrival_overrides.get("request_rate")
                isl_name = isl_profile.get("name") or f"isl{isl_profile['isl_mean']}"
                label = (
                    f"mt_conv{conversation_num}_turns{turn_mean}_{isl_name}"
                    f"_c{concurrency}_{pattern}"
                )
                runs.append(
                    PrefixCacheRun(
                        scenario="multi_turn",
                        label=label,
                        isl_mean=int(isl_profile["isl_mean"]),
                        isl_stddev=int(isl_profile.get("isl_stddev", 0)),
                        osl_mean=osl_mean,
                        osl_stddev=osl_stddev,
                        concurrency=concurrency,
                        request_count=request_count,
                        arrival_pattern=pattern,
                        arrival_smoothness=smoothness,
                        request_rate=request_rate,
                        conversation_num=conversation_num,
                        conversation_turn_mean=turn_mean,
                        conversation_turn_stddev=turn_stddev,
                        conversation_turn_delay_mean_ms=delay_mean,
                        metadata={
                            "isl_profile": isl_profile.get("name"),
                            "conversation_num": conversation_num,
                            "conversation_turn_mean": turn_mean,
                            "expected_reuse": "organic (turns>=2)",
                        },
                    )
                )
    return runs


def _expand_baseline(
    cfg: Dict[str, Any],
    *,
    preset_request_count: int,
    isl_profiles: Sequence[Dict[str, Any]],
    osl_mean: int,
    osl_stddev: int,
    concurrencies: Sequence[int],
    arrival_overrides: Dict[str, Any],
) -> List[PrefixCacheRun]:
    arrivals = cfg.get("arrival_patterns") or [{"pattern": "constant"}]
    request_count = int(cfg.get("request_count", preset_request_count))
    runs: List[PrefixCacheRun] = []
    for isl_profile in isl_profiles:
        for arrival in arrivals:
            for concurrency in concurrencies:
                pattern = arrival_overrides.get("arrival_pattern") or arrival.get(
                    "pattern", "constant"
                )
                smoothness = arrival.get("smoothness")
                request_rate = arrival_overrides.get("request_rate")
                isl_name = isl_profile.get("name") or f"isl{isl_profile['isl_mean']}"
                label = f"baseline_{isl_name}_c{concurrency}_{pattern}"
                runs.append(
                    PrefixCacheRun(
                        scenario="baseline",
                        label=label,
                        isl_mean=int(isl_profile["isl_mean"]),
                        isl_stddev=int(isl_profile.get("isl_stddev", 0)),
                        osl_mean=osl_mean,
                        osl_stddev=osl_stddev,
                        concurrency=concurrency,
                        request_count=request_count,
                        arrival_pattern=pattern,
                        arrival_smoothness=smoothness,
                        request_rate=request_rate,
                        metadata={
                            "isl_profile": isl_profile.get("name"),
                            "expected_reuse": "0% (control)",
                        },
                    )
                )
    return runs


def _resolve_trace_path(raw: str) -> Path:
    """Resolve a manifest-declared trace path.

    Absolute paths are taken verbatim. Relative paths are resolved against
    the manifest's containing directory (``benchmark_targets/``), which is
    where in-tree sample traces live.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    base = Path(__file__).parent / "benchmark_targets"
    return (base / p).resolve()


def _expand_mooncake_trace(
    cfg: Dict[str, Any],
    *,
    preset_request_count: int,
    isl_profiles: Sequence[Dict[str, Any]],  # ignored for trace-driven scenarios
    osl_mean: int,  # ignored for trace-driven scenarios
    osl_stddev: int,  # ignored for trace-driven scenarios
    concurrencies: Sequence[int],
    arrival_overrides: Dict[str, Any],
    trace_path_override: Optional[str] = None,
) -> List[PrefixCacheRun]:
    """Build runs for the AIPerf prefix-synthesis trace scenario.

    The manifest entry must include:

    ::

        "mooncake_trace": {
          "traces": [
            {"name": "ci-sample", "path": "sample_traces/ci_mooncake.jsonl"}
          ],
          "synthesis_grid": [
            {"name": "baseline",  "speedup": 1.0, "prefix_len": 1.0,
             "prefix_root": 1, "prompt_len": 1.0},
            {"name": "more_reuse", "prefix_len": 1.5, "prompt_len": 0.7}
          ],
          "block_size": 512,
          "fixed_schedule": false,
          "arrival_patterns": [{"pattern": "poisson"}]   # used only when
                                                          # fixed_schedule=false
        }

    ``trace_path_override`` (from CLI ``--prefix-cache-trace``) replaces
    every manifest-declared trace; this is the recommended way to point at
    a production mooncake JSONL without editing the manifest.
    """
    traces_cfg: List[Dict[str, Any]] = list(cfg.get("traces", []))
    if trace_path_override:
        traces_cfg = [{"name": "override", "path": trace_path_override}]
    if not traces_cfg:
        return []

    synthesis_grid: List[Dict[str, Any]] = list(cfg.get("synthesis_grid", [{}]))
    fixed_schedule: bool = bool(cfg.get("fixed_schedule", False))
    block_size: Optional[int] = (
        int(cfg["block_size"]) if cfg.get("block_size") is not None else None
    )
    arrivals = cfg.get("arrival_patterns") or [{"pattern": "poisson"}]
    request_count = int(cfg.get("request_count", preset_request_count))

    runs: List[PrefixCacheRun] = []
    for trace in traces_cfg:
        raw_path = trace.get("path") or trace.get("file")
        if not raw_path:
            raise ValueError(
                "mooncake_trace entry is missing 'path' (relative to "
                "benchmarking/benchmark_targets/ or an absolute path)."
            )
        resolved = _resolve_trace_path(raw_path)
        trace_name = trace.get("name") or resolved.stem

        for synth in synthesis_grid:
            synth_name = synth.get("name") or "default"
            for concurrency in concurrencies:
                # When fixed_schedule is enabled the arrival pattern is
                # driven entirely by the trace timestamps, so we still
                # emit a single run per concurrency (no arrival sweep).
                arrivals_for_run = (
                    [{"pattern": "constant"}] if fixed_schedule else arrivals
                )
                for arrival in arrivals_for_run:
                    pattern = arrival_overrides.get(
                        "arrival_pattern"
                    ) or arrival.get("pattern", "poisson")
                    smoothness = arrival.get("smoothness")
                    request_rate = arrival_overrides.get("request_rate")
                    label = (
                        f"trace_{trace_name}_{synth_name}_c{concurrency}_"
                        f"{'fixed' if fixed_schedule else pattern}"
                    )
                    runs.append(
                        PrefixCacheRun(
                            scenario="mooncake_trace",
                            label=label,
                            # ISL/OSL are *advisory only* in trace mode; the
                            # actual lengths come from the trace. We surface
                            # the trace's nominal mean so reports stay
                            # consistent across scenarios.
                            isl_mean=int(synth.get("nominal_isl_mean", 0)),
                            isl_stddev=int(synth.get("nominal_isl_stddev", 0)),
                            osl_mean=int(synth.get("nominal_osl_mean", 0)),
                            osl_stddev=int(synth.get("nominal_osl_stddev", 0)),
                            concurrency=int(concurrency),
                            request_count=request_count,
                            arrival_pattern=pattern,
                            arrival_smoothness=smoothness,
                            request_rate=request_rate,
                            trace_input_file=str(resolved),
                            # AIPerf's CLI help advertises "mooncake-trace"
                            # (hyphenated) but its pydantic enum only accepts
                            # the underscored form. See aiperf v0.5.0 source.
                            custom_dataset_type="mooncake_trace",
                            synthesis_speedup_ratio=(
                                float(synth["speedup"]) if "speedup" in synth else None
                            ),
                            synthesis_prefix_len_multiplier=(
                                float(synth["prefix_len"])
                                if "prefix_len" in synth
                                else None
                            ),
                            synthesis_prefix_root_multiplier=(
                                int(synth["prefix_root"])
                                if "prefix_root" in synth
                                else None
                            ),
                            synthesis_prompt_len_multiplier=(
                                float(synth["prompt_len"])
                                if "prompt_len" in synth
                                else None
                            ),
                            synthesis_max_isl=(
                                int(synth["max_isl"]) if "max_isl" in synth else None
                            ),
                            synthesis_max_osl=(
                                int(synth["max_osl"]) if "max_osl" in synth else None
                            ),
                            fixed_schedule=fixed_schedule,
                            block_size=block_size,
                            metadata={
                                "trace_name": trace_name,
                                "trace_path": str(resolved),
                                "synthesis_variant": synth_name,
                                "synthesis": {
                                    k: synth[k]
                                    for k in (
                                        "speedup",
                                        "prefix_len",
                                        "prefix_root",
                                        "prompt_len",
                                        "max_isl",
                                        "max_osl",
                                    )
                                    if k in synth
                                },
                                "fixed_schedule": fixed_schedule,
                                "expected_reuse": "trace-derived",
                            },
                        )
                    )
    return runs


_SCENARIO_EXPANDERS = {
    "shared_system": _expand_shared_system,
    "prefix_pool": _expand_prefix_pool,
    "multi_turn": _expand_multi_turn,
    "baseline": _expand_baseline,
    "mooncake_trace": _expand_mooncake_trace,
}


def build_runs(
    preset: str = "full",
    *,
    scenarios: Optional[str] = None,
    arrival_pattern: Optional[str] = None,
    request_rate: Optional[float] = None,
    manifest_path: Optional[Path] = None,
    trace_path_override: Optional[str] = None,
) -> List[PrefixCacheRun]:
    """Expand the requested scenarios from the manifest into runs.

    Parameters
    ----------
    preset:
        One of the manifest's ``presets`` keys (``ci`` or ``full`` by default).
    scenarios:
        Comma-separated subset of :data:`ALL_SCENARIOS`. ``None`` = all.
    arrival_pattern:
        Override every run's arrival pattern (``constant``, ``poisson`` or
        ``gamma``). For bursty traffic use ``gamma`` with the manifest's
        ``arrival_smoothness`` set to a value below 1.0.
    request_rate:
        Override every run's target request rate (req/s).
    manifest_path:
        Optional alternate manifest. Defaults to :data:`DEFAULT_MANIFEST_PATH`.
    trace_path_override:
        Optional path to a mooncake JSONL trace file. When supplied this
        replaces every ``mooncake_trace`` scenario's manifest-declared trace.
    """
    manifest = load_manifest(manifest_path)
    if preset not in manifest["presets"]:
        raise ValueError(
            f"Unknown prefix-cache preset: {preset}. "
            f"Available: {list(manifest['presets'].keys())}"
        )

    preset_cfg = manifest["presets"][preset]
    isl_profiles = preset_cfg["isl_profiles"]
    concurrencies = preset_cfg["concurrencies"]
    osl_mean = int(preset_cfg.get("osl_mean", 128))
    osl_stddev = int(preset_cfg.get("osl_stddev", 0))
    preset_request_count = int(preset_cfg.get("request_count", 256))

    selected = _normalize_scenario_filter(scenarios)
    arrival_overrides = _build_arrival_overrides(arrival_pattern, request_rate)

    runs: List[PrefixCacheRun] = []
    for scenario in selected:
        scenario_cfg = preset_cfg.get("scenarios", {}).get(scenario)
        if scenario_cfg is None:
            # Scenario not defined in this preset, skip silently.
            continue
        expander = _SCENARIO_EXPANDERS[scenario]
        kwargs = dict(
            preset_request_count=preset_request_count,
            isl_profiles=isl_profiles,
            osl_mean=osl_mean,
            osl_stddev=osl_stddev,
            concurrencies=concurrencies,
            arrival_overrides=arrival_overrides,
        )
        if scenario == "mooncake_trace":
            kwargs["trace_path_override"] = trace_path_override
        runs.extend(expander(scenario_cfg, **kwargs))
    return runs


def summarize_runs(runs: Sequence[PrefixCacheRun]) -> str:
    """Pretty single-string summary used for log output before execution."""
    lines = ["Prefix-cache benchmark plan:"]
    header = (
        f"  {'#':<3} {'scenario':<14} {'label':<55} {'isl':<5} {'osl':<5} "
        f"{'con':<4} {'req':<5} {'arrival':<10} {'rate':<8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i, r in enumerate(runs, 1):
        rate_str = f"{r.request_rate:.2f}" if r.request_rate is not None else "-"
        lines.append(
            f"  {i:<3} {r.scenario:<14} {r.label[:55]:<55} "
            f"{r.isl_mean:<5} {r.osl_mean:<5} {r.concurrency:<4} "
            f"{r.request_count:<5} {r.arrival_pattern:<10} {rate_str:<8}"
        )
    return "\n".join(lines)
