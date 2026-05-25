# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Engine-agnostic helpers for the speculative-decoding benchmark.

`SpecDecodeRunSpec` describes a single sweep config (dataset, output length,
concurrency). `merge_acceptance_rate` annotates the per-sweep result JSON in
place with metrics scraped from Prometheus. `pair_and_compute_speedup`
consumes two such annotated results (baseline + speculative) and emits a
sidecar dict with per-percentile speedup ratios.

Kept separate from the aiperf-specific runner so a future
``run_sglang_spec_decode_benchmarks.py`` (or any other client tool) can reuse
the same sweep specs, result annotation, and pairing math. The runner is
responsible for normalising tool-specific output JSON into the vllm-bench
field names (``mean_e2el_ms``, ``p50_e2el_ms``, ``output_throughput``, etc.)
that ``pair_and_compute_speedup`` reads here.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecDecodeRunSpec:
    """One sweep config for a spec-decode benchmark run.

    ``public_dataset`` is an aiperf ``--public-dataset`` slug. Examples:

      - ``"spec_bench"`` — hemingkx Spec-Bench (480 prompts, no category breakdown)
      - ``"speed_bench_qualitative"`` — nvidia SPEED-Bench qualitative split (whole)
      - ``"speed_bench_coding"`` (and the other 10 per-category SPEED-Bench slugs)

    See ``aiperf/plugin/plugins.yaml`` for the full list of registered slugs.
    """

    public_dataset: str
    output_len: int
    max_concurrency: int
    num_prompts: int

    def __post_init__(self) -> None:
        if not self.public_dataset:
            raise ValueError("public_dataset is required")

    @property
    def slug(self) -> str:
        """Short identifier for use in result filenames."""
        return "_".join(
            [
                self.public_dataset,
                f"osl-{self.output_len}",
                f"maxcon-{self.max_concurrency}",
                f"n-{self.num_prompts}",
            ]
        )


def merge_acceptance_rate(
    result_json_path: Union[str, Path], metrics: Dict[str, Any]
) -> None:
    """Atomically merge spec-decode metrics into a per-sweep result JSON.

    Writes ``data["spec_decode_metrics"] = metrics`` then renames over the
    original file so a reader never sees a half-written JSON, matching the
    pattern from ``benchmarking.run_benchmarks.annotate_structured_output_result``.
    """
    path = Path(result_json_path)
    with open(path) as f:
        data = json.load(f)
    data["spec_decode_metrics"] = metrics
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _safe_ratio(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def pair_and_compute_speedup(
    baseline_path: Union[str, Path], spec_path: Union[str, Path]
) -> Dict[str, Any]:
    """Compute per-percentile speedup from a baseline and speculative result.

    Reads each per-sweep result JSON (normalised to vllm-bench field names by
    the runner) and produces a dict with E2EL speedups (baseline / spec —
    values > 1 mean spec is faster) plus an output-throughput ratio
    (spec / baseline). Returns ``None`` for any ratio whose numerator or
    denominator is missing or zero, so callers can distinguish "not measured"
    from a legitimate zero.
    """
    baseline_path = Path(baseline_path)
    spec_path = Path(spec_path)
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(spec_path) as f:
        spec = json.load(f)

    def latency_speedup(field: str) -> Optional[float]:
        return _safe_ratio(baseline.get(field), spec.get(field))

    return {
        "baseline_path": str(baseline_path),
        "spec_path": str(spec_path),
        "speedup_mean_e2el": latency_speedup("mean_e2el_ms"),
        "speedup_p50_e2el": latency_speedup("p50_e2el_ms"),
        "speedup_p95_e2el": latency_speedup("p95_e2el_ms"),
        "speedup_p99_e2el": latency_speedup("p99_e2el_ms"),
        "tpot_ratio_p50": latency_speedup("p50_tpot_ms"),
        "tpot_ratio_p95": latency_speedup("p95_tpot_ms"),
        "tpot_ratio_p99": latency_speedup("p99_tpot_ms"),
        "output_tput_ratio": _safe_ratio(
            spec.get("output_throughput"),
            baseline.get("output_throughput"),
        ),
        "baseline_acceptance_rate": (
            (baseline.get("spec_decode_metrics") or {}).get("acceptance_rate")
        ),
        "spec_acceptance_rate": (
            (spec.get("spec_decode_metrics") or {}).get("acceptance_rate")
        ),
    }
