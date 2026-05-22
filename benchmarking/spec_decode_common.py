# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Engine-agnostic helpers for the speculative-decoding benchmark.

`SpecDecodeRunSpec` describes a single sweep config (dataset, category,
output length, concurrency). `merge_acceptance_rate` annotates a
``vllm bench serve --result-filename`` JSON in place with metrics scraped
from Prometheus. `pair_and_compute_speedup` consumes two such annotated
results (baseline + speculative) and emits a sidecar dict with
per-percentile speedup ratios.

Kept separate from the vLLM-specific runner so a future
``run_sglang_spec_decode_benchmarks.py`` (or any other backend) can reuse
the same sweep specs, result annotation, and pairing math.
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

    ``dataset_kind`` selects the upstream vLLM-bench dataset name:

      - ``"spec_bench"`` → ``--dataset-name spec_bench``
      - ``"speed_bench"`` → ``--dataset-name speed_bench``

    ``category`` maps to ``--spec-bench-category`` / ``--speed-bench-category``.
    ``None`` means "don't pass the flag", which makes the upstream loader
    use every row in the (sub)set — matching vllm's
    ``if (not self.category) or (self.category == row["category"])``
    semantics. Concretely: pass a real category to isolate per-domain
    acceptance rate; pass ``None`` to sweep the whole (sub)set.

    ``speed_bench_subset`` is only meaningful when
    ``dataset_kind == "speed_bench"`` (``--speed-bench-dataset-subset``).
    """

    dataset_kind: str
    category: Optional[str]
    output_len: int
    max_concurrency: int
    num_prompts: int
    speed_bench_subset: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dataset_kind not in ("spec_bench", "speed_bench"):
            raise ValueError(
                f"Unsupported dataset_kind: {self.dataset_kind!r}. "
                "Expected 'spec_bench' or 'speed_bench'."
            )
        if self.dataset_kind == "speed_bench" and self.speed_bench_subset is None:
            raise ValueError(
                "speed_bench_subset is required when dataset_kind='speed_bench'"
            )

    @property
    def slug(self) -> str:
        """Short identifier for use in result filenames."""
        parts = [self.dataset_kind]
        if self.category:
            parts.append(self.category)
        else:
            parts.append("all")
        if self.speed_bench_subset:
            parts.append(self.speed_bench_subset)
        parts.extend(
            [
                f"osl-{self.output_len}",
                f"maxcon-{self.max_concurrency}",
                f"n-{self.num_prompts}",
            ]
        )
        return "_".join(parts)


def merge_acceptance_rate(
    result_json_path: Union[str, Path], metrics: Dict[str, Any]
) -> None:
    """Atomically merge spec-decode metrics into a vllm-bench result JSON.

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

    Reads each ``vllm bench serve --result-filename`` JSON and produces a
    dict with E2EL speedups (baseline / spec — values > 1 mean spec is faster)
    plus an output-throughput ratio (spec / baseline). Returns ``None`` for
    any ratio whose numerator or denominator is missing or zero, so callers
    can distinguish "not measured" from a legitimate zero.
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
