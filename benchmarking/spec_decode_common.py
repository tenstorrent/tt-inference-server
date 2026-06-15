# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Engine-agnostic helpers for the speculative-decoding benchmark.

`SpecDecodeRunSpec` describes a single sweep config (dataset, output length,
concurrency). `merge_acceptance_rate` annotates the per-sweep result JSON in
place with metrics scraped from Prometheus.

Kept separate from the aiperf-specific runner so a future
``run_sglang_spec_decode_benchmarks.py`` (or any other client tool) can reuse
the same sweep specs and result annotation. The runner is responsible for
normalising tool-specific output JSON into the vllm-bench field names
(``mean_e2el_ms``, ``p50_e2el_ms``, ``output_throughput``, etc.) that the
report layer reads.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecDecodeRunSpec:
    public_dataset: str
    max_concurrency: int
    num_prompts: Optional[int] = None
    output_len: Optional[int] = None
    # Upper bound on tokens generated per request, injected as
    # ``--extra-inputs max_completion_tokens:<N>``. this is not a lower bound, the model
    # is allowed to output its natural length, then it cuts off at N to prevent timeout
    max_completion_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.public_dataset:
            raise ValueError("public_dataset is required")

    @property
    def slug(self) -> str:
        """Short identifier for use in result filenames.

        ``osl-<N>`` is included only when ``output_len`` is set; omitting it
        signals that the run let the model decode to its natural EOS.
        ``n-<N>`` is included only when ``num_prompts`` is set; omitting it
        signals the run consumed every prompt in the public dataset.

        ``max_completion_tokens`` is intentionally left out of the slug: it is
        a wall-clock guard rail, not a workload dimension, and the report
        parsers key off the dataset/osl/maxcon/n fields only.
        """
        parts = [self.public_dataset]
        if self.output_len is not None:
            parts.append(f"osl-{self.output_len}")
        parts.append(f"maxcon-{self.max_concurrency}")
        if self.num_prompts is not None:
            parts.append(f"n-{self.num_prompts}")
        return "_".join(parts)


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
