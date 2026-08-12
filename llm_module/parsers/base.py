# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser interface: raw tool output (dict) -> report ``Block``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

BENCHMARKS_KIND = "benchmarks"


class LLMResultParser(ABC):
    """Adapt one LLM perf tool's raw JSON output into a report Block.

    Each concrete parser knows the schema of one tool's result file and
    names that tool in ``tool`` / ``tool_label``; ``kind`` stays the
    canonical report kind so the block is routed like every other
    benchmark. Drivers must not call parsers themselves; the runner
    orchestrates ``driver.run() -> parser.parse()``.
    """

    kind: str = BENCHMARKS_KIND
    tool: str = ""
    tool_label: str = ""

    @abstractmethod
    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        """Convert a raw result dict into a single Block for the report."""

    def _wrap_record(
        self, record: Dict[str, Any], *, title: Optional[str] = None
    ) -> Block:
        """Wrap a flat report record in the canonical Block shape.

        ``data`` carries the report sections only — never a duplicate of
        the envelope fields (``kind``/``tool``/``model``/``device``/
        ``timestamp``). Per-block envelope fields move to
        ``Block.targets`` so the runner can build report-level metadata
        without hunting them out of section data, while the renderer
        pulls model/device from the schema's metadata via its existing
        fallback in :func:`report_module.renderers._resolve_model_device`.

        ``title`` sets the section heading the generic renderer emits;
        leave it ``None`` to fall back to the tool-derived heading (or,
        for parsers with no ``tool``, to the kind-derived one).
        """
        model = str(record.get("model", ""))
        device = str(record.get("device", ""))
        timestamp = str(record.get("timestamp", ""))
        block_id = _slugify_block_id(model, device)
        section_data = {
            k: v
            for k, v in record.items()
            if k not in ("kind", "tool", "model", "device", "timestamp")
        }
        targets: Dict[str, Any] = {}
        if self.tool:
            targets["tool"] = self.tool
        if model:
            targets["model"] = model
        if device:
            targets["device"] = device
        if timestamp:
            targets["timestamp"] = timestamp
        return Block(
            kind=self.kind,
            id=block_id or None,
            title=title or self._default_title(record),
            data=section_data,
            targets=targets,
        )

    def _default_title(self, record: Mapping[str, Any]) -> Optional[str]:
        """Heading for a benchmark block, naming the tool that ran it."""
        del record  # title depends on the tool only
        if not self.tool:
            return None
        label = self.tool_label or self.tool.replace("_", " ").title()
        return f"{label} Benchmark"


def round_metric(value: Any, digits: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def metric_stat(raw: Mapping[str, Any], key: str, stat: str = "avg") -> Any:
    metric = raw.get(key)
    if not isinstance(metric, Mapping):
        return None
    return round_metric(metric.get(stat), 4)


def metric_stat_int(raw: Mapping[str, Any], key: str) -> Any:
    value = metric_stat(raw, key)
    return int(value) if isinstance(value, (int, float)) else None


def decode_throughput(
    tput_user: Any, mean_tpot_ms: Any, concurrency: Any, digits: int = 4
) -> Any:
    """Aggregate decode throughput: per-user decode rate x concurrency.

    This is the quantity the ``tput`` perf target is defined in terms of --
    median decode interactivity (``1000 / TPOT``) summed over the concurrent
    users -- so it must be measured the same way to be gradeable.

    It is deliberately NOT the tools' own ``output_token_throughput`` /
    ``output_throughput``, which is total output tokens over the whole
    benchmark window and therefore charges decode for time spent in prefill
    and in queueing. The two agree only when prefill is negligible; at
    ISL 131072 the wall-clock figure came out 0.46x of a target the run was
    actually meeting on a decode basis. That figure is still reported, as
    ``tps_output_throughput``.

    Matches how ``test_module/stress_tests/stress_tests_record_builder``
    already derives the same field (``mean_tps * actual_max_con``).
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, (int, float)):
        return None
    if concurrency <= 0:
        return None
    per_user = tput_user
    if isinstance(per_user, bool) or not isinstance(per_user, (int, float)):
        per_user = None
    if per_user is None:
        if isinstance(mean_tpot_ms, bool) or not isinstance(mean_tpot_ms, (int, float)):
            return None
        if mean_tpot_ms <= 0:
            return None
        per_user = 1000.0 / mean_tpot_ms
    return round_metric(per_user * concurrency, digits)


def _slugify_block_id(model: str, device: str) -> str:
    parts = [p for p in (model, device) if p]
    if not parts:
        return ""
    joined = "_".join(parts)
    return joined.replace("/", "__").replace("\\", "__").replace(" ", "_")
