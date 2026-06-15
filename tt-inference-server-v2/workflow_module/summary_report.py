# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Assemble a benchmark-summary report from many per-run reports on disk.

Pipeline:

    discover_run_reports(container) -> [Path]
        -> load_run_reports(...)    -> [ReportSchema]
        -> aggregate_benchmark_runs -> [BenchmarkAggregate]   (pure)
        -> aggregate_to_block(...)  -> Block(kind="benchmarks")  (+ tier checks)
        -> build_summary_schema     -> ReportSchema
        -> acceptance_criteria_check + ReportGenerator.generate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from report_module import (
    GenerateResult,
    ReportGenerator,
    ReportSchema,
    acceptance_criteria_check,
    format_acceptance_summary_markdown,
)
from report_module.acceptance_criteria import CATEGORY_BENCHMARKS
from report_module.display import display_name
from report_module.schema import Block
from report_module.summary import (
    BenchmarkAggregate,
    MetricStats,
    aggregate_benchmark_runs,
)
from test_module._test_common.report_types import ReportCheckTypes
from test_module._test_common.target_check import (
    MetricSpec,
    evaluate_tiered,
    get_performance_targets,
    summary_from_tiered,
)

logger = logging.getLogger(__name__)

SUMMARY_KIND = "benchmarks"

RUN_REPORT_GLOB = "run_*/report_*.json"


@dataclass(frozen=True)
class SummaryMetricSpec:
    """Map an aggregated metric to a PerformanceTargets attribute."""

    metric_paths: Tuple[str, ...]
    target_attr: str
    lower_is_better: bool
    scale: float
    field_name: str


_IMAGE_SUMMARY_SPECS: Tuple[SummaryMetricSpec, ...] = (
    SummaryMetricSpec(("Benchmarks.ttft",), "ttft_ms", True, 1000.0, "ttft"),
    SummaryMetricSpec(
        ("Benchmarks.tput_user", "Benchmarks.inference_steps_per_second"),
        "tput_user",
        False,
        1.0,
        "tput_user",
    ),
)


def discover_run_reports(container_dir: Path) -> List[Path]:
    return sorted(Path(container_dir).glob(RUN_REPORT_GLOB))


def load_run_reports(paths: Sequence[Path]) -> List[ReportSchema]:
    schemas: List[ReportSchema] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable run report %s: %s", path, exc)
            continue
        schemas.append(ReportSchema.from_dict(payload))
    return schemas


def _display_metric_name(metric_path: str) -> str:
    return display_name(metric_path.rsplit(".", 1)[-1])


def _stats_record(metric_path: str, stats: MetricStats) -> dict:
    """One statistics-table row with display-friendly metric + column labels."""
    return {
        "Metric": _display_metric_name(metric_path),
        "Runs": stats.n,
        "Mean": stats.mean,
        "Median": stats.median,
        "Std": stats.stdev,
        "Min": stats.minimum,
        "Max": stats.maximum,
        "P50": stats.p50,
        "P90": stats.p90,
        "P99": stats.p99,
        "CoV": stats.cov,
    }


def _select_actual(
    aggregate: BenchmarkAggregate, spec: SummaryMetricSpec
) -> Optional[float]:
    """Mean of the first matching metric path, scaled to the target unit."""
    for path in spec.metric_paths:
        stats = aggregate.metrics.get(path)
        if stats is not None:
            return stats.mean * spec.scale
    return None


def _summary_target_checks(
    aggregate: BenchmarkAggregate,
) -> Tuple[dict, ReportCheckTypes]:
    """Re-run the 3-tier target check on the aggregated mean of each metric."""
    targets = get_performance_targets(aggregate.model, aggregate.device)
    specs = [
        MetricSpec(
            name=spec.field_name,
            actual=_select_actual(aggregate, spec),
            target_attr=spec.target_attr,
            lower_is_better=spec.lower_is_better,
            field_name=spec.field_name,
        )
        for spec in _IMAGE_SUMMARY_SPECS
    ]
    target_checks = evaluate_tiered(targets, specs)
    return target_checks, summary_from_tiered(target_checks)


def aggregate_to_block(aggregate: BenchmarkAggregate) -> Block:
    target_checks, accuracy_check = _summary_target_checks(aggregate)
    statistics = [
        _stats_record(name, stats) for name, stats in aggregate.metrics.items()
    ]
    data = {
        "num_runs": aggregate.run_count,
        "accuracy_check": accuracy_check,
        "statistics": statistics,
        "target_checks": target_checks,
    }
    title = f"{aggregate.title or 'Benchmark'} (summary of {aggregate.run_count} runs)"
    return Block(
        kind=SUMMARY_KIND,
        task_type=aggregate.task_type or None,
        title=title,
        id=aggregate.block_id,
        targets={"num_runs": aggregate.run_count},
        data=data,
    )


def _latest_generated_at(schemas: Sequence[ReportSchema]) -> str:
    stamps = [
        str(schema.metadata.get("generated_at") or "")
        for schema in schemas
        if schema.metadata.get("generated_at")
    ]
    if stamps:
        return max(stamps)
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(text: str) -> str:
    return text.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _compact_timestamp(text: str) -> str:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return "".join(ch for ch in text if ch.isalnum()) or "unknown"


def build_summary_schema(schemas: Sequence[ReportSchema]) -> Optional[ReportSchema]:
    aggregates = aggregate_benchmark_runs(schemas)
    if not aggregates:
        return None

    sections = [aggregate_to_block(aggregate) for aggregate in aggregates]
    first = aggregates[0]
    generated_at = _latest_generated_at(schemas)
    report_id = (
        f"summary_{_slug(first.model)}_{_slug(first.device)}"
        f"_{_compact_timestamp(generated_at)}"
    )
    metadata = {
        "model_name": first.model,
        "device": first.device,
        "generated_at": generated_at,
        "report_id": report_id,
        "summary": True,
        "num_runs_total": len(schemas),
    }
    return ReportSchema(metadata=metadata, sections=sections)


def generate_summary_report(
    schemas: Sequence[ReportSchema], output_dir: Path
) -> Optional[GenerateResult]:
    """Build, accept, and render the summary report. ``None`` if no benchmarks."""
    schema = build_summary_schema(schemas)
    if schema is None:
        return None

    accepted, blockers, categories = acceptance_criteria_check(schema)
    # A benchmark summary only ever holds benchmark blocks, so drop the always-NA
    # Evals / Spec Tests categories from the acceptance section.
    categories = [
        category for category in categories if category.name == CATEGORY_BENCHMARKS
    ]
    schema.metadata["acceptance_summary_markdown"] = format_acceptance_summary_markdown(
        accepted, blockers, categories
    )
    schema.metadata["acceptance_criteria"] = {
        "accepted": accepted,
        "blockers": blockers,
        "categories": [category.to_dict() for category in categories],
    }
    logger.info(
        "Benchmark summary acceptance: %s (%d blocker(s))",
        "PASS" if accepted else "FAIL",
        len(blockers),
    )
    return ReportGenerator().generate(schema, output_dir)


def summarize_container(
    container_dir: Path, output_dir: Optional[Path] = None
) -> Optional[GenerateResult]:
    """Discover run reports under ``container_dir`` and render their summary."""
    container_dir = Path(container_dir)
    paths = discover_run_reports(container_dir)
    schemas = load_run_reports(paths)
    if not schemas:
        logger.warning("No run reports found under %s", container_dir)
        return None
    target_dir = (
        Path(output_dir) if output_dir is not None else container_dir / "summary"
    )
    return generate_summary_report(schemas, target_dir)


__all__ = [
    "RUN_REPORT_GLOB",
    "SUMMARY_KIND",
    "SummaryMetricSpec",
    "aggregate_to_block",
    "build_summary_schema",
    "discover_run_reports",
    "generate_summary_report",
    "load_run_reports",
    "summarize_container",
]
