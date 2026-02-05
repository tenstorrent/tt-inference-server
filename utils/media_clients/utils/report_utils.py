# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Report generation for media clients.

Flow: strategy builds ReportContext.from_strategy(self), then calls
ReportGenerator.generate_benchmark_report(...) or generate_eval_report(...)
with context and client-specific extra_benchmarks/extra_data.
Status objects must implement get_metrics() for aggregation (BaseTestStatus provides default).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..test_status import BaseTestStatus
from .metrics_utils import aggregate_metrics_from_status_list

AggregateFunction = Callable[[List[BaseTestStatus]], Dict[str, float]]

logger = logging.getLogger(__name__)


class ReportContextStrategy(Protocol):
    """Protocol for objects that can supply report context (device, output_path, model_spec)."""

    device: Any
    output_path: Any
    model_spec: Any


@dataclass
class ReportContext:
    """
    Context data required to generate reports (Dependency Inversion).

    Passed into report generators so they do not depend on strategy/clients.
    """

    model_name: str
    device_name: str
    output_path: Path
    model_id: str
    hf_model_repo: Optional[str] = None

    @staticmethod
    def _device_name(device: Any) -> str:
        return device.name if hasattr(device, "name") else str(device)

    @staticmethod
    def _normalize_output_path(output_path: Any) -> Path:
        return output_path if isinstance(output_path, Path) else Path(output_path)

    @classmethod
    def from_strategy(cls, strategy: ReportContextStrategy) -> "ReportContext":
        """Build context from a strategy-like object (device, output_path, model_spec)."""
        spec = strategy.model_spec
        return cls(
            model_name=spec.model_name,
            device_name=cls._device_name(strategy.device),
            output_path=cls._normalize_output_path(strategy.output_path),
            model_id=spec.model_id,
            hf_model_repo=getattr(spec, "hf_model_repo", None),
        )

    def base_metadata(self) -> Dict[str, Any]:
        """Common report metadata from context."""
        return {
            "model": self.model_name,
            "device": self.device_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }


class ReportGenerator:
    """
    Generates benchmark and eval reports from ReportContext and status/extra data.

    Inject via BaseMediaStrategy(report_generator=...). If not provided, strategy uses ReportGenerator().
    """

    def __init__(self, aggregate_fn: Optional[AggregateFunction] = None):
        self._aggregate_fn = aggregate_fn or aggregate_metrics_from_status_list

    def generate_benchmark_report(
        self,
        context: ReportContext,
        status_list: List[BaseTestStatus],
        task_type: str,
        extra_benchmarks: Optional[Dict[str, Any]] = None,
        extra_top_level: Optional[Dict[str, Any]] = None,
        pre_aggregated: Optional[Dict[str, float]] = None,
    ) -> Optional[Path]:
        """
        Generate benchmark report from status list and context.

        If pre_aggregated is provided (e.g. from MetricsAggregator.result()), uses it
        and does not iterate status_list for aggregation (one less pass).

        Args:
            context: ReportContext with model_name, device_name, output_path, model_id
            status_list: List of test status objects (each provides get_metrics())
            task_type: Task type string (e.g. "image", "audio", "tts")
            extra_benchmarks: Optional client-specific benchmark fields to merge in
            extra_top_level: Optional client-specific top-level report keys
            pre_aggregated: Optional pre-computed metrics (num_requests + means); avoids second pass

        Returns:
            Path to written report file, or None if status_list is empty and no pre_aggregated
        """
        if not status_list and not pre_aggregated:
            logger.error(
                "Empty status list and no pre_aggregated, skipping benchmark report"
            )
            return None

        logger.info("Generating benchmark report...")
        base = {**context.base_metadata(), "task_type": task_type}
        if extra_top_level:
            base.update(extra_top_level)
        benchmarks = (
            pre_aggregated
            if pre_aggregated is not None
            else self._aggregate_fn(status_list)
        )
        if extra_benchmarks:
            benchmarks.update(extra_benchmarks)
        report_data = {**base, "benchmarks": benchmarks}
        return self._write_json_report(self._benchmark_filepath(context), report_data)

    def generate_eval_report(
        self,
        context: ReportContext,
        task_type: str,
        extra_data: Dict[str, Any],
    ) -> Path:
        """
        Generate eval report (single payload in a list) from context and extra data.

        Client provides extra_data with task_name, tolerance, score, accuracy_check, etc.
        This method adds model, device, timestamp, task_type and writes to eval path.

        Args:
            context: ReportContext (hf_model_repo used for eval subpath)
            task_type: Task type string (e.g. "image", "audio", "text_to_speech")
            extra_data: Client-specific eval fields (score, task_name, tolerance, ...)

        Returns:
            Path to written report file
        """
        logger.info("Generating eval report...")
        base = {**context.base_metadata(), "task_type": task_type}
        payload = {**base, **extra_data}
        report_data = [payload]
        return self._write_json_report(self._eval_filepath(context), report_data)

    def _write_json_report(self, filepath: Path, data: Any) -> Path:
        """Write JSON report to file, creating directories as needed."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Report generated: {filepath}")
        return filepath

    def _benchmark_filepath(self, context: ReportContext) -> Path:
        """Generate benchmark report filepath."""
        return context.output_path / f"benchmark_{context.model_id}_{time.time()}.json"

    def _eval_filepath(self, context: ReportContext) -> Path:
        """Generate eval report filepath."""
        repo_part = (context.hf_model_repo or "unknown").replace("/", "__")
        return (
            context.output_path
            / f"eval_{context.model_id}"
            / repo_part
            / f"results_{time.time()}.json"
        )
