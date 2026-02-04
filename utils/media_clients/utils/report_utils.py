# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Report generation for media clients.

Flow: strategy calls BaseMediaStrategy._generate_report(status_list);
base builds ReportContext, gets extras from strategy hooks, uses ReportGenerator.
Status objects must implement get_metrics() for aggregation (BaseTestStatus provides default).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..test_status import BaseTestStatus
from .metrics_utils import aggregate_metrics_from_status_list

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        aggregate_fn: Optional[
            Any
        ] = None,  # Callable[[List[BaseTestStatus]], Dict[str, float]]
    ):
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

        filepath = (
            context.output_path / f"benchmark_{context.model_id}_{time.time()}.json"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Benchmark report generated: {filepath}")
        return filepath

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

        repo_part = (context.hf_model_repo or "unknown").replace("/", "__")
        filepath = (
            context.output_path
            / f"eval_{context.model_id}"
            / repo_part
            / f"results_{time.time()}.json"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Eval report generated: {filepath}")
        return filepath
