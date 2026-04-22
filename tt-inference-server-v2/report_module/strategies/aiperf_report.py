# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from report_module.base_strategy import ReportStrategy
from report_module.markdown.visualizer import MarkdownVisualizer
from report_module.parsing.common import (
    deduplicate_by_config,
    extract_percentile_result,
)
from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)


class AiPerfStrategy(ReportStrategy):
    """AIPerf detailed benchmark report — fully self-contained."""

    name = "aiperf_benchmarks"

    def is_applicable(self, context: ReportContext) -> bool:
        benchmarks_dir = f"{context.workflow_log_dir}/benchmarks_output"
        model_id = context.model_spec.model_id
        return len(glob(f"{benchmarks_dir}/aiperf_benchmark_{model_id}_*.json")) > 0

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        benchmarks_dir = f"{context.workflow_log_dir}/benchmarks_output"
        model_id = context.model_spec.model_id

        aiperf_files = deduplicate_by_config(
            glob(f"{benchmarks_dir}/aiperf_benchmark_{model_id}_*.json")
        )
        logger.info(f"AIPerf: {len(aiperf_files)} files after dedup")

        if not aiperf_files:
            return {self.name: ReportResult.empty(self.name)}

        aiperf_text_files = [f for f in aiperf_files if "images" not in Path(f).name]
        aiperf_vlm_files = [f for f in aiperf_files if "images" in Path(f).name]

        aiperf_text_results = self._parse_files(aiperf_text_files, "aiperf")
        aiperf_vlm_results = self._parse_files(aiperf_vlm_files, "aiperf", is_vlm=True)

        if not aiperf_text_results and not aiperf_vlm_results:
            return {self.name: ReportResult.empty(self.name)}

        aiperf_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))
        aiperf_vlm_results.sort(
            key=lambda x: (
                x["isl"],
                x["osl"],
                x["concurrency"],
                x.get("image_height", 0),
                x.get("image_width", 0),
            )
        )

        release_str = MarkdownVisualizer.build_percentile_benchmark_markdown(
            tool_label="AIPerf",
            tool_url="https://github.com/ai-dynamo/aiperf",
            text_rows=aiperf_text_results,
            vlm_rows=aiperf_vlm_results,
            model_name=context.model_name,
            device_str=context.device_str,
            section_header="Benchmark Performance Results",
        )

        all_results = aiperf_text_results + aiperf_vlm_results
        result = ReportResult(
            name=self.name,
            markdown=release_str,
            data=all_results,
            md_filename=f"aiperf_benchmark_display_{context.report_id}.md",
        )
        return {self.name: result}

    @staticmethod
    def _parse_files(
        files: List[str], source: str, is_vlm: bool = False
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for f in sorted(files):
            result = extract_percentile_result(f, source, is_vlm=is_vlm)
            if result:
                results.append(result)
        return results
