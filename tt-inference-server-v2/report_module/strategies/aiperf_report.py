# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from report_module.base_strategy import ReportStrategy
from report_module.markdown.table_builder import get_markdown_table
from report_module.parsing.common import (
    deduplicate_by_config,
    extract_percentile_result,
)
from report_module.types import NOT_MEASURED_STR, ReportContext, ReportResult

logger = logging.getLogger(__name__)

METRIC_DEFINITIONS = (
    "**Metric Definitions:**\n"
    "> - **ISL**: Input Sequence Length (tokens)\n"
    "> - **OSL**: Output Sequence Length (tokens)\n"
    "> - **Concur**: Concurrent requests (batch size)\n"
    "> - **N**: Total number of requests\n"
    "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)\n"
    "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
    "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
    "> - **Output Tok/s**: Output token throughput\n"
    "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
    "> - **Req/s**: Request throughput\n"
)


def _percentile_table_markdown(rows: List[Dict[str, Any]], is_vlm: bool = False) -> str:
    """Build a percentile table (mean/median/p99) from raw result dicts."""
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
    ]
    if is_vlm:
        display_cols.extend(
            [
                ("image_height", "Image Height"),
                ("image_width", "Image Width"),
                ("images_per_prompt", "Images per Prompt"),
            ]
        )
    display_cols.extend(
        [
            ("num_requests", "N"),
            ("mean_ttft_ms", "TTFT Avg (ms)"),
            ("median_ttft_ms", "TTFT P50 (ms)"),
            ("p99_ttft_ms", "TTFT P99 (ms)"),
            ("mean_tpot_ms", "TPOT Avg (ms)"),
            ("median_tpot_ms", "TPOT P50 (ms)"),
            ("p99_tpot_ms", "TPOT P99 (ms)"),
            ("mean_e2el_ms", "E2EL Avg (ms)"),
            ("median_e2el_ms", "E2EL P50 (ms)"),
            ("p99_e2el_ms", "E2EL P99 (ms)"),
            ("output_token_throughput", "Output Tok/s"),
            ("total_token_throughput", "Total Tok/s"),
            ("request_throughput", "Req/s"),
        ]
    )

    display_dicts: List[Dict[str, str]] = []
    for row in rows:
        row_dict: Dict[str, str] = {}
        for col_name, header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                if col_name == "request_throughput":
                    row_dict[header] = f"{value:.4f}"
                elif col_name in ("output_token_throughput", "total_token_throughput"):
                    row_dict[header] = f"{value:.2f}"
                else:
                    row_dict[header] = f"{value:.1f}"
            else:
                row_dict[header] = str(value)
        display_dicts.append(row_dict)

    return get_markdown_table(display_dicts)


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

        release_str = f"### Benchmark Performance Results for {context.model_name} on {context.device_str}\n\n"

        if aiperf_text_results:
            release_str += "#### AIPerf Text Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf)\n\n"
            release_str += _percentile_table_markdown(aiperf_text_results) + "\n\n"

        if aiperf_vlm_results:
            release_str += "#### AIPerf VLM Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf)\n\n"
            release_str += (
                _percentile_table_markdown(aiperf_vlm_results, is_vlm=True) + "\n\n"
            )

        release_str += METRIC_DEFINITIONS

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
